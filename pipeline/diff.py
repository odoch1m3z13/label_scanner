"""
pipeline/diff.py — Fuzzy LCS semantic diff engine.

Takes two lists of WordEntry (ref and user, both in reference coordinate space)
and returns a list of DiffEntry describing matches, modifications, removals,
additions, and reflow runs.

All functions are pure — no I/O, no image processing, no side effects.

Design:
  The pipeline runs in five stages:

  1. normalise_fused_tokens  — split "100g" vs "100"+"g" mismatches before LCS
  2. fuzzy_lcs_diff          — DP LCS using per-token similarity thresholds
  3. spatial_second_pass     — promote spatially co-located removed+added → modified
  4. detect_reflow_runs      — collapse high-char-overlap runs into unmatched_run
  5. suppress_long_runs      — collapse noisy long non-match runs into unmatched_run

LCS matching thresholds (from config):
  Tokens ≤ FUZZY_EXACT_MAX_LEN (2 chars)     → must match exactly (threshold 1.0)
  Tokens ≤ FUZZY_SHORT_TOKEN_MAX_LEN (4 ch)  → threshold 0.80 (fewer edits tolerated)
  All other tokens                            → threshold 0.75

DiffType semantics:
  match         — text is effectively identical (cleaned similarity == 1.0)
  modified      — LCS-paired but texts differ  (similarity in [threshold, 1.0))
  removed       — present in ref, absent in user
  added         — absent in ref, present in user
  color_changed — matched pair flagged by pipeline/colour.py (added post-diff)
  unmatched_run — collapsed noisy run (from reflow or long-run suppression)

Spatial second pass (fix for v4.4 regression):
  The v4.4 bug used Y-only proximity, which paired words in different columns
  that happened to share a y-coordinate, breaking reading order, collapsing
  the text mask, and triggering full-screen false-positive tamper contours.
  The fix uses 2D normalised Euclidean distance — both X and Y must be within
  SPATIAL_TOLERANCE (0.12) of each other in the [0,1]×[0,1] coordinate space.
"""
from __future__ import annotations

import logging
import math

from config import (
    FUZZY_EXACT_MAX_LEN,
    FUZZY_MATCH_THRESHOLD,
    FUZZY_SHORT_THRESHOLD,
    FUZZY_SHORT_TOKEN_MAX_LEN,
    MAX_CONSECUTIVE_FLAGS,
    MIN_TOKEN_LEN,
    REFLOW_CHAR_OVERLAP,
    REFLOW_WINDOW,
    SPATIAL_TOLERANCE,
)
from models.schemas import DiffEntry, DiffType, WordEntry
from pipeline.words import clean, sort_reading_order

log = logging.getLogger(__name__)


# =============================================================================
#  LEVENSHTEIN AND SIMILARITY
# =============================================================================

def levenshtein(a: str, b: str) -> int:
    """
    Classic O(m·n) dynamic-programming Levenshtein edit distance.
    Operates on raw strings (callers should pass cleaned strings for
    normalised comparison).
    """
    if a == b:  return 0
    if not a:   return len(b)
    if not b:   return len(a)
    prev = list(range(len(b) + 1))
    for ca in a:
        curr = [prev[0] + 1]
        for j, cb in enumerate(b, 1):
            curr.append(min(prev[j] + 1, curr[-1] + 1, prev[j - 1] + (ca != cb)))
        prev = curr
    return prev[-1]


def similarity(a: str, b: str) -> float:
    """
    Normalised character edit similarity in [0.0, 1.0].

    Cleans both strings (lowercase alphanumeric only) before comparison so
    that "Protein:" and "protein" are considered identical (1.0) — punctuation
    and case differences on labels are not genuine content changes.

    Returns:
        1.0 = identical after cleaning.
        0.0 = completely different, or one/both strings are empty.
    """
    ca, cb = clean(a), clean(b)
    if not ca and not cb:  return 1.0
    if not ca or  not cb:  return 0.0
    return 1.0 - levenshtein(ca, cb) / max(len(ca), len(cb))


def match_threshold(text: str) -> float:
    """
    Per-token similarity threshold for LCS pairing, adjusted for token length.

    Short tokens are held to a higher standard because a single edit to a
    2-character token ("to"→"do") represents a 50% change and is genuinely
    significant. Longer tokens have more edit budget.
    """
    n = len(clean(text))
    if n <= FUZZY_EXACT_MAX_LEN:        return 1.00   # "to", "mg" etc — must be exact
    if n <= FUZZY_SHORT_TOKEN_MAX_LEN:  return FUZZY_SHORT_THRESHOLD   # 0.80
    return FUZZY_MATCH_THRESHOLD                                        # 0.75


# =============================================================================
#  FUSED TOKEN NORMALISATION
# =============================================================================

def _split_word_at(word: WordEntry, pivot: int) -> tuple[WordEntry, WordEntry]:
    """
    Split a WordEntry into two at character position `pivot`.
    The bounding box is split proportionally by character count.
    The polygon points are copied to both halves (approximate — exact split
    would require per-character geometry which Cloud Vision doesn't provide).
    """
    text_a = word.text[:pivot]
    text_b = word.text[pivot:]
    frac   = pivot / max(1, len(word.text))

    from models.schemas import BoundingBox, Polygon
    split_x = word.bbox.x + int(word.bbox.w * frac)

    bbox_a = BoundingBox(x=word.bbox.x, y=word.bbox.y,
                         w=max(1, split_x - word.bbox.x), h=word.bbox.h)
    bbox_b = BoundingBox(x=split_x, y=word.bbox.y,
                         w=max(1, word.bbox.x + word.bbox.w - split_x), h=word.bbox.h)

    poly_a = Polygon(points=[
        (word.bbox.x, word.bbox.y), (split_x, word.bbox.y),
        (split_x, word.bbox.y + word.bbox.h), (word.bbox.x, word.bbox.y + word.bbox.h),
    ])
    poly_b = Polygon(points=[
        (split_x, word.bbox.y), (word.bbox.x + word.bbox.w, word.bbox.y),
        (word.bbox.x + word.bbox.w, word.bbox.y + word.bbox.h), (split_x, word.bbox.y + word.bbox.h),
    ])

    wa = WordEntry(text=text_a, confidence=word.confidence,
                   bbox=bbox_a, polygon=poly_a,
                   block_id=word.block_id, para_id=word.para_id)
    wb = WordEntry(text=text_b, confidence=word.confidence,
                   bbox=bbox_b, polygon=poly_b,
                   block_id=word.block_id, para_id=word.para_id)
    return wa, wb


def normalise_fused_tokens(
    wa: list[WordEntry],
    wb: list[WordEntry],
) -> tuple[list[WordEntry], list[WordEntry]]:
    """
    Detect and split fused tokens so the LCS has a fair comparison surface.

    A fused token occurs when one OCR pass tokenises "100g" as a single token
    while the other returns ["100", "g"]. Without normalisation the LCS sees
    a removal ("100g") and two additions ("100", "g") and raises false flags.

    Strategy — scan both directions:
      Pass A: for each word in wa, check if clean(wa[i]) equals the
              concatenation of clean(wb[j]) + clean(wb[j+1]) for any j.
              If so, split wa[i] at the appropriate pivot.
      Pass B: same in the other direction (wb word vs two consecutive wa words).

    Only exact concatenation matches are split — approximate matches are left
    for the LCS + spatial pass to handle.
    """
    def _expand(src: list[WordEntry], other: list[WordEntry]) -> list[WordEntry]:
        """Split tokens in `src` that match two consecutive tokens in `other`."""
        other_clean = [clean(w.text) for w in other]
        result: list[WordEntry] = []
        for word in src:
            cw = clean(word.text)
            split_found = False
            # Only attempt split for tokens that are at least 2 chars combined
            for j in range(len(other_clean) - 1):
                ca, cb = other_clean[j], other_clean[j + 1]
                if ca and cb and cw == ca + cb:
                    # Split at len(original text[: first part length])
                    # Use character-level mapping from clean back to original
                    pivot = _clean_pivot(word.text, len(ca))
                    if pivot > 0:
                        part_a, part_b = _split_word_at(word, pivot)
                        result.extend([part_a, part_b])
                        split_found = True
                        break
            if not split_found:
                result.append(word)
        return result

    wa2 = _expand(wa, wb)
    wb2 = _expand(wb, wa)
    return wa2, wb2


def _clean_pivot(text: str, clean_pivot: int) -> int:
    """
    Map a pivot position in the cleaned string back to the original string.
    Returns the character index in `text` after which `clean_pivot` clean
    characters have been consumed.
    """
    seen = 0
    for i, ch in enumerate(text):
        if clean(ch):   # non-empty after cleaning → a kept char
            seen += 1
        if seen == clean_pivot:
            return i + 1
    return len(text)


# =============================================================================
#  FUZZY LCS DIFF
# =============================================================================

def fuzzy_lcs_diff(
    wa: list[WordEntry],
    wb: list[WordEntry],
) -> list[DiffEntry]:
    """
    Fuzzy Longest Common Subsequence diff between two WordEntry lists.

    DP phase:
      Build an (m+1) × (n+1) table where dp[i][j] = length of the LCS
      considering wa[:i] and wb[:j]. A diagonal move is taken when
      similarity(wa[i], wb[j]) ≥ match_threshold(wa[i].text).

    Traceback phase:
      Reconstruct the alignment from dp[m][n] back to dp[0][0]:
        - Diagonal move: words are paired.
          sim == 1.0  → DiffType.match    (effectively identical text)
          sim <  1.0  → DiffType.modified (text differs but words are aligned)
        - Up move:   wa[i] not matched → DiffType.removed
        - Left move: wb[j] not matched → DiffType.added

    Short tokens (≤ MIN_TOKEN_LEN chars after cleaning) are included in the
    output but filtered from the diff count summary in main.py so that
    punctuation and particles do not inflate the diff count.

    Returns:
        List of DiffEntry in reading order (result of traceback reversal).
    """
    m, n = len(wa), len(wb)

    if m == 0 and n == 0:
        return []

    # ── Build similarity matrix ───────────────────────────────────────────────
    # sim[i][j] = similarity between wa[i] and wb[j]
    sim: list[list[float]] = [
        [similarity(wa[i].text, wb[j].text) for j in range(n)]
        for i in range(m)
    ]

    # ── DP table ─────────────────────────────────────────────────────────────
    dp: list[list[int]] = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        thresh = match_threshold(wa[i - 1].text)
        for j in range(1, n + 1):
            if sim[i - 1][j - 1] >= thresh:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    # ── Traceback ─────────────────────────────────────────────────────────────
    result: list[DiffEntry] = []
    i, j = m, n

    while i > 0 or j > 0:
        if i > 0 and j > 0:
            thresh = match_threshold(wa[i - 1].text)
            s      = sim[i - 1][j - 1]

            if s >= thresh and dp[i][j] == dp[i - 1][j - 1] + 1:
                # Diagonal — words are paired
                if s >= 1.0:
                    result.append(DiffEntry(
                        type=DiffType.match,
                        ref=wa[i - 1],
                        user=wb[j - 1],
                    ))
                else:
                    result.append(DiffEntry(
                        type=DiffType.modified,
                        ref=wa[i - 1],
                        user=wb[j - 1],
                        similarity=round(s, 4),
                    ))
                i -= 1
                j -= 1

            elif dp[i - 1][j] >= dp[i][j - 1]:
                result.append(DiffEntry(type=DiffType.removed, word=wa[i - 1]))
                i -= 1
            else:
                result.append(DiffEntry(type=DiffType.added, word=wb[j - 1]))
                j -= 1

        elif i > 0:
            result.append(DiffEntry(type=DiffType.removed, word=wa[i - 1]))
            i -= 1
        else:
            result.append(DiffEntry(type=DiffType.added, word=wb[j - 1]))
            j -= 1

    result.reverse()
    return result


# =============================================================================
#  SPATIAL SECOND PASS
# =============================================================================

def _word_centre_norm(entry: DiffEntry, ref_w: int, ref_h: int) -> tuple[float, float] | None:
    """
    Extract the normalised (0–1) centre of a removed or added word.
    Both removed and added words use ref coordinate space after project_words.
    Returns None if the entry carries no spatial information.
    """
    word = entry.word
    if word is None:
        return None
    cx = (word.bbox.x + word.bbox.w / 2) / max(1, ref_w)
    cy = (word.bbox.y + word.bbox.h / 2) / max(1, ref_h)
    return cx, cy


def spatial_second_pass(
    diff:   list[DiffEntry],
    ref_h:  int,
    ref_w:  int,
    user_h: int,
    user_w: int,
) -> list[DiffEntry]:
    """
    Promote spatially co-located (removed, added) pairs to 'modified'.

    After LCS, words that were not matched by text similarity but occupy the
    same spatial position (e.g. "5g" → "6g", a nutrition value edit) appear
    as a removed + added pair. The spatial pass detects these and merges them
    into a single modified entry with similarity scored from the text.

    Algorithm:
      1. Collect indices of all removed entries.
      2. Collect indices of all added entries.
      3. For each removed entry, find the nearest added entry in normalised
         2D Euclidean space. Both coordinates must be within SPATIAL_TOLERANCE.
      4. Promote matched pairs to modified, remove originals.
      5. Each added entry may only be consumed once (greedy closest-first).

    Uses ref dimensions for normalisation since both removed words (ref space)
    and added words (projected user space → ref space via project_words) are
    expressed in the same coordinate system after pipeline/align.py projection.

    v4.4 regression fix:
      The v4.4 implementation compared Y coordinate only. This paired words in
      different columns that shared a y row — "Left" (x=50) and "Right" (x=450)
      at the same y would be incorrectly merged. The 2D Euclidean check prevents
      this: at normalised x-distance (450-50)/500 = 0.8 >> SPATIAL_TOLERANCE=0.12.
    """
    # Index removed and added entries
    removed_idx: list[int] = [
        i for i, d in enumerate(diff) if d.type == DiffType.removed and d.word is not None
    ]
    added_idx: list[int] = [
        i for i, d in enumerate(diff) if d.type == DiffType.added and d.word is not None
    ]

    if not removed_idx or not added_idx:
        return diff

    # Precompute normalised centres
    removed_centres = [_word_centre_norm(diff[i], ref_w, ref_h) for i in removed_idx]
    added_centres   = [_word_centre_norm(diff[i], ref_w, ref_h) for i in added_idx]

    consumed_added: set[int] = set()
    promotions: dict[int, int] = {}   # removed_idx_pos → added_idx_pos

    # Greedy nearest-neighbour matching (closest pair first)
    candidates: list[tuple[float, int, int]] = []  # (dist, ri, ai)
    for ri, rc in enumerate(removed_centres):
        if rc is None:
            continue
        for ai, ac in enumerate(added_centres):
            if ac is None:
                continue
            dist = math.sqrt((rc[0] - ac[0]) ** 2 + (rc[1] - ac[1]) ** 2)
            if dist <= SPATIAL_TOLERANCE:
                candidates.append((dist, ri, ai))

    candidates.sort(key=lambda t: t[0])   # closest pairs first

    for dist, ri, ai in candidates:
        if ri in promotions or ai in consumed_added:
            continue
        promotions[ri]    = ai
        consumed_added.add(ai)

    if not promotions:
        return diff

    # Build updated diff list
    # Mark promoted indices; insert modified entry at position of the removed entry
    added_promoted: set[int] = {added_idx[ai] for ai in consumed_added}
    result: list[DiffEntry] = []

    for pos, entry in enumerate(diff):
        # Skip added entries that were consumed by promotion
        if pos in added_promoted:
            continue

        # Check if this removed entry is being promoted
        if entry.type == DiffType.removed:
            try:
                ri = removed_idx.index(pos)
            except ValueError:
                result.append(entry)
                continue

            if ri in promotions:
                ai          = promotions[ri]
                added_entry = diff[added_idx[ai]]
                ref_word    = entry.word
                user_word   = added_entry.word
                sim_score   = similarity(ref_word.text, user_word.text) if ref_word and user_word else 0.0
                result.append(DiffEntry(
                    type=DiffType.modified,
                    ref=ref_word,
                    user=user_word,
                    similarity=round(sim_score, 4),
                ))
            else:
                result.append(entry)
        else:
            result.append(entry)

    return result


# =============================================================================
#  REFLOW DETECTION
# =============================================================================

def detect_reflow_runs(diff: list[DiffEntry]) -> list[DiffEntry]:
    """
    Collapse consecutive removed/added windows that share most of their
    characters into a single 'unmatched_run' entry.

    A reflow artefact occurs when the OCR engine re-wraps the same text
    differently between the reference and user scan — the characters are
    all present but distributed across different word boundaries. This is
    not a genuine content change and should not appear as dozens of
    individual removed/added entries.

    Detection: within a sliding window of size REFLOW_WINDOW, collect the
    character sets of all removed words and all added words. If the
    intersection / union ratio ≥ REFLOW_CHAR_OVERLAP, the window is a
    reflow — collapse it into a single unmatched_run.

    Only windows where BOTH sides have content (at least one removed AND
    one added entry) are candidates.
    """
    result: list[DiffEntry] = []
    i = 0
    n = len(diff)

    while i < n:
        entry = diff[i]

        # Only start a window scan on a removed or added entry
        if entry.type not in (DiffType.removed, DiffType.added):
            result.append(entry)
            i += 1
            continue

        # Collect the window
        window = [diff[j] for j in range(i, min(i + REFLOW_WINDOW, n))
                  if diff[j].type in (DiffType.removed, DiffType.added)]

        removed_in_window = [e for e in window if e.type == DiffType.removed]
        added_in_window   = [e for e in window if e.type == DiffType.added]

        if not removed_in_window or not added_in_window:
            result.append(entry)
            i += 1
            continue

        removed_chars = set(clean("".join(
            e.word.text for e in removed_in_window if e.word
        )))
        added_chars = set(clean("".join(
            e.word.text for e in added_in_window if e.word
        )))

        if not removed_chars or not added_chars:
            result.append(entry)
            i += 1
            continue

        overlap = len(removed_chars & added_chars) / max(1, len(removed_chars | added_chars))

        if overlap >= REFLOW_CHAR_OVERLAP:
            # Collapse the window into a single unmatched_run
            window_size = len(window)
            collapsed_items = [
                {"type": e.type.value, "text": e.word.text if e.word else ""}
                for e in window
            ]
            result.append(DiffEntry(
                type=DiffType.unmatched_run,
                items=collapsed_items,
                count=window_size,
            ))
            # Skip past all entries that make up this window in the original diff
            consumed = 0
            j = i
            while j < n and consumed < window_size:
                if diff[j].type in (DiffType.removed, DiffType.added):
                    consumed += 1
                j += 1
            i = j
        else:
            result.append(entry)
            i += 1

    return result


# =============================================================================
#  LONG-RUN SUPPRESSION
# =============================================================================

def suppress_long_runs(diff: list[DiffEntry]) -> list[DiffEntry]:
    """
    Collapse runs of ≥ MAX_CONSECUTIVE_FLAGS consecutive non-match entries
    into a single 'unmatched_run' DiffEntry.

    This prevents a large block of changed text (e.g. a whole paragraph
    rewritten) from producing hundreds of individual diff entries that
    overwhelm the frontend and the diff count summary.

    'match' entries reset the consecutive counter. 'modified', 'removed',
    'added', and existing 'unmatched_run' entries all count toward the run.

    The collapsed entry carries:
      items: list of the original entries serialised as dicts
      count: number of original entries in the run
    """
    result: list[DiffEntry] = []
    run:    list[DiffEntry] = []

    def _flush_run() -> None:
        if len(run) >= MAX_CONSECUTIVE_FLAGS:
            result.append(DiffEntry(
                type=DiffType.unmatched_run,
                items=[{"type": e.type.value,
                        "text": (e.word or e.ref or e.user or object()).text
                                if hasattr((e.word or e.ref or e.user or object()), "text") else ""}
                       for e in run],
                count=len(run),
            ))
        else:
            result.extend(run)
        run.clear()

    for entry in diff:
        if entry.type == DiffType.match:
            _flush_run()
            result.append(entry)
        else:
            run.append(entry)

    _flush_run()   # flush any trailing run
    return result


# =============================================================================
#  ORCHESTRATOR
# =============================================================================

def run(
    ref_words:  list[WordEntry],
    user_words: list[WordEntry],
    ref_h:      int,
    ref_w:      int,
    user_h:     int,
    user_w:     int,
) -> list[DiffEntry]:
    """
    Full diff pipeline — five stages in sequence.

    Stage 1 — normalise_fused_tokens:
        Detects and splits tokens like "100g" in one list when the other
        has ["100", "g"]. Gives the LCS a fair comparison surface.

    Stage 2 — sort_reading_order:
        Ensures both word lists are in top-to-bottom, left-to-right order
        before LCS. Cloud Vision output is usually ordered but may have
        minor deviations on complex label layouts.

    Stage 3 — fuzzy_lcs_diff:
        DP LCS produces the base match/modified/removed/added sequence.

    Stage 4 — spatial_second_pass:
        Promotes spatially co-located removed+added pairs to modified.
        Handles content edits that were too dissimilar for the LCS threshold
        (e.g. a nutrition value changed from "5g" to "10g").

    Stage 5 — detect_reflow_runs:
        Collapses OCR reflow artefacts (same characters, different word
        boundaries) into unmatched_run entries.

    Stage 6 — suppress_long_runs:
        Collapses noisy long non-match runs into unmatched_run entries.

    Args:
        ref_words:  Reference word list in ref coordinate space.
        user_words: User word list projected to ref coordinate space
                    (via pipeline/align.project_words).
        ref_h/w:    Reference image pixel dimensions.
        user_h/w:   Original user image pixel dimensions (pre-projection).

    Returns:
        Flat list of DiffEntry in reading order.
    """
    log.info(
        "diff.run: ref=%d words  user=%d words  ref=%dx%d",
        len(ref_words), len(user_words), ref_w, ref_h,
    )

    wa, wb = normalise_fused_tokens(ref_words, user_words)
    wa     = sort_reading_order(wa)
    wb     = sort_reading_order(wb)

    result = fuzzy_lcs_diff(wa, wb)
    result = spatial_second_pass(result, ref_h, ref_w, user_h, user_w)
    result = detect_reflow_runs(result)
    result = suppress_long_runs(result)

    counts = {t.value: sum(1 for d in result if d.type == t) for t in DiffType}
    log.info("diff.run complete: %s", counts)

    return result