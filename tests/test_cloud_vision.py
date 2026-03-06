"""
tests/test_cloud_vision.py — unit tests for workers/cloud_vision.py.

Verifies that our helper functions correctly normalise Cloud Vision data
structures.  We can't hit the real API during unit tests, so we construct
minimal protobuf objects instead.
"""
from __future__ import annotations

import pytest

from workers import cloud_vision
from models.schemas import BoundingBox, Polygon, SemanticMap, WordEntry

# import protobuf types so we can synthesise a fake response
from google.cloud.vision_v1.types import (
    AnnotateImageResponse,
    TextAnnotation,
    Page,
    Block,
    Paragraph,
    Word,
    Symbol,
    BoundingPoly,
    Vertex,
)


def _make_word_response(text: str, conf: float, verts: list[tuple[int, int]]):
    """Helper to build a Word protobuf with the given text, confidence and vertices."""
    symbols = [Symbol(text=c) for c in text]
    poly = BoundingPoly(vertices=[Vertex(x=x, y=y) for x, y in verts])
    return Word(symbols=symbols, confidence=conf, bounding_box=poly)


def test_bbox_from_vertices_basic():
    verts = [Vertex(x=10, y=20), Vertex(x=15, y=20), Vertex(x=15, y=25), Vertex(x=10, y=25)]
    bb = cloud_vision._bbox_from_vertices(verts)
    assert isinstance(bb, BoundingBox)
    assert bb.x == 10
    assert bb.y == 20
    assert bb.w == 5
    assert bb.h == 5


def test_bbox_from_vertices_missing_coordinates():
    verts = [Vertex(), Vertex(x=5), Vertex(y=7)]  # some fields default to None/0
    bb = cloud_vision._bbox_from_vertices(verts)
    # should still produce a valid box with non-negative dimensions
    assert bb.w >= 0
    assert bb.h >= 0


def test_polygon_from_vertices():
    verts = [Vertex(x=1, y=2), Vertex(x=3, y=4)]
    poly = cloud_vision._polygon_from_vertices(verts)
    assert isinstance(poly, Polygon)
    assert poly.points == [(1, 2), (3, 4)]


def test_parse_document_text_annotation_empty():
    resp = AnnotateImageResponse()
    sm = cloud_vision._parse_document_text_annotation(resp, label_id="foo", img_w=100, img_h=200)
    assert isinstance(sm, SemanticMap)
    assert sm.label_id == "foo"
    assert sm.image_width == 100
    assert sm.image_height == 200
    assert sm.words == []
    assert sm.regions == []
    # raw_response might be empty dict if the protobuf has no fields
    assert isinstance(sm.raw_response, dict)


def test_parse_document_text_annotation_with_words():
    word1 = _make_word_response("Hi", 0.9, [(0, 0), (10, 0), (10, 10), (0, 10)])
    word2 = _make_word_response("Skip", 0.4, [(20, 20), (30, 20), (30, 30), (20, 30)])
    para = Paragraph(words=[word1, word2])
    block = Block(paragraphs=[para])
    page = Page(blocks=[block])
    text_ann = TextAnnotation(pages=[page])
    resp = AnnotateImageResponse(full_text_annotation=text_ann)

    sm = cloud_vision._parse_document_text_annotation(resp, label_id="bar", img_w=50, img_h=60)
    # only first word should make it past the confidence threshold (default 0.5)
    assert len(sm.words) == 1
    entry = sm.words[0]
    assert isinstance(entry, WordEntry)
    assert entry.text == "Hi"
    assert entry.confidence == pytest.approx(0.9)
    assert entry.bbox.x == 0 and entry.bbox.y == 0
    assert entry.polygon.points[0] == (0, 0)

    # verify that raw_response contains something resembling the message
    assert "fullTextAnnotation" in sm.raw_response or sm.raw_response == {}
