def bbox_to_dict(b):
    return {
        "x": b.x,
        "y": b.y,
        "w": b.w,
        "h": b.h,
    }

def bboxes_to_dicts(boxes):
    return [bbox_to_dict(b) for b in boxes]
