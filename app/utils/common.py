import numpy as np

def crop(img: np.ndarray, box):
    if box is None:
        return None
    x, y, w, h = box.to_xywh()
    h_img, w_img = img.shape[:2]

    x = max(0, min(x, w_img - 1))
    y = max(0, min(y, h_img - 1))
    w = max(1, min(w, w_img - x))
    h = max(1, min(h, h_img - y))

    return img[y:y+h, x:x+w]