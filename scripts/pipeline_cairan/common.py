"""Fungsi bersama eksperimen cairan."""
import cv2, numpy as np

S = 224


def fold_crop(im, x1, y1, x2, y2, pad=0.04):
    """Crop kotak botol -> sumbu panjang horizontal -> potong dua & tumpuk -> pad abu 114 ke 224x224 (sama dgn s02 'fold')."""
    H, W = im.shape[:2]; w, h = x2 - x1, y2 - y1
    x1, x2 = max(0, int(x1 - pad * w)), min(W, int(x2 + pad * w)); y1, y2 = max(0, int(y1 - pad * h)), min(H, int(y2 + pad * h))
    c = im[y1:y2, x1:x2]
    if c.shape[0] > c.shape[1]:
        c = cv2.rotate(c, cv2.ROTATE_90_CLOCKWISE)
    hw = c.shape[1] // 2; c = cv2.vconcat([c[:, :hw], c[:, hw:hw * 2]])
    ch, cw = c.shape[:2]; s = S / max(ch, cw)
    c = cv2.resize(c, (max(1, round(cw * s)), max(1, round(ch * s))), interpolation=cv2.INTER_AREA)
    out = np.full((S, S, 3), 114, np.uint8); oy, ox = (S - c.shape[0]) // 2, (S - c.shape[1]) // 2
    out[oy:oy + c.shape[0], ox:ox + c.shape[1]] = c
    return out
