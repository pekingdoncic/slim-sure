from pathlib import Path

import cv2
import numpy as np

EXP = Path(__file__).resolve().parents[2]
ROOT = EXP.parents[1]
DATA = ROOT / "data/TexturePoorSfM_dataset"
VAL = {"1012", "1026", "1029"}
TEST = set("1000 1002 1003 1005 1006 1007 1008 1010 1014 1021 1024 1025 1027 1030".split())


def read_image(path):
    image = cv2.imread(str(ROOT / path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise OSError(f"unreadable image {path}")
    return image


def preprocess(gray, side):
    h, w = gray.shape
    ratio = side / max(h, w)
    wh = (round(w * ratio), round(h * ratio))
    resized = cv2.resize(gray, wh, interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((side, side), np.float32)
    canvas[:wh[1], :wh[0]] = resized.astype(np.float32) / 255.
    mask = np.zeros_like(canvas, dtype=bool)
    mask[:wh[1], :wh[0]] = True
    return canvas, mask, np.array([w / wh[0], h / wh[1]], np.float32)
