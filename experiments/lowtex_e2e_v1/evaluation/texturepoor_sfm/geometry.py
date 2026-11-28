import numpy as np


def relative(w2c0, w2c1):
    return np.asarray(w2c1) @ np.linalg.inv(w2c0)


def transformed_intrinsics(k, sx, sy, left=0, top=0):
    return np.array([[sx, 0., left], [0., sy, top], [0., 0., 1.]]) @ k


def angle_deg(dot):
    return float(np.degrees(np.arccos(np.clip(dot, -1., 1.))))
