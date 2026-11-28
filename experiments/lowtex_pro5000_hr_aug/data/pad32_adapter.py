"""Network coordinates have pixel centers at n+0.5. OpenCV cameras use centers at n."""
from dataclasses import dataclass
from functools import lru_cache
import cv2
import numpy as np
import torch


@dataclass(frozen=True)
class ResizePlan:
    h: int
    w: int
    rh: int
    rw: int
    side: int

    @classmethod
    def make(cls, shape, side, valid_1216=1200):
        h, w = shape[:2]
        if side <= 0 or side % 32 or min(h, w) < 1:
            raise ValueError((shape, side))
        active = valid_1216 if side == 1216 else side
        if not 0 < active <= side:
            raise ValueError(active)
        return cls(h, w, max(1, round(h*active/max(h,w))), max(1, round(w*active/max(h,w))), side)

    @property
    def scale(self):
        return np.array([self.rw/self.w, self.rh/self.h], np.float64)

    def to_original(self, points):
        return np.asarray(points) / self.scale - 0.5

    def to_network(self, points):
        return (np.asarray(points) + 0.5) * self.scale

    def intrinsics(self, k):
        sx, sy = self.scale
        return np.array([[sx, 0, sx/2], [0, sy, sy/2], [0, 0, 1.]]) @ k

    def canvas(self, array, depth=False):
        method = cv2.INTER_NEAREST_EXACT if depth else cv2.INTER_LINEAR
        small = cv2.resize(array, (self.rw, self.rh), interpolation=method)
        out = np.zeros((self.side,self.side), np.float32)
        out[:self.rh,:self.rw] = np.where(np.isfinite(small) & (small > 0), small, 0) if depth else small / 255.
        return torch.from_numpy(out)

    def mask(self):
        valid = torch.zeros((self.side,self.side), dtype=torch.bool)
        valid[:self.rh,:self.rw] = True
        return valid


@lru_cache(maxsize=1)
def active_1216():
    from ..utils.run_io import config
    return config("P1")["valid_long_side_1216"]


def prepare(images, cameras, side, depths=None, valid_1216=None):
    valid_1216 = active_1216() if valid_1216 is None else valid_1216
    data = {}
    plans = []
    for i, gray in enumerate(images):
        s = str(i)
        plan = ResizePlan.make(gray.shape,side,valid_1216)
        plans.append(plan)
        data["image"+s] = plan.canvas(gray)[None]
        data["valid_mask"+s] = plan.mask()
        data["mask"+s] = data["valid_mask"+s][::8,::8].contiguous()
        data["scale"+s] = torch.ones(2)  # native windows now live in input coordinates
        data["K"+s] = torch.as_tensor(plan.intrinsics(cameras[i]),dtype=torch.float32)
        if depths is not None:
            data["depth"+s] = plan.canvas(depths[i].astype(np.float32),depth=True)
    return data, plans
