"""Calibration uses the same training visits on both sides at each evaluation resolution."""
import numpy as np
import torch
from src.texture_budget import local_texture, TEXTURE_SCHEMA
from .dataset_adapter import TrainPool
from ..utils.run_io import read, write


def energy(d,k):
    m = d["mask"+k]
    v,c = local_texture(d["image"+k],d["valid_mask"+k],m.shape[-2:])
    return v.flatten(),m.flatten() & (c.flatten()>0) & v.flatten().isfinite()


def calibrate(base,cache,directory,side,cfg):
    dataset = TrainPool(base,cache,side,cfg)
    n = cfg["texture_calibration_pairs"]
    values,ids = [],[]
    for ix in np.linspace(0,dataset.size-1,n,dtype=np.int64):
        item = dataset[int(ix)]
        d = {k:v[None] for k,v in item.items() if isinstance(v,torch.Tensor)}
        ids.append(item["pair_id"])
        for k in ("0","1"):
            v,ok = energy(d,k)
            values.append(v[ok].numpy())
    v = np.concatenate(values)
    result = dict(schema=TEXTURE_SCHEMA,side=side,num_grids=len(v),visits=n,train_pair_ids=ids,
                  threshold=float(np.quantile(v,1/3)),quantile="linear",window_px=16,stride=8)
    write(directory/f"texture_R{side}.json",result)
    return result


def threshold(directory,side):
    item = read(directory/f"texture_R{side}.json")
    if item["side"]!=side or item["schema"]!=TEXTURE_SCHEMA:
        raise ValueError("texture calibration mismatch")
    return item["threshold"]
