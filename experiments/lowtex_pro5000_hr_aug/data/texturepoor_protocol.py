"""Verify the published split and the frozen two-view adaptation against the server copy."""
from collections import Counter
from functools import lru_cache
from pathlib import Path
import numpy as np
from .dataset_adapter import manifest

VAL={"1012","1026","1029"}
TEST=set("1000 1002 1003 1005 1006 1007 1008 1010 1014 1021 1024 1025 1027 1030".split())


@lru_cache(maxsize=8)
def verify(root,formal):
    rows=manifest("texturepoor_test_pairs" if formal else "texturepoor_val_pairs")
    scenes=TEST if formal else VAL
    actual={d.name for d in root.iterdir() if d.is_dir()}
    missing=scenes-actual
    if missing:
        raise FileNotFoundError(f"TexturePoor missing scenes at {root}: {sorted(missing)}")
    counts=Counter(r["scene"] for r in rows)
    if set(counts)!=scenes or set(counts.values())!={100}:
        raise ValueError(f"invalid frozen scene counts: {counts}")
    cameras={}
    unique=set()
    for r in rows:
        images=[root.joinpath(*Path(r[f"image{i}"]).parts[-4:]) for i in (0,1)]
        if images[0].stem==images[1].stem:
            raise ValueError(r["pair_id"])
        pair=(r["scene"],*sorted(p.stem for p in images))
        if pair in unique:
            raise ValueError(f"duplicate two-view identity: {pair}")
        unique.add(pair)
        if not any(all((root/r["scene"]/bag/"images"/p.name).is_file() for p in images) for bag in r["source_bags"]):
            raise FileNotFoundError(f"no shared bag for {pair}")
        poses=[]
        for i,p in enumerate(images):
            if p not in cameras:
                if not p.is_file():
                    raise FileNotFoundError(p)
                bag=p.parents[1]
                cameras[p]=(np.loadtxt(bag/"intrins"/(p.stem+".txt")),np.loadtxt(bag/"poses"/(p.stem+".txt")))
            k,t=cameras[p]
            if not np.allclose(k,r[f"K{i}"],atol=1e-6,rtol=1e-8):
                raise ValueError(f"TexturePoor K differs: {p}")
            poses.append(t)
        relative=poses[1]@np.linalg.inv(poses[0])
        if not np.allclose(relative,r["T_0to1"],atol=2e-6) or not 2<=r["optical_axis_deg"]<=85 or r["baseline"]<.005:
            raise ValueError(f"frozen camera/geometry differs for {pair}")
    return dict(status="PASS",counts=dict(counts),pairs=len(unique),camera_records=len(cameras))
