"""Evaluate original-pixel predictions on frozen development pairs; check inputs without a model."""
import argparse
import json
from pathlib import Path

import numpy as np

from .dataset import EXP, VAL, read_image
from .metrics import aggregate, evaluate_pair
from .visualize import draw


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--split",choices=["val","smoke"],default="smoke")
    parser.add_argument("--predictions",type=Path,help="JSONL with pair_id, xy0, xy1 in original pixels")
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    selection="texturepoor_val_pairs.json" if args.split=="val" else "texturepoor_smoke_val.json"
    records=json.loads((EXP/"manifests"/selection).read_text())
    if not all(r["scene"] in VAL for r in records):
        raise ValueError("unexpected scene in development manifest")
    if args.predictions is None:
        for r in records:
            for i in (0,1):
                img=read_image(r[f"image{i}"])
                if list(img.shape[::-1])!=r[f"wh{i}"]:
                    raise ValueError(r["pair_id"])
        print(f"read both original images for all {len(records)} development pairs")
        return
    if not args.output or EXP not in args.output.resolve().parents:
        parser.error("a new output directory under the experiment is required")
    pred=[json.loads(l) for l in args.predictions.read_text().splitlines()]
    index={p["pair_id"]:p for p in pred}
    if len(index)!=len(pred) or set(index)!={r["pair_id"] for r in records}:
        raise ValueError("prediction identity mismatch; encode no-match as empty arrays")
    args.output.mkdir(parents=True,exist_ok=False)
    rows=[]
    seen=set()
    with (args.output/"per_pair.jsonl").open("x") as out:
        for r in records:
            p=index[r["pair_id"]]
            result=evaluate_pair(r,p["xy0"],p["xy1"])
            for key in ("matcher_ms","peak_allocated_bytes","peak_reserved_bytes"):
                result[key]=p.get(key)
            out.write(json.dumps(result,allow_nan=False)+"\n")
            group=(r["scene"],result["pose_recovered"])
            if group not in seen:
                draw(r,p,args.output/(r["pair_id"].replace("/","_")+".jpg"))
                seen.add(group)
            rows.append(result)
    (args.output/"summary.json").write_text(json.dumps(aggregate(rows),indent=2,allow_nan=False))


if __name__=="__main__":
    main()
