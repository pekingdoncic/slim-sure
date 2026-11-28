"""Dense, stable unordered image-pair IDs for cumulative coverage, derived from the original ordered rows."""
import hashlib
import json
from pathlib import Path
import fcntl

import numpy as np

from ..evaluation.texturepoor_sfm.dataset import EXP,ROOT


def build():
    manifest=EXP/"manifests/train_pool.json"
    pool=json.loads(manifest.read_text())
    directory=EXP/"cache/sampler"/hashlib.sha256(manifest.read_bytes()).hexdigest()
    directory.mkdir(parents=True,exist_ok=True)
    report=directory/"unordered_identity.json"
    with (directory/"identity.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if report.is_file():
            return report
        offset=0
        checks={}
        for scene in pool["scenes"]:
            bins=[b for b in pool["bins"] if b["scene"]==scene["scene"]]
            frame_ids={}
            packed={}
            for b in bins:
                name=b["bin"]
                index=ROOT/"data/megadepth/index/scene_info_0.1_0.7_no_sfm"/(name+".npz")
                if hashlib.sha256(index.read_bytes()).hexdigest()!=b["local_layout_sha256"]:
                    raise ValueError(index)
                with np.load(index,allow_pickle=True) as z:
                    frames=z["image_paths"]
                    mapping=np.array([frame_ids.setdefault(str(v),len(frame_ids)) if v is not None else -1 for v in frames],dtype=np.int64)
                    rows=z["pair_infos"]
                    ids=np.array([r[0] for r in rows if r[1]>0],dtype=np.int32)
                if len(ids)!=b["legal_rows"]:raise ValueError(name)
                order_path=directory/(name+".npy")
                with (directory/(name+".lock")).open("a") as binlock:
                    fcntl.flock(binlock,fcntl.LOCK_EX)
                    if order_path.exists():
                        if not np.array_equal(ids,np.load(order_path,mmap_mode="r")):raise ValueError(order_path)
                    else:
                        temp=order_path.with_suffix(".tmp")
                        with temp.open("wb") as out: np.save(out,ids)
                        temp.replace(order_path)
                pairs=np.sort(mapping[ids],axis=1)
                packed[name]=(pairs[:,0]<<32) | pairs[:,1]
            unique=np.unique(np.concatenate(list(packed.values())))
            if len(unique)!=scene["unique_unordered"]:raise ValueError(scene)
            for name,keys in packed.items():
                path=directory/(name+".unordered.npy")
                arr=(offset+np.searchsorted(unique,keys)).astype(np.int32)
                tmp=path.with_suffix(".tmp")
                with tmp.open("wb") as out:np.save(out,arr)
                tmp.replace(path)
                checks[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
            offset+=len(unique)
        if offset!=pool["unique_unordered"]:raise ValueError(offset)
        tmp=report.with_suffix(".tmp")
        tmp.write_text(json.dumps(dict(total=offset,hashes=checks),sort_keys=True))
        tmp.replace(report)
        print("identity",offset,report)
        return report


if __name__=="__main__":
    build()
