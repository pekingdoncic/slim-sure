from functools import lru_cache
from pathlib import Path
import hashlib
import json
import os
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from ..training.data import load_pair
from ..evaluation.texturepoor_sfm.dataset import EXP, ROOT

BASE=ROOT/"data/megadepth"
OUT=EXP/"reports/fast_screening"
CACHE=EXP/"cache/fast_screening_v1"
N=8862673
U=6105188
SEED=66

def sha(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda:f.read(1048576),b""):h.update(chunk)
    return h.hexdigest()

def save(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_name(p.name+f".{os.getpid()}.tmp")
    t.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+"\n")
    t.replace(p)

def read(p):
    return json.loads(p.read_text())

def prepare():
    import fcntl
    CACHE.mkdir(parents=True,exist_ok=True)
    OUT.mkdir(parents=True,exist_ok=True)
    with (CACHE/"prepare.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        path=CACHE/"order_66.npy"
        if not path.exists():
            order=np.arange(N,dtype=np.int32)
            np.random.default_rng(SEED).shuffle(order)
            with (CACHE/"order.tmp").open("wb") as f:np.save(f,order)
            (CACHE/"order.tmp").replace(path)
        receipt=OUT/"frozen_inputs.json"
        manifests=EXP/"manifests"
        original=read(manifests/"dev_manifest_hashes.json")
        for name in ("megadepth_dev_pairs.json","texturepoor_val_pairs.json","megadepth_smoke_val.json","texturepoor_smoke_val.json"):
            if sha(manifests/name)!=original[name]:raise RuntimeError(f"Development manifest changed: {name}")
        budget=read(manifests/"full_budget.json")
        if sha(manifests/"train_pool.json")!=budget["pool_sha256"] or sha(EXP/"protocol_frozen.md")!=budget["protocol_sha256"]:
            raise RuntimeError("Historical training pool or parent protocol changed")
        inputs={"order":sha(path),"parent_protocol":sha(EXP/"protocol_frozen.md"),"protocol":sha(EXP/"protocols/fast_screening_v1.md"),
                "pool":sha(manifests/"train_pool.json"),"prior_budget":sha(manifests/"full_budget.json"),"seed":SEED,"rows":N,"unordered":U,
                "updates":(N+3)//4,"dev":{}}
        for domain,k in (("texturepoor",8),("megadepth",4)):
            data=read(manifests/("texturepoor_val_pairs.json" if domain=="texturepoor" else "megadepth_dev_pairs.json"))
            pick=[]
            for scene in sorted({r["scene"] for r in data}):
                ranked=sorted((r for r in data if r["scene"]==scene),key=lambda r:hashlib.sha256(("66/fast/"+r["pair_id"]).encode()).digest())
                pick.extend(ranked[:k])
            p=OUT/(domain+"_quarter_val.json")
            if not p.exists():save(p,pick)
            if read(p)!=pick:raise ValueError(p)
            inputs["dev"][domain]=sha(p)
        if receipt.exists() and read(receipt)!=inputs:raise ValueError("Frozen inputs changed")
        if not receipt.exists():save(receipt,inputs)
    return inputs

class Rows(Dataset):
    def __init__(self,first,stop):
        self.first,self.stop=first,stop
        self.bins=read(EXP/"manifests/train_pool.json")["bins"]
        self.bounds=np.cumsum([b["legal_rows"] for b in self.bins])
        assert self.bounds[-1]==N
        self.orders={c:np.load(CACHE/f"order_{SEED+c}.npy",mmap_mode="r") for c in range(first//N,(stop-1)//N+1)}
        self.cache=EXP/"cache/sampler"/read(OUT/"frozen_inputs.json")["pool"]
        self.tables={}
    def __len__(self):return self.stop-self.first
    def __getitem__(self,i):
        position=self.first+i
        cycle,offset=divmod(position,N)
        gid=int(self.orders[cycle][offset])
        bi=int(self.bounds.searchsorted(gid,side="right"))
        row=gid-(int(self.bounds[bi-1]) if bi else 0)
        name=self.bins[bi]["bin"]
        if name not in self.tables:
            self.tables[name]=(np.load(self.cache/(name+".npy"),mmap_mode="r"),np.load(self.cache/(name+".unordered.npy"),mmap_mode="r"))
        pairs,physical=self.tables[name]
        pairkey=f"{name}/{row}"
        try:d,_=load_pair(BASE,name,pairs[row],640,"cpu",pairkey)
        except Exception as error:raise RuntimeError(f"position={position} row={gid} identity={pairkey}") from error
        d.pop("_gt")
        return (position,gid,int(physical[row]),bi),d

def collate(items):return items

def initworker(_):
    cv2.setNumThreads(1)
    from ..training import data
    data.cameras=lru_cache(maxsize=4)(data.cameras.__wrapped__)

def loader(first,stop,bs,workers,pin,persistent,prefetch):
    return DataLoader(Rows(first,stop),batch_size=bs,num_workers=workers,shuffle=False,pin_memory=pin,
          persistent_workers=persistent and workers>0,prefetch_factor=prefetch if workers else None,
          collate_fn=collate,worker_init_fn=initworker,
          generator=torch.Generator().manual_seed(SEED),multiprocessing_context="fork" if workers else None)

def gpu(batch):
    return [(identity,{k:v.to("cuda",non_blocking=True) if torch.is_tensor(v) else v for k,v in d.items()}) for identity,d in batch]
