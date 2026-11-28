import fcntl
import hashlib
import json
from pathlib import Path

import numpy as np


def partition(ordinal, n, splits=27):
    return ordinal*n//splits


class Pool:
    """A deterministic permutation per bin with coverage encoded by the committed cursor."""
    def __init__(self,experiment,base,seed):
        self.base=base
        self.manifest=experiment/"manifests/train_pool.json"
        self.bins=json.loads(self.manifest.read_text())["bins"]
        self.cache=experiment/"cache/sampler"/hashlib.sha256(self.manifest.read_bytes()).hexdigest()
        self.cache.mkdir(parents=True,exist_ok=True)
        self.seed=seed
        self.sizes=np.array([b["legal_rows"] for b in self.bins])
        self.quota=np.maximum(self.sizes,2700)
        self.ends=np.r_[0,np.cumsum([sum(partition(e+1,self.quota)-partition(e,self.quota)) for e in range(27)])]
        self.epoch=-1
        self.identities={}
        self.order={}

    def bin_rows(self,bi):
        if bi not in self.identities:
            name=self.bins[bi]["bin"]
            path=self.cache/(name+".npy")
            with (self.cache/(name+".lock")).open("a") as lock:
                fcntl.flock(lock,fcntl.LOCK_EX)
                if not path.exists():
                    index=self.base/"index/scene_info_0.1_0.7_no_sfm"/(name+".npz")
                    if hashlib.sha256(index.read_bytes()).hexdigest()!=self.bins[bi]["local_layout_sha256"]:
                        raise ValueError(f"index hash changed: {index}")
                    with np.load(index,allow_pickle=True) as z:
                        pairs=z["pair_infos"]
                    ids=np.array([r[0] for r in pairs if r[1]>0],dtype=np.int32)
                    if len(ids)!=self.sizes[bi]: raise ValueError(name)
                    tmp=path.with_suffix(".tmp")
                    with tmp.open("wb") as out: np.save(out,ids)
                    tmp.replace(path)
            self.identities[bi]=np.load(path,mmap_mode="r")
            self.order[bi]=np.random.default_rng(np.random.SeedSequence([self.seed,bi])).permutation(self.sizes[bi]).astype(np.int32)
        return self.identities[bi]

    def at(self,position):
        if not 0<=position<self.ends[-1]: raise IndexError(position)
        epoch=int(np.searchsorted(self.ends,position,side="right")-1)
        if epoch!=self.epoch:
            start=partition(epoch,self.quota)
            count=partition(epoch+1,self.quota)-start
            self.which=np.repeat(np.arange(len(count),dtype=np.int32),count)
            self.visit=np.concatenate([np.arange(i,i+n,dtype=np.int32) for i,n in zip(start,count)])
            perm=np.random.default_rng(np.random.SeedSequence([self.seed,epoch,431])).permutation(len(self.which))
            # Shuffle bin positions while consuming each bin in its original cycle order.
            self.which=self.which[perm]
            order=np.argsort(self.which,kind="stable")
            visits=self.visit.copy()
            self.visit[order]=visits
            self.epoch=epoch
        ix=int(position-self.ends[epoch])
        bi=int(self.which[ix]); visit=int(self.visit[ix])
        pairs=self.bin_rows(bi)
        row=int(self.order[bi][visit % len(pairs)])
        return bi,row,pairs[row].tolist(),visit<len(pairs)
