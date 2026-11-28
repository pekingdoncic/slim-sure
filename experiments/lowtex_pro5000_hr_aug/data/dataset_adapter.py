"""Portable official indices and deterministic, resumable visits over all legal rows."""
import bisect
from functools import lru_cache
from time import perf_counter
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from .pad32_adapter import prepare
from ..utils.run_io import EXP, atomic_path, lock, read, sha


@lru_cache(maxsize=16)
def manifest(name):
    path = EXP/"manifests"/(name+".json")
    if sha(path) != read(EXP/"manifests/hashes.json")[name]["sha256"]:
        raise ValueError(f"manifest changed: {name}")
    return read(path)


def gray(path):
    if not path.is_file():
        raise FileNotFoundError(path)
    im = cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)
    if im is None:
        raise OSError(f"unreadable image {path}")
    return im


def index_path(base, name, formal=False):
    prefix = "scene_info_val_1500" if formal else "scene_info_0.1_0.7"
    for suffix in ("_no_sfm",""):
        p = base/"index"/(prefix+suffix)/(Path(name).stem+".npz")
        if p.exists():
            return p
    raise FileNotFoundError(base/"index"/prefix/name)


@lru_cache(maxsize=64)
def metadata(path):
    with np.load(path,allow_pickle=True) as z:
        return {k:z[k] for k in ("image_paths","depth_paths","intrinsics","poses") if k in z}


def resolve_image(base, value):
    parts = Path(str(value)).parts
    if "MegaDepth_v1" in parts:
        parts = parts[parts.index("MegaDepth_v1")+1:]
    elif parts and parts[0] == "md":
        parts = parts[1:]
    else:
        raw = base / str(value).lstrip("/")
        if raw.is_file():
            return raw
    for root in (base/"phoenix/S6/zl548/MegaDepth_v1",base/"MegaDepth_v1",base):
        p = root.joinpath(*parts)
        if p.is_file():
            return p
    raise FileNotFoundError(base.joinpath(*parts))


def load_md(base, index, frames, side, formal=False, need_depth=True, augment=None):
    import h5py
    meta = metadata(index)
    root = base/("test" if formal else "train")
    images, depths, cameras, poses = [], [], [], []
    start = perf_counter()
    for i in frames:
        image = gray(resolve_image(root,meta["image_paths"][i]))
        if augment is not None:
            image = augment(image)
        images.append(image)
        if need_depth:
            with h5py.File(resolve_image(root, meta["depth_paths"][i]),"r") as f:
                depths.append(f["depth"][()])
        cameras.append(np.asarray(meta["intrinsics"][i],np.float64))
        poses.append(np.asarray(meta["poses"][i],np.float64))
    t = poses[1] @ np.linalg.inv(poses[0])
    io_ms = (perf_counter()-start)*1000
    batch, plans = prepare(images,cameras,side, depths if need_depth else None)
    batch["T_0to1"] = torch.tensor(t,dtype=torch.float32)
    batch["T_1to0"] = torch.tensor(np.linalg.inv(t),dtype=torch.float32)
    physical = sorted(str(meta["image_paths"][i]) for i in frames)
    return batch, dict(K0=cameras[0],K1=cameras[1],T_0to1=t,plans=plans,io_ms=io_ms,physical_pair=physical)


def split_check():
    pool = manifest("train_pool")
    train = {b["scene"] for b in pool["bins"]}
    dev = {r["scene"] for r in manifest("megadepth_dev_pairs")}
    test = {r["bin"].split("_")[0] for r in manifest("general_test_indices")["megadepth"]}
    if train & dev or train & test or dev & test:
        raise ValueError("MegaDepth scene leakage")
    val = manifest("texturepoor_val_pairs")
    formal = manifest("texturepoor_test_pairs")
    if {r["scene"] for r in val} != {"1012","1026","1029"}:
        raise ValueError("TexturePoor development scenes")
    held = {r["scene"] for r in formal}
    if len(held)!=14 or held & {r["scene"] for r in val}:
        raise ValueError("TexturePoor scene leakage")
    for rows, n in ((val,300),(formal,1400),(manifest("megadepth_dev_pairs"),256)):
        if len(rows)!=n or len({r["pair_id"] for r in rows}) != n:
            raise ValueError("scene manifest size/uniqueness")
    return pool


def prepare_pool(base, cache):
    pool = split_check()
    official = base/"index/trainvaltest_list/train_list.txt"
    bins = pool["bins"]
    if sha(official) != pool["split_sha256"] or set(official.read_text().split()) != {b["bin"] for b in bins}:
        raise ValueError("official train_list differs from the freeze")
    with lock(cache/"build.lock"):
        offset = 0
        for scene in pool["scenes"]:
            members = [b for b in bins if b["scene"] == scene["scene"]]
            packed, numbers = {}, {}
            for b in members:
                name = b["bin"]
                idx = index_path(base,name)
                if sha(idx) not in (b["local_layout_sha256"],b["original_sha256"]):
                    raise ValueError(f"unknown index {idx}")
                with np.load(idx,allow_pickle=True) as z:
                    paths = z["image_paths"]
                    mapping = np.array([numbers.setdefault(str(v),len(numbers)) if v is not None else -1 for v in paths])
                    rows = np.asarray([r[0] for r in z["pair_infos"] if r[1]>0],np.int32)
                if len(rows)!=b["legal_rows"] or (mapping[rows]<0).any():
                    raise ValueError(name)
                pairs = np.sort(mapping[rows],axis=1)
                packed[name] = ((pairs[:,0].astype(np.int64)<<32) | pairs[:,1],rows)
            unique = np.unique(np.concatenate([v[0] for v in packed.values()]))
            if len(unique)!=scene["unique_unordered"]:
                raise ValueError(f"physical identities changed in {scene}")
            for name,(keys,frames) in packed.items():
                table = np.column_stack((frames,offset+np.searchsorted(unique,keys))).astype(np.int32)
                with atomic_path(cache/(name+".npy")) as tmp, tmp.open("wb") as f:
                    np.save(f,table)
            offset += len(unique)
        if offset != pool["unique_unordered"]:
            raise ValueError("physical pair count")


@lru_cache(maxsize=128)
def table(path):
    return np.load(path,mmap_mode="r")


class TrainPool(Dataset):
    def __init__(self, base, cache, side, cfg, start=0, stop=None):
        self.base, self.cache, self.side, self.cfg = base,cache,side,cfg
        self.bins = manifest("train_pool")["bins"]
        self.ends = np.cumsum([b["legal_rows"] for b in self.bins])
        self.size = int(self.ends[-1])
        self.start,self.stop = start,self.size if stop is None else stop
        self.epoch,self.order = None,None

    def __len__(self):
        return self.stop-self.start

    def row_at(self, position):
        epoch, ix = divmod(position,self.size)
        if self.epoch != epoch:
            seed = self.cfg["seed_sampler"]
            path = self.cache/f"order_{seed}_{epoch}.npy"
            with lock(path.with_suffix(".lock")):
                if not path.exists():
                    order = np.arange(self.size,dtype=np.int32)
                    np.random.default_rng([seed,epoch]).shuffle(order)
                    with atomic_path(path) as tmp, tmp.open("wb") as fp:
                        np.save(fp,order)
            self.order = np.load(path,mmap_mode="r")
            self.epoch = epoch
        return int(self.order[ix])

    def __getitem__(self, ix):
        pos = self.start+ix
        raw = self.row_at(pos)
        b = bisect.bisect_right(self.ends,raw)
        name = self.bins[b]["bin"]
        local = raw-(int(self.ends[b-1]) if b else 0)
        a,c,pid = map(int,table(self.cache/(name+".npy"))[local])
        rng = np.random.default_rng([self.cfg["seed_augmentation"],pos])
        aug = None
        if self.cfg["augmentation"] == "brightness":
            aug = lambda im: np.clip(im*rng.uniform(.85,1.15),0,255).astype(np.uint8)
        elif self.cfg["augmentation"] != "none":
            raise ValueError("unknown augmentation")
        d,_ = load_md(self.base,index_path(self.base,name),[a,c],self.side,augment=aug)
        d.update(pair_id=pid,row_id=raw,position=pos)
        return d


def loader(dataset, batch, cfg):
    kwargs = {}
    if cfg["workers"]:
        kwargs.update(persistent_workers=cfg["persistent_workers"],prefetch_factor=cfg["prefetch_factor"])
    return DataLoader(dataset,batch_size=batch,shuffle=False,num_workers=cfg["workers"],
                      pin_memory=cfg["pin_memory"],generator=torch.Generator().manual_seed(4821),**kwargs)
