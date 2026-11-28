from pathlib import Path
from time import perf_counter
import cv2
import numpy as np
from ..data.dataset_adapter import gray,index_path,load_md,manifest
from ..data.pad32_adapter import prepare
from ..utils.run_io import EXP,read,sha


def md_test_indices(base):
    for item in manifest("general_test_indices")["megadepth"]:
        name = item["bin"]+".npz"
        paths = [base/"index"/d/name for d in ("scene_info_val_1500_no_sfm","scene_info_val_1500")]
        official=read(EXP/"manifests/official_megadepth_indices.json")[item["bin"]]
        matching = [p for p,h in zip(paths,(item["index_sha256"],official)) if p.is_file() and sha(p)==h]
        if len(matching)==2:
            with np.load(matching[0],allow_pickle=True) as a, np.load(matching[1],allow_pickle=True) as b:
                if [tuple(r[0]) for r in a["pair_infos"]]!=[tuple(r[0]) for r in b["pair_infos"]]:
                    raise ValueError(f"derived test index row order differs from official {name}")
        if not matching:
            if not any(p.is_file() for p in paths):
                raise FileNotFoundError(paths[0])
            raise ValueError(f"MegaDepth-1500 index hash mismatch {name}")
        with np.load(matching[0],allow_pickle=True) as z:
            pairs = z["pair_infos"]
        if len(pairs)!=item["records"]:
            raise ValueError(name)
        for k,row in enumerate(pairs):
            yield dict(scene=name.split("_")[0],index=matching[0],frame_indices=row[0],
                       pair_id=f"{item['bin']}/{k}",physical_pair=tuple(sorted(map(int,row[0]))))


def records(args,domain,split):
    formal = split=="test"
    if domain=="megadepth":
        if formal:
            return list(md_test_indices(args.data_root))
        return manifest("megadepth_smoke_val" if split=="quick" else "megadepth_dev_pairs")
    if domain=="texturepoor":
        from ..data.texturepoor_protocol import verify
        verify(args.texturepoor_root,formal)
        return manifest("texturepoor_"+("test_pairs" if formal else "smoke_val" if split=="quick" else "val_pairs"))
    if not formal:
        raise ValueError(f"{domain} has no development split")
    indices = manifest("general_test_indices")
    if domain=="hpatches":
        return [dict(scene=Path(p).parent.name,paths=(p,q,h),pair_id=p+"/"+Path(q).name) for p,q,h in indices[domain]]
    if domain!="scannet":
        raise ValueError(domain)
    dirs = [args.scannet_index_root,args.scannet_root/"index",args.scannet_root/"index/scene_data/val_list",
            args.scannet_root.parent/"assets/scannet_test_1500",args.scannet_root]
    directory = next((d for d in dirs if d and (d/"test.npz").is_file()),None)
    if directory is None:
        raise FileNotFoundError("ScanNet test.npz and intrinsics.npz; set --scannet-index-root")
    for name,value in indices["scannet"].items():
        if sha(directory/name)!=value:
            raise ValueError(f"ScanNet index hash mismatch {name}")
    with np.load(directory/"test.npz") as z, np.load(directory/"intrinsics.npz") as cameras:
        if len(z["name"])!=1500:
            raise ValueError("ScanNet-1500 row count")
        rows = []
        for idx,(s,sub,a,b) in enumerate(z["name"]):
            scene = f"scene{int(s):04d}_{int(sub):02d}"
            rows.append(dict(scene=scene,frames=(int(a),int(b)),k=cameras[scene].reshape(3,3),
                             pair_id=f"{idx}/{scene}/{a}/{b}"))
        return rows


def load_record(args,domain,split,row,side,point_gt=True):
    if domain=="megadepth":
        formal = split=="test"
        idx = row["index"] if formal else index_path(args.data_root,row["index"])
        d,meta = load_md(args.data_root,idx,row["frame_indices"],side,formal,need_depth=point_gt and not formal)
        return d,{**row,**meta}
    start = perf_counter()
    if domain=="texturepoor":
        images = [gray(args.texturepoor_root.joinpath(*Path(row[f"image{i}"]).parts[-4:])) for i in (0,1)]
        cameras = [np.asarray(row[f"K{i}"]) for i in (0,1)]
        if any(tuple(row[f"wh{i}"])!=images[i].shape[::-1] for i in (0,1)):
            raise ValueError("TexturePoor image size differs from frozen calibration")
        extra = {"T_0to1":np.asarray(row["T_0to1"])}
    elif domain=="scannet":
        root = args.scannet_root
        if (root/"test").is_dir():
            root = root/"test"
        scene = root/row["scene"]
        images = [gray(scene/"color"/f"{n}.jpg") for n in row["frames"]]
        poses = [np.loadtxt(scene/"pose"/f"{n}.txt") for n in row["frames"]]
        # Official intrinsics.npz is calibrated at 640x480, raw color often 1296x968.
        cameras = [np.diag([im.shape[1]/640,im.shape[0]/480,1.])@row["k"] for im in images]
        extra = {"T_0to1":np.linalg.inv(poses[1])@poses[0]}
    else:
        p,q,hom = [args.hpatches_root.joinpath(*Path(x).parts[-2:]) for x in row["paths"]]
        images = [gray(p),gray(q)]
        cameras = [np.eye(3),np.eye(3)]
        extra = {"H":np.loadtxt(hom)}
    io = (perf_counter()-start)*1000
    d,plans = prepare(images,cameras,side)
    physical = sorted(row["frames"]) if domain=="scannet" else row["pair_id"]
    return d,dict(pair_id=row["pair_id"],scene=row["scene"],K0=cameras[0],K1=cameras[1],plans=plans,
                  io_ms=io,physical_pair=physical,**extra)
