from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
import hashlib
import json

import numpy as np
from PIL import Image

from .dataset import DATA, EXP, ROOT, TEST, VAL
from .geometry import angle_deg, relative

OUT = EXP / "manifests"
SEED = 66


def priority(identity):
    return hashlib.sha256(f"{SEED}/{identity}".encode()).hexdigest()


def freeze(name, data):
    path = OUT / name
    payload = (json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError(f"frozen manifest differs: {path}")
    else:
        temp = path.with_suffix(".tmp")
        temp.write_bytes(payload)
        temp.replace(path)
    return hashlib.sha256(payload).hexdigest()


def camera(bag, file):
    k = np.loadtxt(bag / "intrins" / (file.stem + ".txt"))
    pose = np.loadtxt(bag / "poses" / (file.stem + ".txt"))
    rot = pose[:3, :3]
    if k.shape != (3, 3) or pose.shape != (4, 4) or not np.isfinite(k).all() or not np.isfinite(pose).all():
        raise ValueError(file)
    if not (np.allclose(rot.T @ rot, np.eye(3), atol=2e-4) and abs(np.linalg.det(rot)-1) < 2e-4 and
            np.allclose(pose[3], [0,0,0,1]) and min(k[0,0],k[1,1]) > 0 and np.allclose(k[2],[0,0,1])):
        raise ValueError(f"invalid camera {file}")
    with Image.open(file) as image:
        wh = image.size
    return dict(image=str(file.relative_to(ROOT)), K=k.tolist(), w2c=pose.tolist(), wh=list(wh),
                sha256=hashlib.sha256(file.read_bytes()).hexdigest())


def inventory(scene):
    frames, groups = {}, defaultdict(list)
    occ = 0
    folders = set()
    sizes = Counter()
    for bag in sorted(scene.iterdir()):
        folders.update(x.name for x in bag.iterdir())
        files = sorted((bag / "images").glob("*.png"))
        ids = [f.stem for f in files]
        assert set(ids) == {p.stem for p in (bag/"poses").glob("*.txt")} == {p.stem for p in (bag/"intrins").glob("*.txt")}
        for f in files:
            item = camera(bag, f)
            old = frames.setdefault(f.stem, item)
            if any(item[k] != old[k] for k in ("K", "w2c", "sha256", "wh")):
                raise ValueError(f"camera/image alias conflict {f}")
            sizes[str(item["wh"])] += 1
        occ += len(files)
        for key in combinations(sorted(ids), 2):
            groups[key].append(bag.name)
    candidates = []
    for (a,b), bags in groups.items():
        f0, f1 = frames[a], frames[b]
        pose = relative(f0["w2c"], f1["w2c"])
        angle = angle_deg(pose[2,2])
        baseline = float(np.linalg.norm(pose[:3,3]))
        if not 2. <= angle <= 85. or baseline < 0.005:
            continue
        identity = f"{scene.name}/{a}/{b}"
        candidates.append(dict(pair_id=identity, scene=scene.name, image0=f0["image"], image1=f1["image"],
                               K0=f0["K"], K1=f1["K"], wh0=f0["wh"], wh1=f1["wh"], T_0to1=pose.tolist(),
                               source_bags=bags, baseline=baseline, optical_axis_deg=angle,
                               difficulty=min(2,int(angle//30))))
    counts = dict(bags=len(list(scene.iterdir())), image_occurrences=occ, unique_frames=len(frames),
                  directories=sorted(folders), image_sizes=dict(sizes), unique_bag_pairs=len(groups),
                  eligible=len(candidates), frame_identity={k:v["sha256"] for k,v in frames.items()})
    strata = [sorted((x for x in candidates if x["difficulty"] == d), key=lambda x:priority(x["pair_id"])) for d in range(3)]
    balanced = []
    for i in range(max(map(len,strata),default=0)):
        balanced.extend(s[i] for s in strata if i<len(s))
        if len(balanced)>=100:
            break
    if len(balanced)<100:
        raise ValueError(f"{scene.name} lacks 100 eligible pairs")
    return counts,balanced[:100]


def megadepth():
    index = ROOT/"data/megadepth/index/trainvaltest_list"
    bins = (index/"test_list.txt").read_text().split()
    ids = {x.split("_")[0] for x in bins}
    forbidden = {x.split("_")[0] for f in ["train_list.txt","val_list.txt"] for x in (index/f).read_text().split()}
    assert not ids & forbidden
    source = ROOT/"runs/coarse_rank_20261123T224428/inventory/megadepth_other_holdout.jsonl"
    selected = defaultdict(list)
    import heapq
    for line in source.open():
        row = json.loads(line)
        scene = row["scene"]
        if scene not in ids or Path(row["index"]).stem not in bins:
            continue
        score = -int(priority(row["pair_id"]),16)
        heap = selected[scene]
        entry = (score,row["pair_id"],row)
        if len(heap)<32:
            heapq.heappush(heap,entry)
        elif score>heap[0][0]:
            heapq.heapreplace(heap,entry)
    assert set(selected)==ids and all(len(h)==32 for h in selected.values())
    rows = [r for scene in sorted(ids) for _,_,r in sorted(selected[scene],reverse=True)]
    assert len({r["pair_id"] for r in rows})==len(rows)
    for row in rows:
        base=ROOT/"data/megadepth/train/phoenix/S6/zl548/MegaDepth_v1"
        for field in ("frame0","frame1"):
            rgb=base/row[field].removeprefix("md/")
            depth=rgb.parent.parent/"depths"/(rgb.stem+".h5")
            assert rgb.is_file() and depth.is_file(), rgb
    return rows


def main():
    assert {p.name for p in DATA.iterdir() if p.is_dir()} == VAL | TEST
    results = {"val": [],"test": []}
    audit = {}
    for p in sorted(DATA.iterdir()):
        counts, rows = inventory(p)
        audit[p.name] = counts
        results["val" if p.name in VAL else "test"].extend(rows)
    hashes = {"texturepoor_dataset_audit.json": freeze("texturepoor_dataset_audit.json",audit)}
    for split, rows in results.items():
        name = f"texturepoor_{split}_pairs.json"
        hashes[name] = freeze(name,rows)
    md = megadepth()
    for name, rows in [("texturepoor_smoke_val.json",[r for scene in sorted(VAL) for r in [x for x in results["val"] if x["scene"]==scene][:1]]),
                       ("megadepth_dev_pairs.json",md),("megadepth_smoke_val.json",md[::32])]:
        hashes[name] = freeze(name,rows)
    freeze("dev_manifest_hashes.json",hashes)
    print("development",len(results["val"]),len(md),"sealed test",len(results["test"]),"hashes",hashes)


if __name__=="__main__":
    main()
