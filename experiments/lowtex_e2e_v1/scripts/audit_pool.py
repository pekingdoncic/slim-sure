"""Check official split rows against the installed RGB/depth layout without changing an index."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
INDEX = ROOT / "data/megadepth/index"
TRAIN = ROOT / "data/megadepth/train"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect_bin(name, seen, missing):
    official = INDEX / "scene_info_0.1_0.7" / (name + ".npz")
    mapped = INDEX / "scene_info_0.1_0.7_no_sfm" / official.name
    raw = np.load(official, allow_pickle=True)
    with np.load(mapped, allow_pickle=True) as layout:
        images = layout["image_paths"]
        depths = layout["depth_paths"]
        rows = layout["pair_infos"]
        ks = layout["intrinsics"]
        poses = layout["poses"]
    legal = 0
    readable = 0
    checked = {}
    ordered_set = set()
    assert len(rows) == len(raw["pair_infos"])
    for ref, row in zip(raw["pair_infos"], rows):
        assert np.array_equal(ref[0], row[0]) and ref[1] == row[1] and np.array_equal(ref[2], row[2], equal_nan=True)
        if ref[1] <= 0:
            continue
        i, j = map(int, ref[0])
        for n in (i, j):
            if n in checked:
                continue
            assert Path(raw["image_paths"][n]).stem == Path(images[n]).stem, (name, n, raw["image_paths"][n], images[n])
            assert Path(images[n]).stem == Path(depths[n]).stem, (name, n, images[n], depths[n])
            assert raw["depth_paths"][n] == depths[n]
            assert np.array_equal(raw["intrinsics"][n], ks[n]) and np.array_equal(raw["poses"][n], poses[n])
            absent = [str(path) for path in (images[n], depths[n]) if not (TRAIN / path).is_file()]
            missing.update(absent)
            checked[n] = not absent
        legal += 1
        readable += int(checked[i] and checked[j])
        ordered_set.add((str(images[i]), str(images[j])))
    scene = name.split("_")[0]
    seen[scene].update(tuple(sorted(pair)) for pair in ordered_set)
    return dict(bin=name, scene=scene, raw_rows=len(rows), legal_rows=legal,
                available_rows=readable, distinct_ordered=len(ordered_set), indexed_frames=len(checked),
                original_sha256=digest(official), local_layout_sha256=digest(mapped))


def main():
    split = INDEX / "trainvaltest_list/train_list.txt"
    names = split.read_text().split()
    assert len(set(names)) == len(names)
    groups = defaultdict(list)
    for name in names:
        groups[name.split("_")[0]].append(name)
    bins, scenes, missing = [], [], set()
    for scene, group in sorted(groups.items()):
        seen = defaultdict(set)
        for name in group:
            bins.append(inspect_bin(name, seen, missing))
        scenes.append(dict(scene=scene, unique_unordered=len(seen[scene])))
        print(scene, len(group), scenes[-1], flush=True)
    report = dict(schema=1, split_sha256=digest(split), reference_split=str(split.relative_to(ROOT)),
                  filter="pair_info[1] > 0", identity="scene + unordered mapped image paths; raw row identity = official bin + row",
                  bins=bins, scenes=scenes, missing_files=sorted(missing),
                  legal_rows=sum(b["legal_rows"] for b in bins), available_rows=sum(b["available_rows"] for b in bins),
                  unique_unordered=sum(s["unique_unordered"] for s in scenes), seen_by_training=0,
                  official_draws_per_epoch=100*len(names), official_reference_epochs=27,
                  equal_bin_quota_epochs_for_coverage=(max(b["legal_rows"] for b in bins)+99)//100)
    directory = EXP / "manifests"
    directory.mkdir(exist_ok=True)
    path = directory / "train_pool.json"
    new = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text() != new:
        raise RuntimeError("existing manifest differs, create a versioned audit")
    tmp = path.with_suffix(".tmp")
    tmp.write_text(new)
    tmp.replace(path)
    print({k:v for k,v in report.items() if k not in ("bins","scenes","missing_files")}, "missing",len(missing), "sha256",digest(path),flush=True)
    return int(bool(missing))


if __name__ == "__main__":
    sys.exit(main())
