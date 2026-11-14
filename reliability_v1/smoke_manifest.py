"""Small deterministic engineering manifests from committed audit evidence.

This is intentionally separate from the final complete train/validation split.
It grants no eligibility for formal training.
"""
import hashlib
import contextlib
import json
import math
import os
from pathlib import Path
import random
import sqlite3
from datetime import datetime, timezone, timedelta

from reliability_data import label_contract, sha256_file, atomic_json, canonical_json

SCHEMA = "slim_engineering_smoke_manifest_v1"
DEFAULT_TRAIN_PAIRS = 8
DEFAULT_VAL_PAIRS = 4


def digest(value):
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def _within_logical(path, root):
    return os.path.commonpath((os.path.abspath(path), os.path.abspath(root))) == os.path.abspath(root)


@contextlib.contextmanager
def _connect(path):
    db = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)
    db.execute("PRAGMA query_only=ON")
    db.execute("PRAGMA cache_size=-8192")
    db.execute("PRAGMA temp_store=FILE")
    db.execute("BEGIN")
    try:
        yield db
    finally:
        db.rollback();db.close()


def _exclusions(source_run):
    path = source_run / "test_scene_exclusions.json"
    obj = json.loads(path.read_text())
    scenes = set(obj["scenes"])
    if not {"0015", "0022"}.issubset(scenes):
        raise RuntimeError("Known MegaDepth test scenes missing from exclusions")
    evidence = {"test_scene_exclusions": {"path": str(path), "sha256": sha256_file(path)}}
    prior = source_run / "prior_test_subset_verification.json"
    if not prior.is_file():
        raise RuntimeError("Prior fixed-subset scene verification is required")
    history = json.loads(prior.read_text())
    if history.get("status") != "pass":
        raise RuntimeError("Prior fixed-subset exclusions are not verified")
    for item in history.get("records", []):
        if not set(item.get("scene_ids", [])).issubset(scenes):
            raise RuntimeError("Prior test subset contains an unexcluded scene")
    evidence["prior_test_subsets"] = {"path": str(prior), "sha256": sha256_file(prior)}
    test_ids = set(obj.get("pair_ids", [])) | set(obj.get("test_pair_ids", []))
    return scenes, test_ids, evidence, obj.get("sources", [])


def _source_snapshot(db):
    row = db.execute("SELECT value FROM audit_identity WHERE key='source'").fetchone()
    if not row:
        raise RuntimeError("Missing source audit identity")
    source = json.loads(row[0])
    committed = db.execute("SELECT COUNT(*) FROM audit_chunks").fetchone()[0]
    return source, int(committed)


def _select_ids(db, excluded, forbidden_ids, seed, train_count, val_count):
    # Only complete index commits participate. This query streams a small
    # scene summary; no pair collection is materialized in Python.
    scene_counts = {}
    for scene, count in db.execute("""SELECT p.scene,COUNT(*)
            FROM pairs p JOIN audit_chunks a ON a.index_id=p.index_id
            JOIN indices i ON i.id=p.index_id
            WHERE i.status='audited' GROUP BY p.scene ORDER BY p.scene"""):
        if scene not in excluded:
            scene_counts[scene] = int(count)
    if len(scene_counts) < 2:
        raise RuntimeError("At least two committed legal scenes are required")
    order = sorted(scene_counts)
    random.Random(seed).shuffle(order)
    total = sum(scene_counts.values())
    val_scene_target = max(1, min(len(order)-1, round(len(order)*0.1)))
    val_scenes = []
    val_capacity = 0
    for scene in order:
        # Keep enough actual capacity for the scene-disjoint train set.
        if total - val_capacity - scene_counts[scene] < train_count:
            continue
        val_scenes.append(scene)
        val_capacity += scene_counts[scene]
        if len(val_scenes) >= val_scene_target and val_capacity >= val_count:
            break
    train_scenes = [s for s in order if s not in set(val_scenes)]
    if val_capacity < val_count or sum(scene_counts[s] for s in train_scenes) < train_count:
        raise RuntimeError("Committed scenes cannot provide the required disjoint 8+4 pairs")

    def round_robin(scenes, count):
        cursors = []
        for scene in scenes:
            cursor = db.execute("""SELECT p.pair_id
                FROM pairs p JOIN audit_chunks a ON a.index_id=p.index_id
                JOIN indices i ON i.id=p.index_id
                WHERE p.scene=? AND i.status='audited'
                ORDER BY p.pair_id LIMIT ?""", (scene,count+len(forbidden_ids)))
            cursors.append(cursor)
        chosen = []
        while len(chosen) < count:
            progress = False
            for cursor in cursors:
                while True:
                    row = cursor.fetchone()
                    if row is None:
                        break
                    if row[0] in forbidden_ids:
                        continue
                    chosen.append(row[0])
                    progress = True
                    break
                if len(chosen) == count:
                    break
            if not progress:
                raise RuntimeError("Insufficient non-test pairs after explicit exclusions")
        for cursor in cursors:
            cursor.close()
        return chosen
    return round_robin(train_scenes, train_count), round_robin(val_scenes, val_count), {
        "eligible_scene_count": len(scene_counts),
        "eligible_committed_pair_count_before_explicit_id_exclusion": total,
        "scene_order": order,
        "partition_train_scenes": train_scenes,
        "partition_val_scenes": val_scenes,
        "policy": "Python Random(seed) over sorted legal scene identities; approximately 10% held out with capacity constraints; stable pair_id scene round robin",
        "score_or_label_based_selection": False,
    }


def _record(db, pair_id, source_run, data_root, split):
    row = db.execute("""SELECT p.pair_id,p.scene,p.frame0,p.frame1,p.index_id,p.ordinal,p.overlap,
                i.path,i.sha256,i.mapping_path,i.mapping_sha256,a.path,a.sha256
            FROM pairs p JOIN audit_chunks a ON a.index_id=p.index_id
            JOIN indices i ON i.id=p.index_id
            WHERE p.pair_id=? AND i.status='audited'""", (pair_id,)).fetchone()
    if row is None:
        raise RuntimeError("Selected pair is not part of a complete committed audit index")
    pair, scene, f0, f1, index_id, ordinal, overlap, ipath, ihash, mpath, mhash, cpath, chash = row
    if not math.isfinite(overlap) or overlap <= 0:
        raise RuntimeError("Selected pair violates recorded native overlap filter")
    record = dict(pair_id=pair, scene=scene, frame0=f0, frame1=f1, index_id=index_id,
                  ordinal=ordinal, direction="0to1", overlap=overlap, split=split,
                  audit_dir=str(source_run), file_sha256={}, file_evidence={},
                  source_index={"path": ipath, "sha256": ihash, "mapping_path": mpath,
                                "mapping_sha256": mhash},
                  committed_audit_chunk={"path": cpath, "sha256": chash})
    for side, fid in enumerate((f0, f1)):
        frame = db.execute("SELECT scene,image,depth,K,pose,status FROM frames WHERE id=?", (fid,)).fetchone()
        if frame is None or frame[5] != "valid" or frame[0] != scene:
            raise RuntimeError("Selected frame is not an audited valid member of its scene")
        record[f"image{side}"], record[f"depth{side}"] = frame[1:3]
        record[f"K{side}"], record[f"pose{side}"] = json.loads(frame[3]), json.loads(frame[4])
        frame_identity = dict(scene=scene,image=frame[1],depth=frame[2],K=record[f"K{side}"],pose=record[f"pose{side}"])
        if hashlib.sha256(canonical_json(frame_identity).encode()).hexdigest() != fid:
            raise RuntimeError("Audited camera/frame identity mismatch")
        for kind in ("image", "depth"):
            key = f"{kind}{side}"
            path = record[key]
            if not _within_logical(path, data_root):
                raise RuntimeError("Selected file is outside the existing SLiM logical data root")
            file_row = db.execute("SELECT resolved,sha256,size,status,shape FROM files WHERE path=?", (path,)).fetchone()
            if file_row is None or file_row[3] != "valid" or not file_row[1]:
                raise RuntimeError("Selected file has no valid committed hash/readability evidence")
            record["file_sha256"][key] = file_row[1]
            record["file_evidence"][key] = dict(logical=path,resolved=file_row[0],
                                               sha256=file_row[1],size=file_row[2],
                                               shape=json.loads(file_row[4]))
    expected = hashlib.sha256((f0+"|"+f1+"|0to1").encode()).hexdigest()[:32]
    if expected != pair:
        raise RuntimeError("Audited stable pair identity mismatch")
    record["camera_identity_sha256"] = digest({k:record[k] for k in ("K0","K1","pose0","pose1")})
    return record


def _verify_sources(records, exclusion_sources, verify_selected_files, max_read_bytes):
    expected = {}
    kinds = {}
    for record in records:
        for key, value in record["file_evidence"].items():
            if verify_selected_files:
                expected[value["logical"]] = value["sha256"]
                kinds[value["logical"]] = "selected_image_or_depth"
            if str(Path(value["logical"]).resolve()) != value["resolved"]:
                raise RuntimeError("Selected symlink target changed")
        index = record["source_index"]
        expected[index["path"]] = index["sha256"]; kinds[index["path"]] = "source_index"
        if index["mapping_path"]:
            expected[index["mapping_path"]] = index["mapping_sha256"]
            kinds[index["mapping_path"]] = "path_mapping_index"
        chunk = record["committed_audit_chunk"]
        expected[chunk["path"]] = chunk["sha256"]; kinds[chunk["path"]] = "committed_audit_chunk"
    for entry in exclusion_sources:
        if entry.get("path") and entry.get("sha256"):
            expected[entry["path"]] = entry["sha256"]; kinds[entry["path"]] = "test_exclusion_source"
    total = sum(Path(path).stat().st_size for path in expected)
    if total > max_read_bytes:
        raise RuntimeError(f"Selected manifest verification would read {total} bytes, beyond explicit bounded budget {max_read_bytes}; selection is preserved, not replaced")
    for path, expected_hash in expected.items():
        before = Path(path).stat()
        actual = sha256_file(path)
        after = Path(path).stat()
        if (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns) != (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns):
            raise RuntimeError("Source changed during verification: "+path)
        if actual != expected_hash:
            raise RuntimeError("Frozen source checksum mismatch: "+path)
    return {"files_read":len(expected),"bytes_read":total,
            "selected_data_files":sum(v=="selected_image_or_depth" for v in kinds.values()),
            "metadata_and_chunk_files":sum(v!="selected_image_or_depth" for v in kinds.values()),
            "maximum_read_bytes":max_read_bytes,"all_selected_data_hashes_verified":bool(verify_selected_files)}


def build_smoke_manifest(config, source_run, output_dir, *, verify_selected_files=True,
                         max_read_bytes=2*1024**3):
    """Freeze 8+4 real engineering pairs; caller must first pass environment probe."""
    root = Path(config["repo_root"]).resolve()
    data_root = Path(config.get("data_root", root/"data")).absolute()
    source_run = Path(source_run).resolve();output_dir = Path(output_dir).resolve()
    if data_root != (root/"data").absolute():
        raise ValueError("Engineering data must come from existing SLiM repo_root/data")
    if not source_run.is_relative_to(root/"runs") or not output_dir.is_relative_to(root/"runs"):
        raise ValueError("Audit evidence and derived manifests must remain under SLiM runs")
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir/"engineering_smoke_manifest.json"
    dbpath = source_run/"data_audit.incomplete.sqlite"
    if not dbpath.exists():
        dbpath = source_run/"data_audit.sqlite"
    excluded, forbidden, exclusion_evidence, exclusion_sources = _exclusions(source_run)
    seed = int(config.get("seed", 2026))
    if seed != 2026:
        raise ValueError("This engineering selection is fixed to seed 2026")
    contract = label_contract()
    with _connect(dbpath) as db:
        source_identity, committed_count = _source_snapshot(db)
        if not excluded.issubset(set(source_identity["excluded_scenes"])):
            raise RuntimeError("Committed source audit lacks current test-scene exclusions")
        if path.exists():
            manifest = json.loads(path.read_text())
            content_hash = manifest.pop("content_sha256", None)
            if content_hash != digest(manifest):
                raise RuntimeError("Engineering smoke manifest content checksum mismatch")
            if manifest["schema"] != SCHEMA or manifest["formal_training_eligible"]:
                raise RuntimeError("Not an engineering-only manifest")
            if manifest["source_audit_identity"] != source_identity:
                raise RuntimeError("Source audit identity changed")
            if manifest["exclusion_evidence"] != exclusion_evidence or manifest["coordinate_contract_sha256"] != digest(contract):
                raise RuntimeError("Exclusions or coordinate contract changed")
            for split in ("train","val"):
                for record in manifest[split]:
                    actual = _record(db,record["pair_id"],source_run,data_root,record["split"])
                    if actual != record:
                        raise RuntimeError("Frozen engineering record changed")
            manifest["content_sha256"] = content_hash
            _verify_sources(manifest["train"]+manifest["val"],exclusion_sources,verify_selected_files,max_read_bytes)
            return manifest
        train_ids, val_ids, selection = _select_ids(db,excluded,forbidden,seed,8,4)
        train = [_record(db,p,source_run,data_root,"engineering_train") for p in train_ids]
        val = [_record(db,p,source_run,data_root,"engineering_val") for p in val_ids]
    train_scenes = {r["scene"] for r in train};val_scenes = {r["scene"] for r in val}
    if train_scenes & val_scenes or (train_scenes|val_scenes)&excluded:
        raise RuntimeError("Engineering split leaks scenes")
    train_images = {r["file_evidence"][f"image{s}"]["resolved"] for r in train for s in (0,1)}
    val_images = {r["file_evidence"][f"image{s}"]["resolved"] for r in val for s in (0,1)}
    if train_images & val_images:
        raise RuntimeError("Engineering split shares actual image files")
    verification = _verify_sources(train+val,exclusion_sources,verify_selected_files,max_read_bytes)
    manifest = dict(schema=SCHEMA,status="committed",scope="engineering_smoke_only",
                    formal_training_eligible=False,complete_data_audit_required_for_formal=True,
                    created_at=datetime.now(timezone(timedelta(hours=8))).isoformat(),
                    seed=seed,repo_root=str(root),data_root=str(data_root),source_run=str(source_run),
                    source_database=str(dbpath),source_committed_indices_at_selection=committed_count,
                    source_audit_identity=source_identity,exclusion_evidence=exclusion_evidence,
                    excluded_test_scenes=sorted(excluded),
                    test_pair_exclusion_policy="exclude every pair in every verified test scene; explicit matching pair IDs excluded additionally",
                    coordinate_contract=contract,coordinate_contract_sha256=digest(contract),
                    selection=selection,verification=verification,train=train,val=val,
                    train_pair_count=len(train),val_pair_count=len(val),
                    train_scenes=sorted(train_scenes),val_scenes=sorted(val_scenes))
    manifest = json.loads(canonical_json(manifest))
    manifest["content_sha256"] = digest(manifest)
    atomic_json(path,manifest)
    return manifest


class SmokeDataset:
    """Bounded adapter for existing engineering/calibration entrypoints."""
    formal_training_eligible = False
    scope = "engineering_smoke_only"

    def __init__(self, path, split):
        self.path = str(Path(path).resolve())
        self.manifest_path = self.path
        self.split = split
        if split not in ("train","val"):
            raise ValueError("Smoke split must be train or val")
        manifest = json.loads(Path(path).read_text())
        checksum = manifest.pop("content_sha256",None)
        if checksum != digest(manifest) or manifest["schema"] != SCHEMA or manifest["formal_training_eligible"]:
            raise RuntimeError("Invalid engineering-only smoke manifest")
        self.identity_sha256 = checksum
        self.records = manifest[split]

    def __len__(self):
        return len(self.records)

    def __getitem__(self,index):
        return self.records[index]

    def scene_round_robin_indices(self,count):
        # The manifest already freezes deterministic scene-round-robin order.
        return list(range(min(max(0,int(count)),len(self.records))))

    def close(self):
        pass
