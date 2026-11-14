"""Audited MegaDepth manifests and native-scale coarse correctness labels.

Protected inputs are read only. Every eligible pair comes from the actual SLiM
training list; the legacy test1500 validation source is never used for training.
"""
from __future__ import annotations
import argparse
import collections
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import sqlite3
import time
from functools import lru_cache

import cv2
import h5py
import numpy as np
import torch

SCHEMA = "slim_reliability_data_v1"
REASONS = {0:"valid",1:"source_outside",2:"source_depth_invalid",
           3:"behind_target_camera",4:"target_outside_or_nonfinite",
           5:"target_depth_invalid",6:"visibility_depth_inconsistent",
           7:"selected_target_outside"}
DEFAULT_ROOT = "/home/bupt/projects/SLiM-master"


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4*1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w") as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _npz(path):
    obj = np.load(path, allow_pickle=True)
    if isinstance(obj, dict):
        return obj
    try:
        return {k: obj[k] for k in obj.files}
    finally:
        obj.close()


def _scene(text):
    match = re.search(r"(?:^|/)(\d{4})(?:_|/|$)", str(text))
    if not match:
        raise ValueError("Cannot identify scene: " + str(text))
    return match.group(1)


def _flatten_scene_evidence(value, found):
    if isinstance(value, dict):
        for k, v in value.items():
            if k in ("scene", "scene_id", "scene_name") and isinstance(v, (str,int)):
                try: found.add(_scene(str(v).zfill(4)))
                except ValueError: pass
            _flatten_scene_evidence(v, found)
    elif isinstance(value, list):
        for v in value: _flatten_scene_evidence(v, found)
    elif isinstance(value, str):
        for m in re.finditer(r"(?:MegaDepth_v1|Undistorted_SfM)/(\d{4})/", value):
            found.add(m.group(1))


def test_scene_exclusions(root):
    excluded = {"0015", "0022"}
    evidence = []
    for indexbase in (root/"data/megadepth/index", root/"data/megadepth_indices"):
        for name in ("val_list.txt", "test_list.txt"):
            p = indexbase/"trainvaltest_list"/name
            if p.is_file():
                scenes = sorted({_scene(s) for s in p.read_text().splitlines() if s.strip()})
                excluded.update(scenes)
                evidence.append({"path":str(p),"sha256":sha256_file(p),"scenes":scenes})
        for d in ("scene_info_val_1500", "scene_info_val_1500_no_sfm"):
            for p in sorted((indexbase/d).glob("*.npz")):
                z = _npz(p)
                scenes = {_scene(p.stem)}
                for path in z.get("image_paths", []):
                    if path is not None:
                        try: scenes.add(_scene(path))
                        except ValueError: pass
                excluded.update(scenes)
                evidence.append({"path":str(p),"sha256":sha256_file(p),"scenes":sorted(scenes)})
    history = Path("/home/bupt/projects/hybrid_match/outputs")
    fixed_subset=history/"m2a_20261110T232950/subset300.json"
    if fixed_subset.is_file():
        value=json.loads(fixed_subset.read_text())
        scenes=set();_flatten_scene_evidence(value,scenes)
        excluded.update(scenes)
        evidence.append({"path":str(fixed_subset),"sha256":sha256_file(fixed_subset),"scenes":sorted(scenes),
                         "pair_count":len(value.get("pairs",[]))})
    # Explicitly inspect fixed-subset metadata, never treat unrelated histories as train data.
    for dirname in ("m2b1_cpu_20261112T133600", "m2b2_pilot_diagnosis_20261112T213600"):
        for p in sorted((history/dirname).rglob("*")):
            if not p.is_file() or p.suffix not in (".json",".jsonl"):
                continue
            if "pair_lists" not in p.parts and not any(x in p.name.lower() for x in ("manifest","subset","selection","pair","config")):
                continue
            if p.stat().st_size > 30*1024*1024:
                continue
            try:
                if p.suffix == ".jsonl":
                    values = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
                else: values = json.loads(p.read_text())
                scenes = set(); _flatten_scene_evidence(values, scenes)
                if scenes:
                    excluded.update(scenes)
                    evidence.append({"path":str(p),"sha256":sha256_file(p),"scenes":sorted(scenes)})
            except (ValueError, UnicodeError):
                continue
    return excluded, evidence


def audit_data(config, run_dir):
    """Audit all configured index pairs; freeze complete legal available manifests.

    Disk-backed normalized tables avoid retaining expanded millions of pairs.
    Candidate audit records reference hashed index rows and normalized frame/file
    tables, preserving exact source ordinal, direction, paths and exclusion reason.
    """
    root = Path(config.get("repo_root", DEFAULT_ROOT)).resolve()
    data_root = Path(config.get("data_root", str(root/"data"))).absolute()
    if data_root != (root/"data").absolute():
        raise ValueError("This run requires the existing SLiM repo_root/data")
    run = Path(run_dir).resolve(); run.mkdir(parents=True, exist_ok=True)
    finaldb = run/"data_audit.sqlite"
    summary_path = run/"data_audit_summary.json"
    if finaldb.exists() or summary_path.exists():
        raise FileExistsError("Frozen data audit already exists; do not silently rewrite identity")
    trainbase = data_root/"megadepth/train"
    indexbase = data_root/"megadepth/index"
    indexroot = Path(config.get("train_index_root", str(indexbase/"scene_info_0.1_0.7")))
    trainlist = Path(config.get("train_index", str(indexbase/"trainvaltest_list/train_list.txt")))
    if not trainlist.is_file(): raise FileNotFoundError(trainlist)
    stems = [x.strip().removesuffix(".npz") for x in trainlist.read_text().splitlines() if x.strip()]
    excluded, exclusion_evidence = test_scene_exclusions(root)
    atomic_json(run/"test_scene_exclusions.json", {"scenes":sorted(excluded),"sources":exclusion_evidence})
    # Record all links, including broken ones, without following them in traversal.
    links = []
    for base, dirs, files in os.walk(data_root, followlinks=False):
        for name in dirs+files:
            p = Path(base)/name
            if p.is_symlink():
                links.append({"logical":str(p),"target":os.readlink(p),"resolved":str(p.resolve()),
                              "exists":p.exists()})
    atomic_json(run/"data_paths.json", {
        "repo_root":str(root),"data_root":str(data_root),"data_root_resolved":str(data_root.resolve()),
        "train_data_root":str(trainbase),"train_index":str(trainlist),
        "train_index_root":str(indexroot),
        "legacy_val_index":str(indexbase/"trainvaltest_list/val_list.txt"),
        "legacy_val_rejected":"MegaDepth-1500 test scenes",
        "mapping":"existing no_sfm index path; identical depth/K/pose and image stem verified per frame; encoded extension may differ",
        "symlinks":links,
        "other_data_not_included":[p.name for p in data_root.iterdir() if p.name not in ("megadepth","megadepth_indices")]
    })
    free = __import__("shutil").disk_usage(run).free
    if free < 4*2**30: raise RuntimeError("Insufficient audit space plus 2 GiB reserve")
    dbpath = run/"data_audit.incomplete.sqlite"
    salvage=None
    if dbpath.exists():
        if not config.get("resume_audit",False):
            raise FileExistsError("Incomplete audit exists; explicit resume_audit required")
        probe=sqlite3.connect(dbpath)
        tables={r[0] for r in probe.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "audit_chunks" not in tables:
            # Preserve the interrupted first implementation; reuse only its
            # complete SHA256/readability file evidence, rebuild every index.
            salvage=run/"data_audit.interrupted_legacy.sqlite"
            if salvage.exists():raise FileExistsError(salvage)
            archived=sqlite3.connect(salvage);probe.backup(archived);archived.close()
            probe.close()
            os.rename(dbpath,run/"data_audit.interrupted_raw.sqlite")
            old_candidate=run/"candidate_audit.jsonl.gz.tmp"
            if old_candidate.exists():
                os.rename(old_candidate,run/"candidate_audit.interrupted_legacy.jsonl.gz")
            atomic_json(run/"data_audit_recovery.json",{
                "status":"running","reason":"Audit process ended without Python traceback after index commit",
                "source_db":str(salvage),"source_db_sha256":sha256_file(salvage),
                "policy":"Reuse complete file hashes/readability; rebuild all index/frame/pair records",
                "mapping_correction":"Accept same stem with exact no_sfm depth/K/pose even if PNG-to-JPG extension changes",
                "prior_outputs_preserved":True})
        else:
            probe.close()
    db = sqlite3.connect(dbpath)
    db.executescript("""
      PRAGMA journal_mode=WAL;
      PRAGMA synchronous=FULL;
      CREATE TABLE IF NOT EXISTS audit_chunks(index_id TEXT PRIMARY KEY,path TEXT,sha256 TEXT,counts TEXT);
      CREATE TABLE IF NOT EXISTS audit_identity(key TEXT PRIMARY KEY,value TEXT);
      CREATE TABLE IF NOT EXISTS files(path TEXT PRIMARY KEY, resolved TEXT, sha256 TEXT, size INTEGER,
                         status TEXT, reason TEXT, shape TEXT, detail TEXT);
      CREATE TABLE IF NOT EXISTS frames(id TEXT PRIMARY KEY, scene TEXT, image TEXT, depth TEXT,
                          K TEXT, pose TEXT, status TEXT, reason TEXT);
      CREATE TABLE IF NOT EXISTS indices(id TEXT PRIMARY KEY, path TEXT, sha256 TEXT, mapping_path TEXT,
                           mapping_sha256 TEXT, candidate_count INTEGER, status TEXT);
      CREATE TABLE IF NOT EXISTS pairs(id INTEGER PRIMARY KEY, pair_id TEXT UNIQUE, scene TEXT,
                         frame0 TEXT, frame1 TEXT, index_id TEXT, ordinal INTEGER,
                         overlap REAL, split TEXT);
      CREATE INDEX IF NOT EXISTS pairs_scene ON pairs(scene);
      CREATE TABLE IF NOT EXISTS manifests(split TEXT, position INTEGER, pair_row INTEGER,
                             PRIMARY KEY(split,position));
    """)
    if salvage is not None:
        db.execute("ATTACH DATABASE ? AS salvaged",(str(salvage),))
        db.execute("INSERT INTO files SELECT * FROM salvaged.files")
        db.commit();db.execute("DETACH DATABASE salvaged")
    identity=canonical_json({"train_list_sha256":sha256_file(trainlist),"stems":stems,
                             "excluded_scenes":sorted(excluded),"schema":SCHEMA,
                             "mapping":"same_stem_exact_depth_camera_v1"})
    old_identity=db.execute("SELECT value FROM audit_identity WHERE key='source'").fetchone()
    if old_identity and old_identity[0]!=identity:
        raise ValueError("Audit resume source identity differs")
    db.execute("INSERT OR IGNORE INTO audit_identity VALUES('source',?)",(identity,))
    filecache={}
    for row in db.execute("SELECT path,resolved,sha256,size,status,reason,shape,detail FROM files"):
        filecache[row[0]]={"path":row[0],"resolved":row[1],"sha256":row[2],"size":row[3],
                          "status":row[4],"reason":row[5],"shape":json.loads(row[6]),"detail":json.loads(row[7])}
    counts=collections.Counter()
    for status,n in db.execute("SELECT status,COUNT(*) FROM files GROUP BY status"):
        counts["file_"+status]=n
    for chunk in db.execute("SELECT path,sha256,counts FROM audit_chunks"):
        if sha256_file(chunk[0])!=chunk[1]:raise RuntimeError("Audit chunk checksum mismatch")
        for key,value in json.loads(chunk[2]).items():
            if not key.startswith("file_"):counts[key]+=value
    counts["missing_indices"]=db.execute("SELECT COUNT(*) FROM indices WHERE status='missing_index'").fetchone()[0]
    completed={r[0] for r in db.execute("SELECT index_id FROM audit_chunks")}
    for indexid,path,digest,mapping_path,mapping_digest in db.execute("SELECT id,path,sha256,mapping_path,mapping_sha256 FROM indices WHERE status='audited'"):
        if indexid in completed:
            if sha256_file(path)!=digest or (mapping_path and sha256_file(mapping_path)!=mapping_digest):
                raise RuntimeError("Frozen index changed before audit resume: "+indexid)
    completed.update(r[0] for r in db.execute("SELECT id FROM indices WHERE status='missing_index'"))
    db.commit()
    started = time.time()
    audit_path = run/"candidate_audit.jsonl.gz.tmp"
    chunkdir=run/"candidate_audit_chunks";chunkdir.mkdir(exist_ok=True)

    def progress(index_no, index_id):
        state = {"status":"running","index_position":index_no,"index_count":len(stems),
                 "index":index_id,"counts":dict(counts),"elapsed_s":time.time()-started,
                 "unique_files":len(filecache),"pid":os.getpid()}
        atomic_json(run/"data_audit_progress.json", state)
        print(canonical_json(state), flush=True)

    def auditfile(path, kind):
        key = str(path)
        if key in filecache: return filecache[key]
        result = {"path":key,"resolved":str(path.resolve()),"sha256":None,"size":0,
                  "status":"excluded","reason":"missing_file","shape":None,"detail":{}}
        if path.is_file():
            result["size"] = path.stat().st_size
            try:
                result["sha256"] = sha256_file(path)
                if kind == "image":
                    arr = cv2.imread(key, cv2.IMREAD_GRAYSCALE)
                    if arr is None or arr.ndim != 2 or min(arr.shape) < 1:
                        raise ValueError("image_decode_failed")
                    result["shape"] = list(arr.shape)
                else:
                    with h5py.File(path, "r") as h:
                        if "depth" not in h: raise ValueError("missing_depth_dataset")
                        arr = h["depth"][()]
                    if arr.ndim != 2 or min(arr.shape) < 1:
                        raise ValueError("depth_shape_invalid")
                    result["shape"] = list(arr.shape)
                    finite = np.isfinite(arr)
                    result["detail"] = {"finite_count":int(finite.sum()),
                        "positive_count":int(((arr > 0)&finite).sum()),"pixel_count":int(arr.size)}
                    if result["detail"]["positive_count"] == 0:
                        raise ValueError("no_valid_depth")
                result["status"] = "valid"; result["reason"] = None
                del arr
            except Exception as e:
                result["reason"] = type(e).__name__+":"+str(e)[:250]
        db.execute("INSERT INTO files VALUES(?,?,?,?,?,?,?,?)",
                   (key,result["resolved"],result["sha256"],result["size"],result["status"],
                    result["reason"],canonical_json(result["shape"]),canonical_json(result["detail"])))
        filecache[key] = result
        counts["file_"+result["status"]] += 1
        if len(filecache) % 500 == 0: progress(current_index, current_stem)
        return result

    for current_index, current_stem in enumerate(stems):
        if current_stem in completed:
            continue
        before_counts=counts.copy()
        p = indexroot/(current_stem+".npz")
        scene = _scene(current_stem)
        if not p.is_file():
            db.execute("INSERT INTO indices VALUES(?,?,?,?,?,?,?)",
                       (current_stem,str(p),None,None,None,0,"missing_index"))
            counts["missing_indices"] += 1
            db.commit()
            progress(current_index,current_stem)
            continue
        chunkpath=chunkdir/(current_stem+".jsonl.gz")
        chunktmp=chunkdir/(current_stem+".jsonl.gz.tmp")
        if chunkpath.exists():
            orphan=chunkpath.with_name(chunkpath.name+".uncommitted_"+str(time.time_ns()))
            os.rename(chunkpath,orphan)
        candidate_file=gzip.open(chunktmp,"wt",compresslevel=1)
        z = _npz(p)
        pair_infos = z["pair_infos"]
        mappath = indexbase/"scene_info_0.1_0.7_no_sfm"/p.name
        mapped = _npz(mappath) if mappath.is_file() else None
        db.execute("INSERT INTO indices VALUES(?,?,?,?,?,?,?)",
                   (current_stem,str(p),sha256_file(p),str(mappath) if mapped is not None else None,
                    sha256_file(mappath) if mapped is not None else None,len(pair_infos),"audited"))
        counts["indices_read"] += 1
        frame_lookup = {}
        used = sorted({int(i) for pair in pair_infos for i in pair[0]})
        # Test scenes are rejected before any image/depth reads.
        if scene not in excluded:
            for i in used:
                imgraw, depraw = z["image_paths"][i], z["depth_paths"][i]
                reason = None; K = pose = None
                if imgraw is None or depraw is None:
                    reason = "missing_index_path"
                    image = trainbase/("__missing_image_"+str(i)); depth = trainbase/("__missing_depth_"+str(i))
                else:
                    image = trainbase/str(imgraw); depth = trainbase/str(depraw)
                    if not image.exists() and mapped is not None:
                        try:
                            alternate = mapped["image_paths"][i]
                            assert alternate is not None
                            assert str(mapped["depth_paths"][i]) == str(depraw)
                            assert Path(str(alternate)).stem == Path(str(imgraw)).stem
                            assert np.array_equal(z["intrinsics"][i],mapped["intrinsics"][i])
                            assert np.array_equal(z["poses"][i],mapped["poses"][i])
                            image = trainbase/str(alternate)
                        except (AssertionError,IndexError,KeyError):
                            reason = "path_mapping_metadata_mismatch"
                    # Reject absolute/traversal source paths; existing symlinks remain valid audited inputs.
                    if not image.absolute().is_relative_to(trainbase.absolute()) or not depth.absolute().is_relative_to(trainbase.absolute()):
                        reason = "path_outside_logical_data_root"
                try:
                    K = np.asarray(z["intrinsics"][i], dtype=np.float64).reshape(3,3)
                    pose = np.asarray(z["poses"][i], dtype=np.float64).reshape(4,4)
                    if not np.isfinite(K).all() or not np.isfinite(pose).all():
                        raise ValueError("nonfinite_camera")
                    if abs(np.linalg.det(K)) < 1e-10 or abs(np.linalg.det(pose)) < 1e-10:
                        raise ValueError("singular_camera")
                    if K[0,0] <= 0 or K[1,1] <= 0:
                        raise ValueError("invalid_focal")
                except (ValueError,TypeError):
                    reason = reason or "invalid_camera"
                    K = pose = None
                identity = {"scene":scene,"image":str(image),"depth":str(depth),
                            "K":None if K is None else K.tolist(),"pose":None if pose is None else pose.tolist()}
                frameid = hashlib.sha256(canonical_json(identity).encode()).hexdigest()
                if reason is None:
                    ir, dr = auditfile(image,"image"), auditfile(depth,"depth")
                    if ir["status"] != "valid": reason = "image:"+str(ir["reason"])
                    elif dr["status"] != "valid": reason = "depth:"+str(dr["reason"])
                db.execute("INSERT OR IGNORE INTO frames VALUES(?,?,?,?,?,?,?,?)",
                           (frameid,scene,str(image),str(depth),canonical_json(identity["K"]),
                            canonical_json(identity["pose"]),"valid" if reason is None else "excluded",reason))
                frame_lookup[i] = (frameid,reason)
        for ordinal, pair in enumerate(pair_infos):
            (i,j), overlap, _ = pair
            i,j=int(i),int(j)
            counts["candidate_pairs"] += 1
            if scene in excluded:
                reason = "test_scene"
                f0=f1=None
            else:
                f0,r0=frame_lookup[i];f1,r1=frame_lookup[j]
                reason = r0 or r1
                if i == j: reason = "self_pair"
                if not math.isfinite(float(overlap)) or float(overlap) <= 0:
                    reason = "native_overlap_filter"
            pair_id = hashlib.sha256((str(f0)+"|"+str(f1)+"|0to1").encode()).hexdigest()[:32] if f0 else None
            if reason is None:
                try:
                    db.execute("INSERT INTO pairs(pair_id,scene,frame0,frame1,index_id,ordinal,overlap) VALUES(?,?,?,?,?,?,?)",
                               (pair_id,scene,f0,f1,current_stem,ordinal,float(overlap)))
                    counts["legal_available_pairs"] += 1
                except sqlite3.IntegrityError:
                    reason = "duplicate_directed_pair"
            counts["excluded_"+reason if reason else "included"] += 1
            candidate_file.write(canonical_json({
                "index_id":current_stem,"ordinal":ordinal,"scene":scene,"idx0":i,"idx1":j,
                "direction":"0to1","pair_id":pair_id,"frame0":f0,"frame1":f1,
                "status":"included" if reason is None else "excluded","reason":reason})+"\n")
        candidate_file.close()
        with chunktmp.open("rb") as f:os.fsync(f.fileno())
        os.replace(chunktmp,chunkpath)
        delta={key:value-before_counts.get(key,0) for key,value in counts.items()}
        db.execute("INSERT INTO audit_chunks VALUES(?,?,?,?)",
                   (current_stem,str(chunkpath),sha256_file(chunkpath),canonical_json(delta)))
        db.commit()
        del z, mapped, pair_infos, frame_lookup
        progress(current_index+1,current_stem)
        if __import__("shutil").disk_usage(run).free < 2*2**30:
            raise RuntimeError("Data audit stopped before violating 2 GiB reserve")
    with audit_path.open("wb") as combined:
        for stem in stems:
            entry=db.execute("SELECT path FROM audit_chunks WHERE index_id=?",(stem,)).fetchone()
            if entry:
                with open(entry[0],"rb") as part:__import__("shutil").copyfileobj(part,combined,4*1024*1024)
        combined.flush();os.fsync(combined.fileno())
    scenes = sorted(r[0] for r in db.execute("SELECT DISTINCT scene FROM pairs"))
    if len(scenes) < 2:
        split_status = "blocked"; vals = []
    else:
        ordered = scenes.copy(); random.Random(2026).shuffle(ordered)
        nval=max(1,min(len(ordered)-1,int(round(len(ordered)*0.1))))
        vals=sorted(ordered[:nval]);split_status="committed"
    db.execute("UPDATE pairs SET split='train'")
    for s in vals: db.execute("UPDATE pairs SET split='val' WHERE scene=?", (s,))
    db.execute("DELETE FROM manifests")
    manifest_hashes={}
    split_counts={}
    for split in ("train","val"):
        dst=run/(split+"_manifest.jsonl");tmp=dst.with_name(dst.name+".tmp")
        n=0
        with tmp.open("w") as f:
            for row in db.execute("SELECT id,pair_id,scene,frame0,frame1,index_id,ordinal FROM pairs WHERE split=? ORDER BY scene,pair_id",(split,)):
                rid,pairid,scene,f0,f1,indexid,ordinal=row
                record={"pair_id":pairid,"scene":scene,"frame0":f0,"frame1":f1,
                        "index_id":indexid,"ordinal":ordinal,"direction":"0to1"}
                f.write(canonical_json(record)+"\n")
                db.execute("INSERT INTO manifests VALUES(?,?,?)",(split,n,rid))
                n+=1
            f.flush();os.fsync(f.fileno())
        os.replace(tmp,dst);manifest_hashes[split]=sha256_file(dst);split_counts[split]=n
    db.commit()
    file_bytes=db.execute("SELECT COALESCE(SUM(size),0) FROM files").fetchone()[0]
    checkpoint=db.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
    if checkpoint[0]!=0 or checkpoint[1]!=0 or checkpoint[2]!=0:
        raise RuntimeError("Refusing manifest publication: SQLite WAL checkpoint incomplete "+str(checkpoint))
    journal=db.execute("PRAGMA journal_mode=DELETE").fetchone()[0]
    if str(journal).lower()!="delete":
        raise RuntimeError("Refusing manifest publication: unable to leave WAL mode")
    db.close()
    for suffix in ("-wal","-shm"):
        if Path(str(dbpath)+suffix).exists():
            raise RuntimeError("Refusing manifest publication: SQLite sidecar remains "+suffix)
    os.replace(dbpath,finaldb)
    os.replace(audit_path,run/"candidate_audit.jsonl.gz")
    summary={"schema":SCHEMA,"status":split_status,"dataset":"MegaDepth","scope":"full_available",
             "official_complete_dataset_claim":False,"seed":2026,"counts":dict(counts),
             "train_count":split_counts["train"],"val_count":split_counts["val"],
             "train_scenes":[s for s in scenes if s not in vals],"val_scenes":vals,
             "excluded_test_scenes":sorted(excluded),"file_bytes":file_bytes,
             "elapsed_s":time.time()-started,"manifest_sha256":manifest_hashes,
             "data_audit_db":str(finaldb),"data_audit_sha256":sha256_file(finaldb),
             "candidate_audit_sha256":sha256_file(run/"candidate_audit.jsonl.gz"),
             "train_list_sha256":sha256_file(trainlist),
             "free_disk_bytes":__import__("shutil").disk_usage(run).free,
             "split_policy":"scene shuffle Python Random(2026); held out round(0.1*scenes), at least one",
             "legacy_val_rejected":True}
    atomic_json(summary_path,summary)
    atomic_json(run/"feature_label_contract.json",label_contract())
    atomic_json(run/"data_audit_progress.json",{"status":"committed","counts":dict(counts),"elapsed_s":time.time()-started})
    return summary


class ManifestDataset:
    """Disk-backed sequence of frozen records with O(1) random indexed access."""
    def __init__(self, path):
        path=Path(path).resolve()
        self.path=str(path)
        self.run_dir=path.parent
        self.split=path.name.split("_manifest")[0]
        if self.split not in ("train","val"): raise ValueError(path)
        self.connection=sqlite3.connect(f"file:{self.run_dir/'data_audit.sqlite'}?mode=ro", uri=True)
        self.length=self.connection.execute("SELECT COUNT(*) FROM manifests WHERE split=?",(self.split,)).fetchone()[0]
    def __len__(self): return self.length
    def __getitem__(self,index):
        index=int(index)
        if index<0:index+=self.length
        if not 0<=index<self.length: raise IndexError(index)
        row=self.connection.execute("""SELECT p.pair_id,p.scene,p.frame0,p.frame1,p.index_id,p.ordinal
              FROM manifests m JOIN pairs p ON p.id=m.pair_row WHERE m.split=? AND m.position=?""",(self.split,index)).fetchone()
        pair_id,scene,f0,f1,indexid,ordinal=row
        record={"pair_id":pair_id,"scene":scene,"index_id":indexid,"ordinal":ordinal,"direction":"0to1",
                "frame0":f0,"frame1":f1,"manifest_position":index,"split":self.split,"audit_dir":str(self.run_dir)}
        record["file_sha256"]={}
        for side,fid in enumerate((f0,f1)):
            r=self.connection.execute("SELECT image,depth,K,pose FROM frames WHERE id=?",(fid,)).fetchone()
            record[f"image{side}"],record[f"depth{side}"]=r[:2]
            record[f"K{side}"]=json.loads(r[2]);record[f"pose{side}"]=json.loads(r[3])
            for kind in ("image","depth"):
                pathkey=f"{kind}{side}"
                h=self.connection.execute("SELECT sha256,status FROM files WHERE path=?",(record[pathkey],)).fetchone()
                if h is None or h[1]!="valid" or h[0] is None:
                    raise RuntimeError("Manifest references unaudited file: "+record[pathkey])
                record["file_sha256"][pathkey]=h[0]
        return record
    def scene_round_robin_indices(self, count):
        # Manifest positions were frozen in ORDER BY scene,pair_id. Aggregate
        # scene sizes once instead of scanning a multi-million-pair join per scene.
        count=min(max(0,int(count)),len(self))
        if count==0:return []
        if not hasattr(self,"_scene_ranges"):
            offset=0;ranges=[]
            for scene,size in self.connection.execute("SELECT scene,COUNT(*) FROM pairs WHERE split=? GROUP BY scene ORDER BY scene",(self.split,)):
                ranges.append((offset,int(size)));offset+=int(size)
            if offset!=len(self):raise RuntimeError("Manifest scene coverage mismatch")
            self._scene_ranges=ranges
        output=[]
        for depth in range(count):
            for start,size in self._scene_ranges:
                if depth<size:output.append(start+depth)
                if len(output)>=count:return output
        return output
    def close(self): self.connection.close()


def iter_manifest(path):
    dataset=ManifestDataset(path)
    try:
        for i in range(len(dataset)):yield dataset[i]
    finally:dataset.close()


def plan_geometry(orig_hw, resolution=832):
    h,w=map(int,orig_hw); scale=float(resolution)/max(h,w)
    rh,rw=int(round(h*scale)),int(round(w*scale))
    sx,sy=rw/w,rh/h
    return {"orig_hw":[h,w],"resized_hw":[rh,rw],"input_hw":[resolution,resolution],"pad_ltrb":[0,0,resolution-rw,resolution-rh],
            "orig_to_input":[[sx,0.,0.],[0.,sy,0.],[0.,0.,1.]],
            "input_to_orig":[[1./sx,0.,0.],[0.,1./sy,0.],[0.,0.,1.]],
            "scale_input_to_orig_xy":[1./sx,1./sy],
            "coordinate_convention":"native_scale_only_xy_integer_grid_origin_0"}


_VERIFIED_FILES = {}


def verify_frozen_file(path, expected_sha256):
    """Rehash first use or stat change, rejecting frozen-input changes."""
    if not isinstance(expected_sha256,str) or len(expected_sha256)!=64:
        raise ValueError("Missing frozen file SHA256: "+str(path))
    path=Path(path)
    st=path.stat()
    signature=(st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns)
    key=str(path)
    if _VERIFIED_FILES.get(key)==(signature,expected_sha256):
        return
    actual=sha256_file(path)
    after=path.stat()
    signature_after=(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns)
    if signature_after!=signature:
        raise RuntimeError("Frozen file changed during checksum verification: "+key)
    if actual!=expected_sha256:
        raise RuntimeError("Frozen data checksum mismatch: "+key)
    _VERIFIED_FILES[key]=(signature,expected_sha256)


def load_pair(record, device="cuda", resolution=832, include_gt=True):
    if int(resolution)!=832: raise ValueError("This experiment is fixed to R832")
    file_keys=("image0","image1","depth0","depth1") if include_gt else ("image0","image1")
    for key in file_keys:
        verify_frozen_file(record[key],record["file_sha256"][key])
    result={"pair_id":record["pair_id"],"scene_id":record["scene"],"direction":"0to1"}
    gt={}
    for side in (0,1):
        raw=cv2.imread(record[f"image{side}"],cv2.IMREAD_GRAYSCALE)
        if raw is None:raise ValueError("Frozen image became unreadable: "+record[f"image{side}"])
        plan=plan_geometry(raw.shape,resolution)
        rh,rw=plan["resized_hw"]
        resized=cv2.resize(raw,(rw,rh),interpolation=cv2.INTER_LINEAR)
        canvas=np.zeros((resolution,resolution),np.float32);canvas[:rh,:rw]=resized.astype(np.float32)/255.
        mask=np.zeros((resolution,resolution),bool);mask[:rh,:rw]=True
        image=torch.from_numpy(canvas)[None,None].to(device)
        result[f"image{side}"]=image
        result[f"valid_mask{side}"]=torch.from_numpy(mask)[None].to(device)
        result[f"mask{side}"]=torch.from_numpy(mask[::8,::8].copy())[None].to(device)
        # The canonical inference wrapper omits native scale0/1, keeping all
        # model coordinates in INPUT pixels. Geometry maps to original once.
        result[f"input_to_orig_scale{side}"]=torch.tensor(plan["scale_input_to_orig_xy"],dtype=torch.float32,device=device)[None]
        result[f"preprocessing_plan{side}"]=plan
        if not include_gt:
            continue
        with h5py.File(record[f"depth{side}"],"r") as h: depth=h["depth"][()]
        gt[f"depth{side}"]=depth
        gt[f"plan{side}"]=plan
        gt[f"K{side}"]=np.asarray(record[f"K{side}"],np.float64)
        gt[f"pose{side}"]=np.asarray(record[f"pose{side}"],np.float64)
    result["model_coordinate_space"]="input"
    if not include_gt:
        return result
    gt["T_0to1"]=gt["pose1"]@np.linalg.inv(gt["pose0"])
    gt["depth_absolute_tolerance"]=0.01;gt["depth_relative_tolerance"]=0.05
    result["_gt"]=gt
    result["K0"]=torch.as_tensor(gt["K0"],dtype=torch.float32,device=device)[None]
    result["K1"]=torch.as_tensor(gt["K1"],dtype=torch.float32,device=device)[None]
    result["T_0to1"]=torch.as_tensor(gt["T_0to1"],dtype=torch.float32,device=device)[None]
    return result


def _numpy(x):
    return x.detach().cpu().numpy() if isinstance(x,torch.Tensor) else np.asarray(x)


def project_visible(points0_input, data):
    """Canonical projection for arbitrary INPUT source points, never clamped.

    Adapted from hybrid_match/common/gt_coarse.py::project_visible; source contract
    records hash. Nearest depth uses numpy rint (half-to-even) and original K.
    """
    points0_input=np.asarray(_numpy(points0_input),np.float64).reshape(-1,2)
    g=data["_gt"];plan0,plan1=g["plan0"],g["plan1"]
    depth0,depth1=g["depth0"],g["depth1"];K0,K1=g["K0"],g["K1"];T=g["T_0to1"]
    scale0=np.asarray(plan0["scale_input_to_orig_xy"])
    scale1=np.asarray(plan1["scale_input_to_orig_xy"])
    points0=points0_input*scale0
    n=len(points0);reasons=np.zeros(n,np.int32)
    h0,w0=plan0["orig_hw"];rh0,rw0=plan0["resized_hw"];rh1,rw1=plan1["resized_hw"]
    dp=points0*np.array([depth0.shape[1]/w0,depth0.shape[0]/h0])
    finite0=np.isfinite(dp).all(axis=1)
    x=np.rint(np.where(finite0,dp[:,0],0)).astype(np.int64)
    y=np.rint(np.where(finite0,dp[:,1],0)).astype(np.int64)
    valid=finite0&(points0_input[:,0]>=0)&(points0_input[:,1]>=0)&(points0_input[:,0]<rw0)&(points0_input[:,1]<rh0)&(x>=0)&(y>=0)&(x<depth0.shape[1])&(y<depth0.shape[0])
    reasons[~valid]=1
    d=np.zeros(n);d[valid]=depth0[y[valid],x[valid]]
    bad=valid&((d<=0)|~np.isfinite(d));reasons[bad]=2;valid&=~bad
    rays=np.column_stack([points0,np.ones(n)])@np.linalg.inv(K0).T
    target=(rays*d[:,None])@T[:3,:3].T+T[:3,3]
    bad=valid&((target[:,2]<=0)|~np.isfinite(target).all(axis=1));reasons[bad]=3;valid&=~bad
    uvh=target@K1.T
    with np.errstate(divide="ignore",invalid="ignore"):uv=uvh[:,:2]/uvh[:,2:]
    uv_input=uv/scale1
    finite=np.isfinite(uv).all(axis=1)
    h1,w1=plan1["orig_hw"]
    depth_uv=uv*np.array([depth1.shape[1]/w1,depth1.shape[0]/h1])
    xx=np.rint(np.where(finite,depth_uv[:,0],0)).astype(np.int64)
    yy=np.rint(np.where(finite,depth_uv[:,1],0)).astype(np.int64)
    inside=finite&(uv_input[:,0]>=0)&(uv_input[:,1]>=0)&(uv_input[:,0]<rw1)&(uv_input[:,1]<rh1)&(xx>=0)&(yy>=0)&(xx<depth1.shape[1])&(yy<depth1.shape[0])
    bad=valid&~inside;reasons[bad]=4;valid&=inside
    observed=np.zeros(n);observed[valid]=depth1[yy[valid],xx[valid]]
    bad=valid&((observed<=0)|~np.isfinite(observed));reasons[bad]=5;valid&=~bad
    tolerance=g.get("depth_absolute_tolerance",.01)+g.get("depth_relative_tolerance",.05)*observed
    bad=valid&(np.abs(observed-target[:,2])>tolerance);reasons[bad]=6;valid&=~bad
    return {"points1_gt_input":uv_input,"points1_gt_orig":uv,"valid_gt":valid,
            "invalid_reason":reasons,"target_depth":target[:,2],"sampled_depth1":observed}


def assert_parent_anchor_contract(data, points0, points1):
    """Production guard against original/input units or half-cell corruption."""
    has_ids=("i_idx_c" in data,"j_idx_c" in data)
    if not any(has_ids):
        return False
    if not all(has_ids):
        raise ValueError("Coarse anchor contract: both source/target grid IDs are required")
    if data["image0"].shape[0]!=1 or data["image1"].shape[0]!=1:
        raise ValueError("Coarse anchor contract requires batch_size=1")
    count=len(points0)
    batches=_numpy(data.get("b_idx_c",np.zeros(count,dtype=np.int64))).reshape(-1)
    if len(batches)!=count or np.any(batches!=0):
        raise ValueError("Coarse anchor contract: invalid batch IDs")
    ids_by_side=[]
    sizes=[]
    for side,points,key in ((0,points0,"i_idx_c"),(1,points1,"j_idx_c")):
        ids=_numpy(data[key]).reshape(-1)
        hi,wi=map(int,data[f"image{side}"].shape[-2:])
        hc,wc=map(int,data[f"hw{side}_c"])
        if len(ids)!=count or not np.issubdtype(ids.dtype,np.integer):
            raise ValueError("Coarse anchor contract: noninteger or misaligned grid IDs")
        if np.any((ids<0)|(ids>=hc*wc)):
            raise ValueError("Coarse anchor contract: grid IDs outside feature grid")
        stride=np.array([wi/wc,hi/hc],dtype=np.float64)
        expected=(np.column_stack((ids%wc,ids//wc))*stride).astype(points.dtype)
        if not np.array_equal(points,expected):
            raise ValueError("Coarse anchor contract: coordinates differ from input grid anchors on side "+str(side))
        mask=data.get(f"mask{side}")
        if mask is None:
            raise ValueError("Coarse anchor contract: missing native valid coarse mask")
        flat=_numpy(mask).reshape(-1).astype(bool)
        if len(flat)!=hc*wc or not flat[ids].all():
            raise ValueError("Coarse anchor contract: selected anchor falls in padding")
        ids_by_side.append(ids.astype(np.int64));sizes.append(hc*wc)
    expected_raw=ids_by_side[0]*sizes[1]+ids_by_side[1]
    if not np.array_equal(_numpy(data["coarse_match_id"]).reshape(-1),expected_raw):
        raise ValueError("Coarse anchor contract: raw parent identity differs from actual grid pair")
    return True


def label_parents(data):
    if "scale0" in data or "scale1" in data:
        raise ValueError("Canonical label path requires INPUT model coordinates: omit native scale0/1")
    points0=_numpy(data["coarse_coord_0"]).reshape(-1,2)
    points1=_numpy(data["coarse_coord_1"]).reshape(-1,2)
    if len(points0)!=len(points1):raise ValueError("Parent coordinate count mismatch")
    contract_checked=assert_parent_anchor_contract(data,points0,points1)
    projection=project_visible(points0,data)
    valid=projection["valid_gt"].copy();reasons=projection["invalid_reason"].copy()
    rh,rw=data["_gt"]["plan1"]["resized_hw"]
    selected_valid=np.isfinite(points1).all(axis=1)&(points1[:,0]>=0)&(points1[:,1]>=0)&(points1[:,0]<rw)&(points1[:,1]<rh)
    bad=valid&~selected_valid;reasons[bad]=7;valid&=~bad
    hi,wi=data["image1"].shape[-2:];hc,wc=data["hw1_c"]
    stride=np.array([wi/wc,hi/hc],dtype=np.float64)
    residual=projection["points1_gt_input"]-points1
    y=(np.abs(residual)<=stride/2).all(axis=1)
    labels=np.full(len(y),-1,np.int8);labels[valid]=y[valid].astype(np.int8)
    raw_ids=_numpy(data["coarse_match_id"]).copy()
    if len(raw_ids)!=len(y) or len(np.unique(raw_ids))!=len(y):
        raise ValueError("Missing, duplicate or misaligned raw coarse parent identity")
    device=data["coarse_coord_0"].device if isinstance(data["coarse_coord_0"],torch.Tensor) else "cpu"
    return dict(projection,contract_checked=contract_checked,labels=labels,y=torch.as_tensor(y,dtype=torch.float32,device=device),
                valid=torch.as_tensor(valid,dtype=torch.bool,device=device),reasons=reasons,
                raw_coarse_match_id=raw_ids,raw_coarse_ids=raw_ids,
                pair_id=data.get("pair_id"),direction=data.get("direction","0to1"),
                residual_input=residual,stride_xy=stride,ignore_counts=dict(collections.Counter(REASONS[int(x)] for x in reasons[~valid])))


def label_contract():
    source=Path("/home/bupt/projects/hybrid_match/common/gt_coarse.py")
    return {"schema":SCHEMA,"target":"coarse parent correctness within inclusive per-axis half stride",
            "feature_layout":"raw post-interaction [B,C,Hc,Wc], before correlation /sqrt(C)",
            "anchor":"actual native coarse_coord_0/1 INPUT top-left grid anchor; no half-stride offset",
            "model_scale_policy":"native scale0/scale1 keys omitted, matching approved canonical inference wrapper; original mapping only in GT/evaluation",
            "projection":"T1 @ inv(T0), original image K, source input->original scale",
            "depth_sample":"nearest np.rint half-to-even; original-image-to-depth shape scale",
            "visibility":{"absolute":0.01,"relative":0.05,"reference":"observed target depth"},
            "outside_policy":"ignore; no boundary clamp or negative labels for unconfirmed visibility",
            "ignore_reasons":REASONS,"boundary_comparison":"abs(residual_input_xy) <= stride_xy/2",
            "original_gt_source":str(source),"original_gt_source_sha256":sha256_file(source) if source.exists() else None,
            "preprocessing":"OpenCV grayscale /255, INTER_LINEAR, Python round long-side832, bottom/right zero pad",
            "augmentation":False,"join":"pair_id + direction + raw coarse_match_id; fine children inherit parent ID"}


def projection_diagnostics(data, labels):
    """Fixed-grid coverage denominator and original-space canonical geometry."""
    h0,w0=map(int,data["hw0_c"])
    ih0,iw0=map(int,data["image0"].shape[-2:])
    stride=np.array([iw0/w0,ih0/h0],dtype=np.float64)
    gridids=np.arange(h0*w0,dtype=np.int64)
    anchors=np.column_stack((gridids%w0,gridids//w0))*stride
    plan0=data["_gt"]["plan0"]
    rh,rw=plan0["resized_hw"]
    gridvalid=(anchors[:,0]>=0)&(anchors[:,1]>=0)&(anchors[:,0]<rw)&(anchors[:,1]<rh)
    gridproj=project_visible(anchors,data)
    fine0=_numpy(data.get("fine_coord_0",np.empty((0,2)))).reshape(-1,2)
    fine1=_numpy(data.get("fine_coord_1",np.empty((0,2)))).reshape(-1,2)
    if len(fine0)!=len(fine1): raise ValueError("Final point counts differ")
    fineproj=project_visible(fine0,data)
    scale0=np.asarray(plan0["scale_input_to_orig_xy"])
    scale1=np.asarray(data["_gt"]["plan1"]["scale_input_to_orig_xy"])
    result={"source_grid_valid":gridvalid,
            "source_grid_valid_gt":gridproj["valid_gt"]&gridvalid,
            "source_grid_ignore_reason":gridproj["invalid_reason"],
            "coarse_source_ids":_numpy(data["i_idx_c"]).astype(np.int64),
            "labels":labels["labels"],"source_grid_hw":(h0,w0),
            "coarse_stride_xy":stride,"final_source_input":fine0,
            "final_target_input":fine1,"final_gt_target_input":fineproj["points1_gt_input"],
            "final_gt_valid":fineproj["valid_gt"],"points0_orig":fine0*scale0,
            "points1_orig":fine1*scale1,"K0":data["_gt"]["K0"],
            "K1":data["_gt"]["K1"],"T_0to1":data["_gt"]["T_0to1"]}
    if "source_grid_low_texture" in data:
        low=_numpy(data["source_grid_low_texture"]).reshape(-1).astype(bool)
        if len(low)!=len(gridvalid):raise ValueError("Local texture grid shape mismatch")
        result["source_grid_low_texture"]=low&gridvalid
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--run-dir",required=True)
    parser.add_argument("--config")
    parser.add_argument("--resume-audit",action="store_true")
    args=parser.parse_args()
    config=json.loads(Path(args.config).read_text()) if args.config else {}
    config["resume_audit"]=args.resume_audit
    print(json.dumps(audit_data(config,args.run_dir),indent=2))
