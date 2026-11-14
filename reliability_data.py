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


def _npz(path, fields=None):
    obj = np.load(path, allow_pickle=True)
    if isinstance(obj, dict):
        return obj if fields is None else {k:obj[k] for k in fields}
    try:
        return {k: obj[k] for k in (obj.files if fields is None else fields)}
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
                z = _npz(p,("image_paths",))
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
                scenes=set()
                if p.suffix == ".jsonl":
                    with p.open() as handle:
                        for line in handle:
                            if line.strip():_flatten_scene_evidence(json.loads(line),scenes)
                else:_flatten_scene_evidence(json.loads(p.read_text()),scenes)
                if scenes:
                    excluded.update(scenes)
                    evidence.append({"path":str(p),"sha256":sha256_file(p),"scenes":sorted(scenes)})
            except (ValueError, UnicodeError):
                continue
    return excluded, evidence



def _audit_resources():
    import resource
    info={}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key,rest=line.split(":",1)
        if rest.strip().endswith("kB"):info[key]=int(rest.split()[0])*1024
    return {"rss_bytes":int(Path("/proc/self/statm").read_text().split()[1])*os.sysconf("SC_PAGE_SIZE"),
            "rss_peak_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            "available_bytes":info.get("MemAvailable"),"swap_used_bytes":info.get("SwapTotal",0)-info.get("SwapFree",0),
            "boot_id":Path("/proc/sys/kernel/random/boot_id").read_text().strip()}


# Native pair_infos is a pickled object ndarray and NumPy cannot mmap it.
# Decode it exactly once in an isolated address-space-capped process, retain no
# unused 3D/central-match arrays afterward, and normalize pairs to disk batches.
# An index which cannot fit this cap is BLOCKED, never silently shortened.
_INDEX_DECODE_PROGRAM = r'''
import os,sys,json,sqlite3,resource,time
request=json.loads(sys.argv[1])
cap=int(request["memory_limit_bytes"])
resource.setrlimit(resource.RLIMIT_AS,(cap,cap))
import numpy as np
start=time.monotonic()
db=sqlite3.connect(request["temporary"])
db.executescript("PRAGMA journal_mode=DELETE;PRAGMA synchronous=FULL;PRAGMA cache_size=-4096;PRAGMA temp_store=FILE;CREATE TABLE frames(idx INTEGER PRIMARY KEY,payload TEXT);CREATE TABLE pairs(ordinal INTEGER PRIMARY KEY,i INTEGER,j INTEGER,overlap TEXT);")
keys=("image_paths","depth_paths","intrinsics","poses")
def fields(path,include_pairs=False):
    # Real source files have an .npz suffix but are protocol-4 pickled dicts.
    # Their entire dictionary must unpickle once within the hard process cap;
    # unused 3D fields are dropped immediately after that unavoidable peak.
    obj=np.load(path,allow_pickle=True)
    wanted=keys+(("pair_infos",) if include_pairs else ())
    if isinstance(obj,dict):
        selected={k:obj[k] for k in wanted}
        return selected,"raw_pickle_dict_whole_decode_within_cap"
    try:return {k:obj[k] for k in wanted},"npz_selected_fields_within_cap"
    finally:obj.close()
def value(v):
    if isinstance(v,np.ndarray):return v.tolist()
    if isinstance(v,np.generic):return v.item()
    return v
mapping_format=None
# Normalize and release mapping metadata before loading the native dictionary,
# avoiding simultaneous expansion of both native and mapped pair arrays.
db.execute("CREATE TABLE mapped_frames(idx INTEGER PRIMARY KEY,payload TEXT)")
if request["mapping"]:
    mapped,mapping_format=fields(request["mapping"])
    for i in range(min(len(mapped[k]) for k in keys)):
        row={k:value(mapped[k][i]) for k in keys}
        db.execute("INSERT INTO mapped_frames VALUES(?,?)",(i,json.dumps(row,allow_nan=True,separators=(",",":"))))
        if i%2048==2047:db.commit()
    db.commit();del mapped
native,source_format=fields(request["source"],include_pairs=True)
pair_infos=native.pop("pair_infos")
for i in range(len(native["image_paths"])):
    row={k:value(native[k][i]) for k in keys}
    mapped_row=db.execute("SELECT payload FROM mapped_frames WHERE idx=?",(i,)).fetchone()
    row["mapped"]=json.loads(mapped_row[0]) if mapped_row else None
    db.execute("INSERT INTO frames VALUES(?,?)",(i,json.dumps(row,allow_nan=True,separators=(",",":"))))
    if i%2048==2047:db.commit()
db.execute("DROP TABLE mapped_frames");del native
count=len(pair_infos)
for ordinal,pair in enumerate(pair_infos):
    (i,j),overlap,_=pair
    i,j=int(i),int(j)
    db.execute("INSERT INTO pairs VALUES(?,?,?,?)",(ordinal,i,j,repr(float(overlap))))
    # Release nested central-match arrays once consumed; no second pair list.
    pair_infos[ordinal]=None
    if ordinal%2048==2047:db.commit()
del pair_infos
db.commit();db.close()
record={"status":"committed","pair_count":count,"address_space_limit_bytes":cap,
        "peak_rss_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        "elapsed_seconds":time.monotonic()-start,"source_format":source_format,"mapping_format":mapping_format,
         "source_loads":1,"mapping_loads":int(bool(request["mapping"])),
         "native_pickle_decode":"raw pickle requires whole dict; NPZ loads selected fields; every expansion within hard address-space cap"}
with open(request["measurement"],"w") as f:json.dump(record,f,indent=2);f.flush();os.fsync(f.fileno())
'''


def _decode_index_bounded(source,mapping,run,config,source_sha,mapping_sha):
    import subprocess,sys
    folder=Path(run)/"bounded_index_cache";folder.mkdir(exist_ok=True)
    key=hashlib.sha256(canonical_json({"source":source_sha,"mapping":mapping_sha,"schema":"normalized_index_v2"}).encode()).hexdigest()
    final=folder/(key+".sqlite");receipt=folder/(key+".json")
    cap=int(config.get("audit_decode_memory_bytes",1024**3))
    if cap<256*1024**2 or cap>2*1024**3:
        raise ValueError("Registered bounded decoder cap must be 256MiB..2GiB")
    expected={"source_sha256":source_sha,"mapping_sha256":mapping_sha,"schema":"normalized_index_v2"}
    if final.exists() and receipt.exists():
        saved=json.loads(receipt.read_text())
        if saved["identity"]!=expected or sha256_file(final)!=saved["sha256"]:
            raise RuntimeError("Normalized index cache identity/checksum mismatch")
        return final,saved
    if _audit_resources()["available_bytes"]<cap+1024**3:
        raise RuntimeError("Insufficient available RAM for bounded index decoder plus 1GiB reserve")
    token=str(time.time_ns());temporary=folder/(key+"."+token+".tmp.sqlite")
    measurement=folder/(key+"."+token+".measure.json")
    request=dict(source=str(source),mapping=str(mapping) if mapping else None,
                 temporary=str(temporary),measurement=str(measurement),memory_limit_bytes=cap)
    log=folder/(key+"."+token+".decode.log")
    started=time.monotonic()
    with log.open("x") as handle:
        result=subprocess.run([sys.executable,"-c",_INDEX_DECODE_PROGRAM,json.dumps(request)],
            stdout=handle,stderr=subprocess.STDOUT,env=dict(os.environ,OPENBLAS_NUM_THREADS="1",
            OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",PYTHONDONTWRITEBYTECODE="1"),
            timeout=float(config.get("audit_decode_timeout_seconds",300)))
    if result.returncode!=0:
        blocked=dict(status="blocked",identity=expected,returncode=result.returncode,log=str(log),
            memory_limit_bytes=cap,reason="Native pickled index failed bounded decode; no candidates skipped")
        atomic_json(folder/(key+"."+token+".blocked.json"),blocked)
        raise RuntimeError("Bounded index decode blocked; inspect "+str(log))
    saved=json.loads(measurement.read_text())
    os.replace(temporary,final)
    saved.update(identity=expected,sha256=sha256_file(final),parent_wall_seconds=time.monotonic()-started,log=str(log))
    atomic_json(receipt,saved)
    return final,saved


def audit_data(config, run_dir):
    """Bounded, resumable audit; preserve committed index identities.

    File/point evidence is SQLite-backed, Python LRUs are bounded, and each
    candidate batch commits its cursor plus gzip ledger in one DB transaction.
    A final audit_chunks row is written only after every ordinal of that index.
    The legacy 106 whole-index commits are verified and reused unchanged.
    """
    import shutil
    root=Path(config.get("repo_root",DEFAULT_ROOT)).resolve()
    data_root=Path(config.get("data_root",str(root/"data"))).absolute()
    if data_root!=(root/"data").absolute():raise ValueError("This run requires existing repo_root/data")
    run=Path(run_dir).resolve();run.mkdir(parents=True,exist_ok=True)
    evidence=Path(config.get("audit_evidence_dir",str(run))).resolve();evidence.mkdir(parents=True,exist_ok=True)
    finaldb=run/"data_audit.sqlite";summary_path=run/"data_audit_summary.json"
    if finaldb.exists() or summary_path.exists():raise FileExistsError("Frozen audit already published")
    trainbase=data_root/"megadepth/train";indexbase=data_root/"megadepth/index"
    indexroot=Path(config.get("train_index_root",str(indexbase/"scene_info_0.1_0.7")))
    trainlist=Path(config.get("train_index",str(indexbase/"trainvaltest_list/train_list.txt")))
    stems=[x.strip().removesuffix(".npz") for x in trainlist.read_text().splitlines() if x.strip()]
    excluded,exclusion_evidence=test_scene_exclusions(root)
    if (run/"test_scene_exclusions.json").exists():
        old=json.loads((run/"test_scene_exclusions.json").read_text())
        if set(old["scenes"])!=excluded:raise ValueError("Excluded test scenes changed")
    else:atomic_json(run/"test_scene_exclusions.json",{"scenes":sorted(excluded),"sources":exclusion_evidence})
    if not (run/"data_paths.json").exists():
        # Stream complete symlink evidence; only bounded preview is retained.
        links_path=run/"data_links.jsonl";preview=[];link_count=0
        with links_path.open("x") as f:
            for base,dirs,files in os.walk(data_root,followlinks=False):
                for name in dirs+files:
                    p=Path(base)/name
                    if p.is_symlink():
                        entry=dict(logical=str(p),target=os.readlink(p),resolved=str(p.resolve()),exists=p.exists())
                        f.write(canonical_json(entry)+"\n");link_count+=1
                        if len(preview)<128:preview.append(entry)
        atomic_json(run/"data_paths.json",dict(repo_root=str(root),data_root=str(data_root),data_root_resolved=str(data_root.resolve()),
            train_data_root=str(trainbase),train_index=str(trainlist),train_index_root=str(indexroot),
            legacy_val_index=str(indexbase/"trainvaltest_list/val_list.txt"),legacy_val_rejected="MegaDepth1500 test scenes",
            mapping="same stem with exact existing no_sfm depth/K/pose; encoded extension may differ",
            symlinks=preview,symlink_count=link_count,symlink_complete_jsonl=str(links_path),
            other_data_not_included=[p.name for p in data_root.iterdir() if p.name not in ("megadepth","megadepth_indices")]))
    if shutil.disk_usage(run).free<4*2**30:raise RuntimeError("Insufficient ext4 audit disk reserve")
    dbpath=run/"data_audit.incomplete.sqlite"
    if dbpath.exists() and not config.get("resume_audit",False):raise FileExistsError("Explicit resume_audit required")
    db=sqlite3.connect(dbpath)
    db.executescript("""
      PRAGMA journal_mode=WAL;
      PRAGMA synchronous=FULL;
      PRAGMA temp_store=FILE;
      PRAGMA cache_size=-16384;
      PRAGMA mmap_size=0;
      PRAGMA threads=1;
      CREATE TABLE IF NOT EXISTS audit_chunks(index_id TEXT PRIMARY KEY,path TEXT,sha256 TEXT,counts TEXT);
      CREATE TABLE IF NOT EXISTS audit_identity(key TEXT PRIMARY KEY,value TEXT);
      CREATE TABLE IF NOT EXISTS files(path TEXT PRIMARY KEY,resolved TEXT,sha256 TEXT,size INTEGER,status TEXT,reason TEXT,shape TEXT,detail TEXT);
      CREATE TABLE IF NOT EXISTS frames(id TEXT PRIMARY KEY,scene TEXT,image TEXT,depth TEXT,K TEXT,pose TEXT,status TEXT,reason TEXT);
      CREATE TABLE IF NOT EXISTS indices(id TEXT PRIMARY KEY,path TEXT,sha256 TEXT,mapping_path TEXT,mapping_sha256 TEXT,candidate_count INTEGER,status TEXT);
      CREATE TABLE IF NOT EXISTS pairs(id INTEGER PRIMARY KEY,pair_id TEXT UNIQUE,scene TEXT,frame0 TEXT,frame1 TEXT,index_id TEXT,ordinal INTEGER,overlap REAL,split TEXT);
      CREATE INDEX IF NOT EXISTS pairs_scene ON pairs(scene);
      CREATE TABLE IF NOT EXISTS manifests(split TEXT,position INTEGER,pair_row INTEGER,PRIMARY KEY(split,position));
      CREATE TABLE IF NOT EXISTS audit_work(index_id TEXT PRIMARY KEY,cursor INTEGER,counts TEXT,decoded_path TEXT,decoded_sha256 TEXT);
      CREATE TABLE IF NOT EXISTS audit_batches(index_id TEXT,start INTEGER,stop INTEGER,path TEXT,sha256 TEXT,PRIMARY KEY(index_id,start));
      CREATE TABLE IF NOT EXISTS audit_frame_map(index_id TEXT,frame_index INTEGER,frame_id TEXT,reason TEXT,PRIMARY KEY(index_id,frame_index));
    """)
    source_identity=canonical_json({"train_list_sha256":sha256_file(trainlist),"stems":stems,"excluded_scenes":sorted(excluded),
                                  "schema":SCHEMA,"mapping":"same_stem_exact_depth_camera_v1"})
    old=db.execute("SELECT value FROM audit_identity WHERE key='source'").fetchone()
    if old and old[0]!=source_identity:raise ValueError("Audit resume source identity differs")
    if not old and db.execute("SELECT COUNT(*) FROM indices").fetchone()[0]:
        raise RuntimeError("Legacy pre-resumable DB requires explicit preserved migration; do not rebuild silently")
    db.execute("INSERT OR IGNORE INTO audit_identity VALUES('source',?)",(source_identity,));db.commit()
    counts=collections.Counter()
    for status,n in db.execute("SELECT status,COUNT(*) FROM files GROUP BY status"):counts["file_"+status]=n
    for row in db.execute("SELECT index_id,path,sha256,counts FROM audit_chunks"):
        if sha256_file(row[1])!=row[2]:raise RuntimeError("Committed candidate chunk checksum mismatch")
        for key,value in json.loads(row[3]).items():
            if not key.startswith("file_"):counts[key]+=value
    for row in db.execute("SELECT counts FROM audit_work"):
        for key,value in json.loads(row[0]).items():
            if not key.startswith("file_"):counts[key]+=value
    counts["missing_indices"]=db.execute("SELECT COUNT(*) FROM indices WHERE status='missing_index'").fetchone()[0]
    completed={r[0] for r in db.execute("SELECT index_id FROM audit_chunks")}
    for indexid,path,digest,mapping_path,mapping_digest in db.execute("SELECT id,path,sha256,mapping_path,mapping_sha256 FROM indices WHERE status='audited'"):
        if indexid in completed and (sha256_file(path)!=digest or (mapping_path and sha256_file(mapping_path)!=mapping_digest)):
            raise RuntimeError("Committed source index identity changed: "+indexid)
    completed.update(r[0] for r in db.execute("SELECT id FROM indices WHERE status='missing_index'"))
    batch_size=int(config.get("audit_pair_batch_size",2048))
    if not 1<=batch_size<=4096:raise ValueError("Audit pair batch must be 1..4096")
    max_new=config.get("max_new_indices")
    if max_new is not None and int(max_new)<1:raise ValueError("max_new_indices must be positive")
    started=time.time();starting_resources=_audit_resources();starting_boot=starting_resources["boot_id"]
    newly_completed=0
    chunkdir=run/"candidate_audit_chunks";chunkdir.mkdir(exist_ok=True)
    bounds=dict(pair_batch_size=batch_size,file_cache_max_entries=256,frame_cache_max_entries=256,
        sqlite_cache_bytes=16*1024**2,sqlite_temp_store="FILE",decoder_address_space_limit_bytes=int(config.get("audit_decode_memory_bytes",1024**3)),
        hdf5_row_batch=64,hdf5_max_width=16384,image_max_pixels=64*1024**2,decoder_processes=1,whole_native_pickle_decode_within_cap=True,raw_pickle_whole_dictionary_decode_within_cap=True,
        no_global_pair_list=True,global_file_dictionary=False)
    atomic_json(evidence/"audit_memory_bounds.json",bounds)
    resource_log=evidence/("audit_resources_"+str(time.time_ns())+".jsonl")
    filecache=collections.OrderedDict();framecache=collections.OrderedDict()
    def put(cache,key,value):
        cache[key]=value;cache.move_to_end(key)
        while len(cache)>256:cache.popitem(last=False)
    def progress(index_no,index_id,status="running",cursor=None):
        resources=_audit_resources()
        if resources["boot_id"]!=starting_boot:raise RuntimeError("WSL boot identity changed during audit")
        state=dict(status=status,index_position=index_no,index_count=len(stems),index=index_id,cursor=cursor,counts=dict(counts),
            elapsed_s=time.time()-started,unique_files=sum(v for k,v in counts.items() if k.startswith("file_")),
            committed_indices=db.execute("SELECT COUNT(*) FROM audit_chunks").fetchone()[0],
            pid=os.getpid(),resources=resources,database_bytes=dbpath.stat().st_size,wal_bytes=Path(str(dbpath)+"-wal").stat().st_size if Path(str(dbpath)+"-wal").exists() else 0,cache_entries=len(filecache),frame_cache_entries=len(framecache),newly_completed_indices=newly_completed)
        atomic_json(evidence/"data_audit_progress.json",state)
        with resource_log.open("a") as handle:handle.write(canonical_json(state)+"\n");handle.flush()
        print(canonical_json(state),flush=True)
        if resources["available_bytes"]<1024**3:raise RuntimeError("Audit stopped below 1GiB available RAM")
        if shutil.disk_usage(run).free<2*2**30:raise RuntimeError("Audit stopped below 2GiB ext4 disk reserve")
        return state
    def auditfile(path,kind):
        key=str(path)
        if key in filecache:filecache.move_to_end(key);return filecache[key]
        row=db.execute("SELECT path,resolved,sha256,size,status,reason,shape,detail FROM files WHERE path=?",(key,)).fetchone()
        if row:
            value=dict(zip(("path","resolved","sha256","size","status","reason","shape","detail"),row))
            value["shape"]=json.loads(value["shape"]);value["detail"]=json.loads(value["detail"]);put(filecache,key,value);return value
        value=dict(path=key,resolved=str(path.resolve()),sha256=None,size=0,status="excluded",reason="missing_file",shape=None,detail={})
        if path.is_file():
            value["size"]=path.stat().st_size
            try:
                value["sha256"]=sha256_file(path)
                if kind=="image":
                    from PIL import Image,UnidentifiedImageError
                    try:
                        with Image.open(path) as header:width,height=header.size
                    except UnidentifiedImageError as error:raise ValueError("image_decode_failed") from error
                    if width*height>64*1024**2:raise MemoryError("Image exceeds bounded 64Mpixel decode budget")
                    arr=cv2.imread(key,cv2.IMREAD_GRAYSCALE)
                    if arr is None or arr.ndim!=2 or min(arr.shape)<1:raise ValueError("image_decode_failed")
                    value["shape"]=list(arr.shape);del arr
                else:
                    with h5py.File(path,"r") as h:
                        if "depth" not in h:raise ValueError("missing_depth_dataset")
                        depth=h["depth"]
                        if len(depth.shape)!=2 or min(depth.shape)<1:raise ValueError("depth_shape_invalid")
                        if depth.shape[1]>16384:raise MemoryError("Depth width exceeds bounded row-read budget")
                        value["shape"]=list(depth.shape);positive=finite=0
                        for row0 in range(0,depth.shape[0],64):
                            arr=depth[row0:row0+64,:];mask=np.isfinite(arr)
                            finite+=int(mask.sum());positive+=int(((arr>0)&mask).sum())
                        value["detail"]=dict(finite_count=finite,positive_count=positive,pixel_count=int(np.prod(depth.shape)))
                        if positive==0:raise ValueError("no_valid_depth")
                value["status"]="valid";value["reason"]=None
            except (OSError,MemoryError):
                # Storage/resource failures are not data exclusions.
                raise
            except Exception as error:value["reason"]=type(error).__name__+":"+str(error)[:250]
        db.execute("INSERT INTO files VALUES(?,?,?,?,?,?,?,?)",(key,value["resolved"],value["sha256"],value["size"],value["status"],value["reason"],canonical_json(value["shape"]),canonical_json(value["detail"])))
        counts["file_"+value["status"]]+=1;put(filecache,key,value);return value
    try:
        progress(len(completed),"resume_identity_verified")
        priority=config.get("audit_priority_index")
        if priority is not None and priority not in stems:raise ValueError("Unknown priority index")
        audit_order=([priority]+[s for s in stems if s!=priority]) if priority else stems
        for index_no,stem in enumerate(audit_order):
            if stem in completed:continue
            if max_new is not None and newly_completed>=int(max_new):break
            source=indexroot/(stem+".npz");scene=_scene(stem)
            if not source.is_file():
                db.execute("INSERT OR REPLACE INTO indices VALUES(?,?,?,?,?,?,?)",(stem,str(source),None,None,None,0,"missing_index"))
                counts["missing_indices"]+=1;db.commit();completed.add(stem);newly_completed+=1
                progress(index_no+1,stem);continue
            source_sha=sha256_file(source);mapping=indexbase/"scene_info_0.1_0.7_no_sfm"/source.name
            mapping=mapping if mapping.is_file() else None;mapping_sha=sha256_file(mapping) if mapping else None
            decoded,decoder_measure=_decode_index_bounded(source,mapping,run,config,source_sha,mapping_sha)
            indexdb=sqlite3.connect("file:"+str(decoded)+"?mode=ro",uri=True)
            indexdb.execute("PRAGMA cache_size=-4096");indexdb.execute("PRAGMA mmap_size=0")
            candidate_count=decoder_measure["pair_count"]
            work=db.execute("SELECT cursor,counts,decoded_path,decoded_sha256 FROM audit_work WHERE index_id=?",(stem,)).fetchone()
            if work:
                cursor,chunk_counts=work[0],collections.Counter(json.loads(work[1]))
                if work[3]!=decoder_measure["sha256"]:raise RuntimeError("Partial index decoder identity changed")
                row=db.execute("SELECT sha256,mapping_sha256 FROM indices WHERE id=?",(stem,)).fetchone()
                if row!=(source_sha,mapping_sha):raise RuntimeError("Partial index source changed")
                for path,digest in db.execute("SELECT path,sha256 FROM audit_batches WHERE index_id=?",(stem,)):
                    if sha256_file(path)!=digest:raise RuntimeError("Partial audit batch checksum mismatch")
            else:
                cursor=0;chunk_counts=collections.Counter(indices_read=1);counts["indices_read"]+=1
                db.execute("INSERT INTO indices VALUES(?,?,?,?,?,?,?)",(stem,str(source),source_sha,str(mapping) if mapping else None,mapping_sha,candidate_count,"processing"))
                db.execute("INSERT INTO audit_work VALUES(?,?,?,?,?)",(stem,0,canonical_json(dict(chunk_counts)),str(decoded),decoder_measure["sha256"]));db.commit()
            index_chunks=chunkdir/(stem+".batches");index_chunks.mkdir(exist_ok=True)
            framecache.clear()
            def frame(i):
                if i in framecache:framecache.move_to_end(i);return framecache[i]
                row=db.execute("SELECT frame_id,reason FROM audit_frame_map WHERE index_id=? AND frame_index=?",(stem,i)).fetchone()
                if row:put(framecache,i,row);return row
                raw=indexdb.execute("SELECT payload FROM frames WHERE idx=?",(i,)).fetchone()
                if raw is None:raise ValueError("Pair refers to missing index frame")
                raw=json.loads(raw[0]);imgraw,depraw=raw["image_paths"],raw["depth_paths"];reason=None;K=pose=None
                if imgraw is None or depraw is None:
                    reason="missing_index_path";image=trainbase/("__missing_image_"+str(i));depth=trainbase/("__missing_depth_"+str(i))
                else:
                    image=trainbase/str(imgraw);depth=trainbase/str(depraw)
                    if not image.exists() and raw["mapped"] is not None:
                        alt=raw["mapped"]
                        if (alt["image_paths"] is None or str(alt["depth_paths"])!=str(depraw) or
                            Path(str(alt["image_paths"])).stem!=Path(str(imgraw)).stem or
                            not np.array_equal(raw["intrinsics"],alt["intrinsics"]) or not np.array_equal(raw["poses"],alt["poses"])):
                            reason="path_mapping_metadata_mismatch"
                        else:image=trainbase/str(alt["image_paths"])
                    if not image.absolute().is_relative_to(trainbase.absolute()) or not depth.absolute().is_relative_to(trainbase.absolute()):
                        reason="path_outside_logical_data_root"
                try:
                    K=np.asarray(raw["intrinsics"],dtype=np.float64).reshape(3,3);pose=np.asarray(raw["poses"],dtype=np.float64).reshape(4,4)
                    if not np.isfinite(K).all() or not np.isfinite(pose).all():raise ValueError("nonfinite_camera")
                    if abs(np.linalg.det(K))<1e-10 or abs(np.linalg.det(pose))<1e-10 or K[0,0]<=0 or K[1,1]<=0:raise ValueError("invalid_camera")
                except (ValueError,TypeError):reason=reason or "invalid_camera";K=pose=None
                info=dict(scene=scene,image=str(image),depth=str(depth),K=None if K is None else K.tolist(),pose=None if pose is None else pose.tolist())
                fid=hashlib.sha256(canonical_json(info).encode()).hexdigest()
                if reason is None:
                    ir,dr=auditfile(image,"image"),auditfile(depth,"depth")
                    if ir["status"]!="valid":reason="image:"+str(ir["reason"])
                    elif dr["status"]!="valid":reason="depth:"+str(dr["reason"])
                db.execute("INSERT OR IGNORE INTO frames VALUES(?,?,?,?,?,?,?,?)",(fid,scene,str(image),str(depth),canonical_json(info["K"]),canonical_json(info["pose"]),"valid" if reason is None else "excluded",reason))
                db.execute("INSERT INTO audit_frame_map VALUES(?,?,?,?)",(stem,i,fid,reason))
                put(framecache,i,(fid,reason));return fid,reason
            while cursor<candidate_count:
                rows=indexdb.execute("SELECT ordinal,i,j,overlap FROM pairs WHERE ordinal>=? ORDER BY ordinal LIMIT ?",(cursor,batch_size)).fetchall()
                if not rows or rows[0][0]!=cursor:raise RuntimeError("Decoded candidate ordinal gap")
                stop=rows[-1][0]+1;temp=index_chunks/(str(cursor)+"-"+str(stop)+"."+str(time.time_ns())+".tmp.gz")
                final=index_chunks/(str(cursor)+"-"+str(stop)+"."+str(time.time_ns())+".jsonl.gz")
                before=counts.copy()
                with gzip.open(temp,"wt",compresslevel=1) as ledger:
                    for ordinal,i,j,overlap_text in rows:
                        overlap=float(overlap_text);counts["candidate_pairs"]+=1
                        if scene in excluded:reason="test_scene";f0=f1=None
                        else:
                            f0,r0=frame(i);f1,r1=frame(j);reason=r0 or r1
                            if i==j:reason="self_pair"
                            if not math.isfinite(overlap) or overlap<=0:reason="native_overlap_filter"
                        pairid=hashlib.sha256((str(f0)+"|"+str(f1)+"|0to1").encode()).hexdigest()[:32] if f0 else None
                        if reason is None:
                            try:
                                db.execute("INSERT INTO pairs(pair_id,scene,frame0,frame1,index_id,ordinal,overlap) VALUES(?,?,?,?,?,?,?)",(pairid,scene,f0,f1,stem,ordinal,overlap))
                                counts["legal_available_pairs"]+=1
                            except sqlite3.IntegrityError:reason="duplicate_directed_pair"
                        counts["excluded_"+reason if reason else "included"]+=1
                        ledger.write(canonical_json(dict(index_id=stem,ordinal=ordinal,scene=scene,idx0=i,idx1=j,direction="0to1",
                            pair_id=pairid,frame0=f0,frame1=f1,status="included" if reason is None else "excluded",reason=reason))+"\n")
                with temp.open("rb") as f:os.fsync(f.fileno())
                os.replace(temp,final)
                chunk_counts.update({k:v-before.get(k,0) for k,v in counts.items()})
                db.execute("INSERT INTO audit_batches VALUES(?,?,?,?,?)",(stem,cursor,stop,str(final),sha256_file(final)))
                db.execute("UPDATE audit_work SET cursor=?,counts=? WHERE index_id=?",(stop,canonical_json(dict(chunk_counts)),stem))
                db.commit();cursor=stop
                db.execute("PRAGMA wal_checkpoint(PASSIVE)")
                progress(index_no,stem,cursor=cursor)
            indexdb.close()
            chunk=chunkdir/(stem+".jsonl.gz");temp=chunkdir/(stem+"."+str(time.time_ns())+".tmp.gz")
            with temp.open("wb") as f:
                for row in db.execute("SELECT path FROM audit_batches WHERE index_id=? ORDER BY start",(stem,)):
                    with open(row[0],"rb") as part:shutil.copyfileobj(part,f,1024**2)
                f.flush();os.fsync(f.fileno())
            if chunk.exists():os.rename(chunk,chunk.with_name(chunk.name+".uncommitted_"+str(time.time_ns())))
            os.replace(temp,chunk)
            db.execute("INSERT INTO audit_chunks VALUES(?,?,?,?)",(stem,str(chunk),sha256_file(chunk),canonical_json(dict(chunk_counts))))
            db.execute("UPDATE indices SET status='audited' WHERE id=?",(stem,))
            db.execute("DELETE FROM audit_work WHERE index_id=?",(stem,));db.commit()
            completed.add(stem);newly_completed+=1
            progress(index_no+1,stem)
        remaining=[s for s in stems if s not in completed]
        if remaining:
            state=progress(len(completed),remaining[0],status="incomplete")
            state.update(reason="configured bounded index boundary",remaining_indices=len(remaining),full_manifest_published=False,bounds=bounds,expected_indices=len(stems),committed_indices=len(completed),audit_complete=False)
            atomic_json(evidence/"bounded_audit_result.json",state);db.close();return state
        return _publish_audit_bounded(db,dbpath,finaldb,run,stems,excluded,counts,started,trainlist,batch_size)
    except BaseException as error:
        db.rollback()
        durable=collections.Counter()
        for status,n in db.execute("SELECT status,COUNT(*) FROM files GROUP BY status"):durable["file_"+status]=n
        for table in ("audit_chunks","audit_work"):
            for row in db.execute("SELECT counts FROM "+table):
                for key,value in json.loads(row[0]).items():
                    if not key.startswith("file_"):durable[key]+=value
        durable["missing_indices"]=db.execute("SELECT COUNT(*) FROM indices WHERE status='missing_index'").fetchone()[0]
        state=dict(status="blocked" if isinstance(error,(MemoryError,OSError)) else "failed",type=type(error).__name__,message=str(error),
            resources=_audit_resources(),counts=dict(durable),uncommitted_observations=dict(counts),bounds=bounds,full_manifest_published=False)
        atomic_json(evidence/"bounded_audit_failure.json",state)
        db.close();raise


def _publish_audit_bounded(db,dbpath,finaldb,run,stems,excluded,counts,started,trainlist,batch_size):
    import shutil
    # Sorting is backed by a SQLite file index; no global Python pair array.
    db.execute("CREATE INDEX IF NOT EXISTS pairs_split_scene_identity ON pairs(split,scene,pair_id)")
    db.commit()
    scenes=[r[0] for r in db.execute("SELECT DISTINCT scene FROM pairs ORDER BY scene")]
    ordered=scenes.copy();random.Random(2026).shuffle(ordered)
    nval=max(1,min(len(ordered)-1,int(round(len(ordered)*.1)))) if len(scenes)>=2 else 0
    vals=sorted(ordered[:nval]);split_status="committed" if nval else "blocked"
    for scene in scenes:
        last=0;split="val" if scene in vals else "train"
        while True:
            ids=[r[0] for r in db.execute("SELECT id FROM pairs WHERE scene=? AND id>? ORDER BY id LIMIT ?",(scene,last,batch_size))]
            if not ids:break
            db.executemany("UPDATE pairs SET split=? WHERE id=?",((split,rid) for rid in ids))
            db.commit();last=ids[-1]
    # Export is restartable before final publication. Remove old partial positions
    # in bounded transactions, then stream rows through bounded commit groups.
    while True:
        rows=db.execute("SELECT rowid FROM manifests LIMIT ?",(batch_size,)).fetchall()
        if not rows:break
        db.executemany("DELETE FROM manifests WHERE rowid=?",rows);db.commit()
    manifest_hashes={};split_counts={}
    for split in ("train","val"):
        dst=run/(split+"_manifest.jsonl");tmp=dst.with_name(dst.name+".tmp")
        n=0
        with tmp.open("w") as f:
            query=db.execute("SELECT id,pair_id,scene,frame0,frame1,index_id,ordinal FROM pairs WHERE split=? ORDER BY scene,pair_id",(split,))
            while True:
                rows=query.fetchmany(batch_size)
                if not rows:break
                for rid,pid,scene,f0,f1,indexid,ordinal in rows:
                    f.write(canonical_json(dict(pair_id=pid,scene=scene,frame0=f0,frame1=f1,index_id=indexid,ordinal=ordinal,direction="0to1"))+"\n")
                    db.execute("INSERT INTO manifests VALUES(?,?,?)",(split,n,rid));n+=1
                db.commit()
            f.flush();os.fsync(f.fileno())
        # Final filenames are exposed only after all indices are complete; a
        # manifest without the final summary remains explicitly uncommitted.
        os.replace(tmp,dst);manifest_hashes[split]=sha256_file(dst);split_counts[split]=n
    audit_tmp=run/"candidate_audit.jsonl.gz.tmp"
    with audit_tmp.open("wb") as f:
        for stem in stems:
            entry=db.execute("SELECT path FROM audit_chunks WHERE index_id=?",(stem,)).fetchone()
            if entry:
                with open(entry[0],"rb") as part:shutil.copyfileobj(part,f,1024**2)
        f.flush();os.fsync(f.fileno())
    file_bytes=db.execute("SELECT COALESCE(SUM(size),0) FROM files").fetchone()[0]
    unique_images=db.execute("SELECT COUNT(DISTINCT image) FROM frames WHERE status='valid'").fetchone()[0]
    complete=db.execute("SELECT COUNT(*) FROM indices WHERE status IN ('audited','missing_index')").fetchone()[0]
    if complete!=len(stems) or db.execute("SELECT COUNT(*) FROM audit_work").fetchone()[0]:
        raise RuntimeError("Refusing publication with incomplete source indices")
    db.commit()
    checkpoint=db.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
    if tuple(checkpoint)!=(0,0,0):raise RuntimeError("Refusing publication with uncheckpointed WAL: "+str(checkpoint))
    if str(db.execute("PRAGMA journal_mode=DELETE").fetchone()[0]).lower()!="delete":
        raise RuntimeError("Refusing publication in WAL mode")
    db.close()
    if any(Path(str(dbpath)+suffix).exists() for suffix in ("-wal","-shm")):
        raise RuntimeError("Refusing publication while SQLite sidecar remains")
    os.replace(dbpath,finaldb);os.replace(audit_tmp,run/"candidate_audit.jsonl.gz")
    summary=dict(schema=SCHEMA,status=split_status,dataset="MegaDepth",scope="full_available",
        official_complete_dataset_claim=False,seed=2026,counts=dict(counts),
        expected_indices=len(stems),committed_indices=complete,audit_complete=True,
        expected_index_count=len(stems),completed_index_count=complete,
        train_count=split_counts["train"],val_count=split_counts["val"],
        train_scenes=[s for s in scenes if s not in vals],val_scenes=vals,
        excluded_test_scenes=sorted(excluded),unique_images=unique_images,file_bytes=file_bytes,
        elapsed_s=time.time()-started,manifest_sha256=manifest_hashes,data_audit_db=str(finaldb),
        data_audit_sha256=sha256_file(finaldb),candidate_audit_sha256=sha256_file(run/"candidate_audit.jsonl.gz"),
        train_list_sha256=sha256_file(trainlist),free_disk_bytes=shutil.disk_usage(run).free,
        split_policy="scene shuffle Python Random(2026); held out round(0.1*scenes), at least one",
        legacy_val_rejected=True,overlap_filter="finite and strictly >0.0; native cfg threshold unchanged")
    atomic_json(run/"data_audit_summary.json",summary)
    atomic_json(run/"feature_label_contract.json",label_contract())
    atomic_json(run/"data_audit_progress.json",dict(status="committed",counts=dict(counts),elapsed_s=time.time()-started,
        expected_indices=len(stems),committed_indices=complete,audit_complete=True))
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
    parser.add_argument("--max-new-indices",type=int)
    parser.add_argument("--audit-evidence-dir")
    args=parser.parse_args()
    config=json.loads(Path(args.config).read_text()) if args.config else {}
    config["resume_audit"]=args.resume_audit
    if args.max_new_indices is not None:config["max_new_indices"]=args.max_new_indices
    if args.audit_evidence_dir:config["audit_evidence_dir"]=args.audit_evidence_dir
    print(json.dumps(audit_data(config,args.run_dir),indent=2))
