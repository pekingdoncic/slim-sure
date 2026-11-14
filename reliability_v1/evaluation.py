"""Full-validation reliability and fixed-budget pilot, with injectable production predictor.

predict_fn(record, mode) receives A0 / A_score / A_texture and returns CPU arrays.
No reliability score is used for matching selection or geometry.
"""
from collections import defaultdict,deque
from contextlib import contextmanager
from pathlib import Path
import fcntl
import json
import os
import subprocess
import time

import numpy as np

from .geometry_worker import (ARRAY_KEYS,array_sha,atomic_json,digest,file_sha,prediction_identity,validate_protocol)
from .metrics import ScoreAccumulator,paired_pose_summary


def _json(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {str(k):_json(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [_json(v) for v in value]
    return value


@contextmanager
def writer_lock(directory):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    with (directory/".writer.lock").open("a") as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:yield
        finally:fcntl.flock(f,fcntl.LOCK_UN)


def _atomic_npz(path,arrays):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+".tmp."+str(os.getpid()))
    with tmp.open("wb") as f:
        np.savez_compressed(f,**arrays);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)


def _pair_id(record):
    return str(record["pair_id"])


def freeze_pilot(records,limit=100):
    if not 0<limit<=100:raise ValueError("Pilot limit must be in [1,100]")
    if hasattr(records,"scene_round_robin_indices"):
        return [records[i] for i in records.scene_round_robin_indices(limit)]
    scenes=defaultdict(list)
    for r in records:
        values=scenes[str(r["scene"])]
        values.append(r)
        values.sort(key=_pair_id)
        del values[limit:]
    queues={s:deque(sorted(rs,key=_pair_id)) for s,rs in sorted(scenes.items())}
    result=[]
    while len(result)<limit and any(queues.values()):
        for s in queues:
            if queues[s] and len(result)<limit:result.append(queues[s].popleft())
    return result


def _path_id(pair_id):
    # Pair IDs are identity strings; filesystem components need not share their syntax.
    return digest({"pair_id":pair_id})[:24]


def _records_digest(records):
    if hasattr(records,"path"):
        path=Path(records.path)
        return digest({"manifest_path":str(path.resolve()),"sha256":file_sha(path),"count":len(records)})
    import hashlib
    h=hashlib.sha256()
    for record in records:
        h.update(json.dumps(_json(record),sort_keys=True,separators=(",",":"),allow_nan=False).encode())
        h.update(b"\n")
    return h.hexdigest()


def evaluate_reliability(records,predict_fn,out_dir,checkpoint_id,identity=None,retain_predictions=False):
    """Stream every pair/valid parent; retain exact rankings and per-pair summaries.

    SQLite scalar rows support exact pooled ranking with bounded RAM and are removed
    only after atomic committed aggregate/per-pair artifacts, unless retain_predictions.
    No high-resolution feature or full validation manifest is materialized in RAM.
    """
    out=Path(out_dir);count=len(records)
    ident={"checkpoint_id":_json(checkpoint_id),"records_sha256":_records_digest(records),
           "identity":_json(identity),"retain_predictions":bool(retain_predictions),
           "schema":"slim_reliability_validation_v1"}
    with writer_lock(out):
        marker=out/"commit.json"
        if marker.exists():
            m=json.loads(marker.read_text())
            if m["identity"]!=ident:raise ValueError("Validation cache identity differs")
            for key,sha in m["retained_sha256"].items():
                if file_sha(out/key)!=sha:raise ValueError("Validation cache checksum failed: "+key)
            return json.loads((out/"score_metrics.json").read_text())
        identity_path=out/"identity.json"
        if identity_path.exists() and json.loads(identity_path.read_text())!=ident:
            raise ValueError("Incomplete validation has a different identity")
        atomic_json(identity_path,ident)
        atomic_json(out/"state.json",{"status":"running","pair_count":count,"checkpoint_id":checkpoint_id})
        accumulator=ScoreAccumulator(out/"scores.sqlite")
        forwards=0;reused=0;begin=time.perf_counter()
        try:
            for i,record in enumerate(records):
                pid=_pair_id(record)
                existing=accumulator.db.execute("SELECT summary FROM pairs WHERE pair_id=?",(pid,)).fetchone()
                if existing is not None:
                    # Manifest uniqueness is an audit prerequisite; duplicate positions still
                    # fail final count/coverage rather than silently counting as valid coverage.
                    reused+=1
                    if retain_predictions:
                        cache=out/"parents"/(_path_id(pid)+".npz")
                        sidecar=out/"parents"/(_path_id(pid)+".json")
                        item=json.loads(sidecar.read_text())
                        if item["identity"]!=dict(ident,pair_id=pid) or file_sha(cache)!=item["sha256"]:
                            raise ValueError("Parent cache checksum changed")
                else:
                    with (out/"physical_calls.jsonl").open("a") as log:
                        log.write(json.dumps({"pair_id":pid,"purpose":"full_validation","status":"started","time":time.time()})+"\n")
                        log.flush();os.fsync(log.fileno())
                    pred=predict_fn(record,"A_score");forwards+=1
                    if str(pred.get("pair_id",pid))!=pid:raise ValueError("Predictor returned wrong pair")
                    pred["scene"]=record["scene"]
                    if "coarse_match_id" not in pred or len(pred["coarse_match_id"])!=len(pred["labels"]):
                        raise ValueError("Production predictor must provide raw parent-aligned identity")
                    if retain_predictions:
                        fields=("labels","logits","coarse_confidence","norm0","norm1","coarse_match_id",
                                "coarse_source_ids","coarse_target_ids","low_texture_mask")
                        arrays={k:np.asarray(pred[k]) for k in fields if k in pred}
                        arrays.update(pair_id=np.array(pid),direction=np.array(str(record.get("direction","0to1"))))
                        cache=out/"parents"/(_path_id(pid)+".npz")
                        _atomic_npz(cache,arrays)
                        atomic_json(out/"parents"/(_path_id(pid)+".json"),
                            {"identity":dict(ident,pair_id=pid),"sha256":file_sha(cache)})
                    summary=accumulator.add(pid,pred)
                    del pred
                    with (out/"physical_calls.jsonl").open("a") as log:
                        log.write(json.dumps({"pair_id":pid,"purpose":"full_validation","status":"committed","time":time.time()})+"\n")
                atomic_json(out/"state.json",{"status":"running","visited_pairs":i+1,"pair_count":count,
                    "new_forwards":forwards,"reused_pairs":reused,"checkpoint_id":checkpoint_id,"heartbeat_unix":time.time()})
            summary=accumulator.summary()
            attempts=0
            if (out/"physical_calls.jsonl").exists():
                with (out/"physical_calls.jsonl").open() as f:
                    for line in f:
                        if json.loads(line)["status"]=="started":attempts+=1
            summary.update(status="committed",checkpoint_id=checkpoint_id,identity=ident,
                complete_pair_coverage=summary["pair_count"]==count,
                costs={"new_validation_forwards":forwards,"reused_committed_pairs":reused,
                       "total_physical_forward_attempts":attempts,"replayed_or_uncommitted_attempts":max(0,attempts-count),
                       "wall_seconds_this_invocation":time.perf_counter()-begin})
            if not summary["complete_pair_coverage"]:raise ValueError("Incomplete validation or duplicate pair identity")
            per_pair=out/"per_pair.jsonl";tmp=out/"per_pair.jsonl.tmp"
            with tmp.open("w") as f:
                for row, in accumulator.db.execute("SELECT summary FROM pairs ORDER BY pair_id"):
                    f.write(row+"\n")
                f.flush();os.fsync(f.fileno())
            os.replace(tmp,per_pair)
            accumulator.close()
            db_path=out/"scores.sqlite";db_sha=file_sha(db_path)
            summary["storage"]={"temporary_exact_scalar_database_bytes":db_path.stat().st_size,
                "temporary_database_sha256":db_sha,"scalar_database_retained":bool(retain_predictions),
                "parent_predictions_retained":bool(retain_predictions),
                "note":"Temporary scalar SQL sort space is needed for exact ranks; no feature tensors stored."}
            atomic_json(out/"score_metrics.json",_json(summary))
            retained={k:file_sha(out/k) for k in ("score_metrics.json","per_pair.jsonl","physical_calls.jsonl")}
            if retain_predictions:retained["scores.sqlite"]=db_sha
            atomic_json(marker,{"status":"committed","identity":ident,"retained_sha256":retained,
                "temporary_score_database_sha256":db_sha})
            if not retain_predictions:db_path.unlink()
            atomic_json(out/"state.json",{"status":"committed","pair_count":count,"checkpoint_id":checkpoint_id})
            return summary
        except BaseException as error:
            try:accumulator.close()
            except Exception:pass
            atomic_json(out/"state.json",{"status":"failed","error":repr(error),"new_forwards":forwards,
                "reused_pairs":reused,"heartbeat_unix":time.time()})
            raise

def canonical_protocol(hybrid_root="/home/bupt/projects/hybrid_match"):
    root=Path(hybrid_root)
    contract=root/"outputs/m2b1_cpu_20261112T133600/solver_contract.json"
    sources=[root/"evaluation/pose.py",root/"evaluation/metrics.py",contract]
    if not all(p.is_file() for p in sources):raise FileNotFoundError("Canonical geometry source or contract missing")
    old=json.loads(contract.read_text())
    for name in ("pose.py","metrics.py"):
        if old["source_sha256"][name]!=file_sha(root/"evaluation"/name):
            raise ValueError("Canonical solver source changed since recorded contract")
    protocol={"schema":"slim_reliability_R832_development_geometry_v1","hybrid_root":str(root),
        "geometry":{"backend":"LO-RANSAC","pixel_threshold":0.5,"confidence":0.99999},
        "seeds":list(range(5)),"source_sha256":{str(p):file_sha(p) for p in sources},
        "coordinates":"original_image_pixels; original K; T1 @ inverse(T0)",
        "point_order":"native_order_then_RandomState(seed).permutation","resolution":"R832",
        "scope":"independent development validation; never public test"}
    validate_protocol(protocol)
    return protocol


def coverage_metrics(pred):
    required=("source_grid_valid","source_grid_valid_gt","source_grid_low_texture",
              "coarse_source_ids","labels","final_source_input","source_grid_hw","coarse_stride_xy")
    missing=[k for k in required if k not in pred]
    if missing:return {"status":"blocked","missing":missing}
    valid=np.asarray(pred["source_grid_valid"],bool).reshape(-1)
    gt=np.asarray(pred["source_grid_valid_gt"],bool).reshape(-1)&valid
    low=np.asarray(pred["source_grid_low_texture"],bool).reshape(-1)&valid
    if len(gt)!=len(valid) or len(low)!=len(valid):raise ValueError("Coverage grid masks differ")
    source=np.asarray(pred["coarse_source_ids"],np.int64).reshape(-1)
    labels=np.asarray(pred["labels"],np.int8).reshape(-1)
    if len(source)!=len(labels) or (source<0).any() or (source>=len(valid)).any():
        raise ValueError("Invalid coarse source grid identity")
    correct=np.zeros(len(valid),bool);correct[source[labels==1]]=True
    all_matches=np.zeros(len(valid),bool);all_matches[source]=True
    gh,gw=map(int,pred["source_grid_hw"])
    if gh*gw!=len(valid):raise ValueError("Grid shape differs")
    fine=np.asarray(pred["final_source_input"],float).reshape(-1,2)
    stride=np.asarray(pred["coarse_stride_xy"],float)
    cells=np.floor(fine/stride).astype(np.int64)
    inside=(cells[:,0]>=0)&(cells[:,0]<gw)&(cells[:,1]>=0)&(cells[:,1]<gh)
    final=np.zeros(len(valid),bool);final[cells[inside,1]*gw+cells[inside,0]]=True
    def part(region):
        total=int(region.sum());known=int((region & gt).sum());hit=int((region&correct&gt).sum())
        return {"valid_grid_cells":total,"GT_visible_cells":known,"GT_coverage":known/total if total else None,
            "correct_coarse_cells":hit,"correct_coverage_all_valid_cells":hit/total if total else None,
            "effective_recall_proxy_GT_visible":hit/known if known else None,
            "matched_coarse_cells":int((all_matches&region).sum()),
            "final_covered_cells":int((final&region).sum()),
            "final_coverage":int((final&region).sum())/total if total else None}
    result={"status":"committed","all_region":part(valid),"local_low_texture":part(low),
        "coarse_correct_fraction":float(np.mean(labels[labels>=0])) if np.any(labels>=0) else None,
        "coarse_valid_count":int(np.sum(labels>=0)),"coarse_ignore_count":int(np.sum(labels<0)),
        "final_match_count":len(fine),"coverage_note":"Denominators are fixed valid source-grid cells, with GT visibility reported separately."}
    if all(k in pred for k in ("final_target_input","final_gt_target_input","final_gt_valid")):
        mask=np.asarray(pred["final_gt_valid"],bool)
        target=np.asarray(pred["final_target_input"],float);truth=np.asarray(pred["final_gt_target_input"],float)
        if target.shape!=truth.shape or len(mask)!=len(target):raise ValueError("Final localization arrays differ")
        errors=np.linalg.norm(target[mask]-truth[mask],axis=-1)
        result["localization_input_pixels"]={"valid_count":len(errors),"ignore_count":len(mask)-len(errors),
            "mean":float(errors.mean()) if len(errors) else None,
            "median":float(np.median(errors)) if len(errors) else None,
            "p90":float(np.quantile(errors,.9)) if len(errors) else None}
    else:result["localization_input_pixels"]={"status":"blocked","reason":"missing final-point visibility/projection"}
    return result


def run_pilot(records,predict_fn,out_dir,protocol=None,identity=None,
              geometry_python="/home/bupt/miniconda3/envs/hybrid-eval/bin/python",
              geometry_watchdog_seconds=None):
    """Freeze <=100 scene-round-robin pairs, export native/score/texture, then CPU geometry."""
    if geometry_watchdog_seconds is None or geometry_watchdog_seconds<=0:
        raise ValueError("Geometry watchdog must be provided from measured/approved cost evidence")
    out=Path(out_dir);pilot=freeze_pilot(records);protocol=protocol or canonical_protocol()
    if not pilot:raise ValueError("No independent validation pairs for pilot")
    ident={"records_sha256":digest(_json(pilot)),"protocol_sha256":digest(protocol),
           "identity":_json(identity),"schema":"slim_reliability_pilot_v1"}
    with writer_lock(out):
        identity_path=out/"identity.json"
        if identity_path.exists() and json.loads(identity_path.read_text())!=ident:
            raise ValueError("Pilot identity changed")
        atomic_json(identity_path,ident);atomic_json(out/"pilot_manifest.json",_json(pilot))
        atomic_json(out/"protocol.json",protocol)
        atomic_json(out/"state.json",{"status":"running","phase":"predictions","pair_count":len(pilot)})
        calls=0;reused_predictions=0;summary_rows=[]
        # Complete all GPU predictions before starting serial CPU solves.
        for i,record in enumerate(pilot):
            pid=_pair_id(record);predictions={}
            for mode in ("A0","A_score","A_texture"):
                dest=out/"predictions"/mode/_path_id(pid);commit=dest/"commit.json"
                if commit.exists():
                    m=json.loads(commit.read_text())
                    if m["identity"]!=dict(ident,pair_id=pid,mode=mode) or file_sha(dest/"pred.npz")!=m["sha256"]:
                        raise ValueError("Pilot prediction identity or checksum changed")
                    with np.load(dest/"pred.npz",allow_pickle=False) as f:
                        pred={k:f[k] for k in f.files}
                    metrics=m["coverage"];reused_predictions+=1
                else:
                    pred=predict_fn(record,mode);calls+=1
                    for key in ARRAY_KEYS:
                        if key not in pred:raise ValueError("Pilot predictor lacks "+key)
                        if not np.isfinite(pred[key]).all():raise FloatingPointError("Nonfinite geometry input")
                    if "candidate_counts" not in pred:raise ValueError("Pilot must record both actual candidate counts")
                    metrics=coverage_metrics(pred)
                    fields=ARRAY_KEYS+("candidate_counts","coarse_match_id","coarse_source_ids","coarse_target_ids",
                        "labels","source_grid_valid","source_grid_valid_gt","source_grid_low_texture",
                        "source_grid_hw","coarse_stride_xy","final_source_input","final_target_input",
                        "final_gt_target_input","final_gt_valid","coarse_confidence","sparse_correlation_shape")
                    arrays={k:np.asarray(pred[k]) for k in fields if k in pred}
                    arrays["pair_id"]=np.array(pid)
                    _atomic_npz(dest/"pred.npz",arrays)
                    atomic_json(commit,{"status":"committed","identity":dict(ident,pair_id=pid,mode=mode),
                        "sha256":file_sha(dest/"pred.npz"),"coverage":_json(metrics),
                        "timing":_json(pred.get("timing",{})),"memory":_json(pred.get("memory",{}))})
                comparison_keys=ARRAY_KEYS+("candidate_counts","coarse_match_id","coarse_source_ids","coarse_target_ids","coarse_confidence")
                predictions[mode]={k:pred[k] for k in comparison_keys if k in pred}
                summary_rows.append({"pair_id":pid,"scene":record["scene"],"mode":mode,
                    "candidate_counts":np.asarray(pred["candidate_counts"]).tolist(),"coverage":metrics})
                del pred
            a=predictions["A0"]
            if any(k not in predictions["A_score"] or array_sha(a[k])!=array_sha(predictions["A_score"][k]) for k in a):
                raise AssertionError("score_only changed ordered geometry inputs")
            if not np.array_equal(a["candidate_counts"],predictions["A_texture"]["candidate_counts"]):
                raise AssertionError("texture_budget changed the per-side candidate budget")
            atomic_json(out/"state.json",{"status":"running","phase":"predictions","visited_pairs":i+1,
                "pair_count":len(pilot),"new_prediction_forwards":calls,"heartbeat_unix":time.time()})
        atomic_json(out/"pilot_metrics.json",_json(summary_rows))
        env=dict(os.environ,CUDA_VISIBLE_DEVICES="",OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",
                 MKL_NUM_THREADS="1",NUMEXPR_NUM_THREADS="1",PYTHONNOUSERSITE="1")
        physical_workers=0
        errors={mode:[] for mode in ("A0","A_texture")}
        for record in pilot:
            pid=_pair_id(record);name=_path_id(pid)
            for mode in ("A0","A_texture"):
                output=out/"geometry"/mode/name;prediction=out/"predictions"/mode/name/"pred.npz"
                output.mkdir(parents=True,exist_ok=True)
                complete=all((output/("seed_%d.json"%j)).exists() and (output/("seed_%d_commit.json"%j)).exists() for j in range(5))
                if not complete:
                    command=[str(geometry_python),str(Path(__file__).with_name("geometry_worker.py")),
                        "--prediction",str(prediction),"--protocol",str(out/"protocol.json"),"--output",str(output)]
                    atomic_json(out/"state.json",{"status":"running","phase":"geometry","pair_id":pid,"mode":mode,
                        "command":command,"heartbeat_unix":time.time()})
                    with (output/"worker.log").open("a") as log:
                        try:
                            subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,
                                           timeout=geometry_watchdog_seconds,check=True)
                            physical_workers+=1
                        except BaseException as error:
                            atomic_json(out/"state.json",{"status":"failed","phase":"geometry","pair_id":pid,
                                "mode":mode,"error":repr(error),"note":"Uncommitted geometry is not an Inf failure."})
                            raise
                with np.load(prediction,allow_pickle=False) as f:
                    expected_identity=prediction_identity({k:f[k] for k in ARRAY_KEYS},protocol)
                values=[]
                for j in range(5):
                    result_path=output/("seed_%d.json"%j)
                    checksum=json.loads((output/("seed_%d_commit.json"%j)).read_text())
                    if checksum["sha256"]!=file_sha(result_path):raise ValueError("Geometry result checksum failed")
                    row=json.loads(result_path.read_text())
                    if row["status"]!="committed" or row["identity"]!=dict(expected_identity,pair_id=pid,seed=j):
                        raise ValueError("Invalid or stale geometry result")
                    if row["solver_status"]=="success":values.append(float(row["pose_error_deg"]))
                    elif row["solver_status"]=="geometry_failure":values.append(float("inf"))
                    else:raise ValueError("Unexecuted geometry cannot enter pose denominator")
                errors[mode].append(values)
            for seed in range(5):
                ref=out/"geometry"/"A0"/name/("seed_%d.json"%seed)
                atomic_json(out/"geometry"/"A_score"/name/("seed_%d.json"%seed),
                    {"status":"committed","geometry_reused":True,"pair_id":pid,"seed":seed,
                     "reuse_reason":"exact_ordered_points_camera_pose_and_protocol_hash",
                     "reference":str(ref),"reference_sha256":file_sha(ref),"protocol_sha256":digest(protocol)})
        pose=paired_pose_summary(errors["A0"],errors["A_texture"])
        # Counts use committed result evidence, including any resumed calls.
        backend_calls=0
        for mode in ("A0","A_texture"):
            for record in pilot:
                for seed in range(5):
                    row=json.loads((out/"geometry"/mode/_path_id(_pair_id(record))/("seed_%d.json"%seed)).read_text())
                    backend_calls+=int(row["backend_invoked"])
        blocked=[{"pair_id":r["pair_id"],"mode":r["mode"]} for r in summary_rows
            if r["coverage"]["status"]!="committed" or r["coverage"].get("localization_input_pixels",{}).get("status")=="blocked"]
        result={"status":"blocked" if blocked else "committed","identity":ident,"pilot_pairs":len(pilot),
            "geometry_logical_records":2*len(pilot)*5,"score_geometry_reused_records":len(pilot)*5,
            "backend_invocations":backend_calls,"new_geometry_worker_processes":physical_workers,
            "new_prediction_forwards":calls,"reused_predictions":reused_predictions,
            "blocked_coverage_records":blocked,"pose":pose,
            "note":"Functional pilot only; full reliability validation is a separate manifest traversal."}
        atomic_json(out/"pilot_summary.json",result);atomic_json(out/"state.json",result)
        return result


def measure_costs(records,predict_fn,out_dir,synchronize):
    """Same <=10 validation pairs, each path: 3 warmups + 10 synchronized forwards."""
    chosen=freeze_pilot(records,10)
    if not chosen:raise ValueError("No timing pairs")
    if hasattr(predict_fn,"measure_costs"):
        return predict_fn.measure_costs(chosen,out_dir)
    if synchronize is None:raise ValueError("A CUDA synchronization callback is required")
    out=Path(out_dir);rows=[];total=0
    with writer_lock(out):
        atomic_json(out/"timing_manifest.json",_json(chosen))
        for record in chosen:
            for mode in ("A0","A_score","A_texture"):
                for iteration in range(13):
                    synchronize();begin=time.perf_counter()
                    pred=predict_fn(record,mode)
                    synchronize();elapsed=time.perf_counter()-begin;total+=1
                    if pred.get("executed_forward",True) is not True:
                        raise ValueError("Timing predictor returned cached output instead of running a forward")
                    rows.append({"pair_id":_pair_id(record),"mode":mode,"iteration":iteration,
                        "purpose":"warmup" if iteration<3 else "steady_repeat",
                        "seconds":elapsed,"components":_json(pred.get("timing",{})),
                        "memory":_json(pred.get("memory",{}))})
                    del pred
                    atomic_json(out/"state.json",{"status":"running","completed_forwards":total,
                        "expected_forwards":len(chosen)*39,"heartbeat_unix":time.time()})
        modes={}
        for mode in ("A0","A_score","A_texture"):
            sample=[r["seconds"] for r in rows if r["mode"]==mode and r["purpose"]=="steady_repeat"]
            modes[mode]={"count":len(sample),"mean_seconds":float(np.mean(sample)),
                "median_seconds":float(np.median(sample)),"p95_seconds":float(np.quantile(sample,.95))}
        result={"status":"committed","pair_count":len(chosen),"warmup_forwards":len(chosen)*9,
            "timing_forwards":len(chosen)*30,"geometry_calls":0,"modes":modes,
            "scope":"Small fixed validation subset, resident execution. Not a deployment p95. Cold model load and I/O/component costs are reported by the caller."}
        atomic_json(out/"timing_records.json",rows);atomic_json(out/"timing_summary.json",result)
        atomic_json(out/"state.json",result);return result


def run_evaluation(records,predict_fn,out_dir,checkpoint_id,identity=None,synchronize=None,
                   geometry_watchdog_seconds=None,geometry_python="/home/bupt/miniconda3/envs/hybrid-eval/bin/python"):
    """Best-checkpoint whole validation, independent texture pilot, and measured cost."""
    out=Path(out_dir)
    scores=evaluate_reliability(records,predict_fn,out/"reliability",checkpoint_id,identity)
    pilot=run_pilot(records,predict_fn,out/"pilot",identity=identity,
        geometry_python=geometry_python,geometry_watchdog_seconds=geometry_watchdog_seconds)
    costs=measure_costs(records,predict_fn,out/"costs",synchronize)
    result={"status":"committed" if pilot["status"]=="committed" else "blocked",
        "checkpoint_id":checkpoint_id,"score_metrics":scores,"pilot":pilot,"costs":costs}
    atomic_json(out/"evaluation_summary.json",result)
    return result
