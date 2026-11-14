"""Real-pair GPU regression gates. Called serially by the stage controller."""
import gc
import importlib.util
import math
import os
from pathlib import Path
import sys
import time
import types
import uuid
import numpy as np
import torch
from .runtime import atomic_json, rng_state, set_rng, resource_sample, sha

CORE_KEYS = ("b_idx_c", "i_idx_c", "j_idx_c", "coarse_coord_0", "coarse_coord_1", "coarse_confidence")
FINE_KEYS = ("b_idx_it", "m_idx_it", "i_idx_it", "j_idx_it", "fine_coord_0", "fine_coord_1")


def _save_arrays(path, arrays):
    path=Path(path);tmp=path.with_name(path.name+"."+uuid.uuid4().hex+".tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(f,**{k:v.numpy() for k,v in arrays.items()});f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)


def _compare(expected, actual, keys, atol=0.0, rtol=0.0):
    result={}
    for key in keys:
        if key not in expected or key not in actual:
            raise AssertionError("Regression field missing: "+key)
        a,b=expected[key],actual[key]
        if a.shape!=b.shape:raise AssertionError(f"{key} shape/order changed: {tuple(a.shape)} != {tuple(b.shape)}")
        discrete=not a.is_floating_point()
        same=torch.equal(a,b) if discrete else torch.allclose(a,b,atol=atol,rtol=rtol,equal_nan=False)
        maximum=float((a.double()-b.double()).abs().max()) if a.numel() else 0.0
        result[key]={"pass":bool(same),"exact":bool(torch.equal(a,b)),"max_abs_difference":maximum,"count":a.numel()}
        if not same:raise AssertionError(f"{key} regression: max_abs={maximum}, atol={atol}, rtol={rtol}")
    return result


def _capture_native_confidence(model):
    """Observe native top-k locals, without replacing its matching calculation.

    Engineering is batch_size=1. The native method's final frame therefore
    contains the one actual sparse matrix's top-k values and validity mask.
    """
    original=model._coarse_match_norm_clue
    code=original.__func__.__wrapped__.__code__
    def wrapped(self,data,*args,**kwargs):
        if data["batch_size"]!=1:raise ValueError("Baseline observer requires batch_size=1")
        captured={}
        def observe(frame,event,arg):
            if event=="return" and frame.f_code is code:
                values=frame.f_locals.get("top_k_values")
                mask=frame.f_locals.get("valid_matches")
                if values is not None and mask is not None:captured["confidence"]=values[mask].detach()
        previous=sys.getprofile();sys.setprofile(observe)
        try:original(data,*args,**kwargs)
        finally:sys.setprofile(previous)
        data["coarse_confidence"]=captured.get("confidence",data["feat0_c"].new_empty((0,)))
    model._coarse_match_norm_clue=types.MethodType(wrapped,model)


def _run(runtime, model, record, full, variant, out_dir, baseline=False):
    from reliability_data import load_pair
    started=time.perf_counter()
    data=load_pair(record,device="cuda",resolution=runtime.config["resolution"])
    io=time.perf_counter()-started
    model.eval()
    torch.cuda.synchronize();started=time.perf_counter()
    with torch.no_grad(),torch.cuda.amp.autocast(enabled=runtime.config["amp"]):
        if baseline:model(data)
        else:model(data,coarse_only=not full)
    torch.cuda.synchronize();seconds=time.perf_counter()-started
    runtime.calls["backbone"]+=1
    runtime.calls["engineering_forward"]=runtime.calls.get("engineering_forward",0)+1
    if baseline:runtime.calls["baseline_backbone"]=runtime.calls.get("baseline_backbone",0)+1
    if full:runtime.calls["original_fine"]+=1
    if not baseline and model.reliability_enabled:runtime.calls["head_forward"]+=1
    for key in ("feat0_c","feat1_c","coarse_confidence","fine_coord_0","fine_coord_1","reliability_logit","reliability_score"):
        if key in data and not torch.isfinite(data[key]).all():raise FloatingPointError("Nonfinite "+variant+":"+key)
    keys=CORE_KEYS+(FINE_KEYS if full else ())+("coarse_match_id","coarse_row_id","fine_parent_id","reliability_score","reliability_logit","fine_parent_reliability_score")
    arrays={k:data[k].detach().cpu().clone() for k in keys if k in data}
    candidates={k:[v.detach().cpu().clone() for v in data[k]] for k in
                ("candidate_ids0","candidate_ids1","native_candidate_ids0","native_candidate_ids1") if k in data}
    metadata={"pair_id":record["pair_id"],"scene":record["scene"],"variant":variant,"seconds":seconds,"io_seconds":io,
              "coarse_parents":int(data["b_idx_c"].numel()),"fine_outputs":int(data["fine_coord_0"].shape[0]) if full else 0,
              "resources":resource_sample(),"candidate_counts":data.get("candidate_counts"),
              "correlation_shapes":data.get("correlation_shapes"),"feature_shapes":[list(data["feat0_c"].shape),list(data["feat1_c"].shape)]}
    artifact=Path(out_dir)/(variant+".npz")
    _save_arrays(artifact,arrays)
    metadata["artifact"]={"path":str(artifact),"sha256":sha(artifact)}
    del data
    gc.collect()
    return arrays,candidates,metadata


def check_model(runtime, records, out_dir):
    """Check three fixed real pairs; does not perform any optimizer update."""
    out_dir=Path(out_dir);out_dir.mkdir(parents=True,exist_ok=True)
    records=list(records)[:3]
    if not records:raise RuntimeError("No real training pair for model acceptance")
    baseline_path=runtime.run/"baseline_code"/"src"/"slim.py"
    if not baseline_path.is_file():
        baseline_path=Path(runtime.config["run_dir"])/"baseline_code"/"src"/"slim.py"
    if not baseline_path.is_file():raise RuntimeError("Frozen baseline source missing")
    spec=importlib.util.spec_from_file_location("src._baseline_slim",baseline_path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    rng=rng_state();model=runtime.model
    saved={k:getattr(model,k) for k in ("capture_reliability","reliability_enabled","reliability_head","sparse_direct","texture_budget_enabled")}
    head_mode=runtime.head.training
    evidence={"status":"running","baseline_source":str(baseline_path),"baseline_sha256":sha(baseline_path),
              "pairs":[],"tolerance":{"atol":runtime.config.get("regression_atol",runtime.config.get("atol",0.0)),"rtol":runtime.config.get("regression_rtol",runtime.config.get("rtol",0.0))},
              "cpu_edge_case_evidence":str(Path(runtime.config["run_dir"])/"model_cpu_contracts_observer_pass.log")}
    atol,rtol=evidence["tolerance"]["atol"],evidence["tolerance"]["rtol"]
    baseline=None
    try:
        baseline=module.SLiM(runtime.native.MODEL)
        baseline.load_state_dict(model.state_dict(),strict=True)
        baseline=baseline.cuda().eval().requires_grad_(False)
        _capture_native_confidence(baseline)
        for number,record in enumerate(records):
            pair_dir=out_dir/f"pair_{number:02d}";pair_dir.mkdir(exist_ok=True)
            runs=[];checks={}
            original,_,meta=_run(runtime,baseline,record,True,"baseline",pair_dir,baseline=True);runs.append(meta)
            model.reliability_enabled=False;model.texture_budget_enabled=False;model.sparse_direct=False;model.capture_reliability=True
            disabled,_,meta=_run(runtime,model,record,True,"disabled",pair_dir);runs.append(meta)
            checks["baseline_disabled"]=_compare(original,disabled,CORE_KEYS+FINE_KEYS,atol,rtol)
            repeated,_,meta=_run(runtime,model,record,True,"repeated_r832",pair_dir);runs.append(meta)
            checks["repeated_r832"]=_compare(disabled,repeated,CORE_KEYS+FINE_KEYS,atol,rtol)
            model.sparse_direct=True
            sparse,_,meta=_run(runtime,model,record,True,"sparse_direct",pair_dir);runs.append(meta)
            checks["sparse_direct"]=_compare(disabled,sparse,CORE_KEYS+FINE_KEYS,atol,rtol)
            coarse,_,meta=_run(runtime,model,record,False,"coarse_only",pair_dir);runs.append(meta)
            checks["coarse_only"]=_compare(sparse,coarse,CORE_KEYS+("coarse_match_id",),atol,rtol)
            model.sparse_direct=saved["sparse_direct"]
            runtime.head.eval();model.enable_reliability(runtime.head)
            score,_,meta=_run(runtime,model,record,True,"score_only",pair_dir);runs.append(meta)
            checks["score_only"]=_compare(disabled,score,CORE_KEYS+FINE_KEYS,atol,rtol)
            if not torch.equal(score["fine_parent_id"],score["coarse_match_id"][score["m_idx_it"]]):
                raise AssertionError("Fine parent identity inheritance changed")
            if not torch.equal(score["fine_parent_reliability_score"],score["reliability_score"][score["m_idx_it"]]):
                raise AssertionError("Fine score was not inherited by parent")
            model.reliability_enabled=False;model.reliability_head=saved["reliability_head"]
            rollback,_,meta=_run(runtime,model,record,True,"rollback",pair_dir);runs.append(meta)
            checks["rollback"]=_compare(disabled,rollback,CORE_KEYS+FINE_KEYS,atol,rtol)
            evidence["pairs"].append({"pair_id":record["pair_id"],"checks":checks,"physical_runs":runs})
            atomic_json(out_dir/"model_checks.json",evidence)
        evidence["status"]="pass"
        evidence["real_pair_count"]=len(records)
        evidence["scope"]="Engineering R832 native, sparse shortcut, coarse shortcut, score-only, rollback; no generalization claim"
    except Exception as exc:
        evidence["status"]="fail";evidence["error"]=repr(exc)
        atomic_json(out_dir/"model_checks.json",evidence)
        raise
    finally:
        for key,value in saved.items():setattr(model,key,value)
        runtime.head.train(head_mode)
        if baseline is not None:del baseline
        gc.collect();set_rng(rng)
    runtime.check_frozen()
    atomic_json(out_dir/"model_checks.json",evidence)
    return {name:{"status":"pass","evidence":str(out_dir/"model_checks.json"),"real_pairs":len(records)}
            for name in ("baseline_disabled_regression","r832_numerical_stability","sparse_direct_equivalence",
                         "coarse_only_equivalence","score_only_invariance","disabled_rollback","parent_id_inheritance")}


def check_texture(runtime, records, out_dir):
    """Production-path fixed-budget gate after training-only threshold freezing."""
    out_dir=Path(out_dir);out_dir.mkdir(parents=True,exist_ok=True)
    threshold=runtime.model.texture_threshold
    if threshold is None or not math.isfinite(float(threshold)):
        raise RuntimeError("Texture threshold must be frozen from training pairs before acceptance")
    records=list(records)[:3]
    if not records:raise RuntimeError("No real texture acceptance pair")
    model=runtime.model
    saved={k:getattr(model,k) for k in ("texture_budget_enabled","texture_fraction","reliability_enabled")}
    evidence={"status":"running","threshold":float(threshold),"pairs":[]}
    try:
        model.reliability_enabled=False
        for number,record in enumerate(records):
            pair_dir=out_dir/f"pair_{number:02d}";pair_dir.mkdir(exist_ok=True)
            model.texture_budget_enabled=False
            native,native_c,meta0=_run(runtime,model,record,True,"native",pair_dir)
            model.texture_budget_enabled=True;model.texture_fraction=0.0
            zero,zero_c,meta1=_run(runtime,model,record,True,"reserve_0",pair_dir)
            _compare(native,zero,CORE_KEYS+FINE_KEYS,runtime.config.get("regression_atol",runtime.config.get("atol",0.0)),runtime.config.get("regression_rtol",runtime.config.get("rtol",0.0)))
            for key in ("candidate_ids0","candidate_ids1"):
                for left,right in zip(native_c[key],zero_c[key]):
                    if not torch.equal(left,right):raise AssertionError("Zero texture allocation changed native candidates")
            model.texture_fraction=0.2
            allocated,allocated_c,meta2=_run(runtime,model,record,True,"reserve_02",pair_dir)
            if meta0["candidate_counts"]!=meta2["candidate_counts"] or meta0["correlation_shapes"]!=meta2["correlation_shapes"]:
                raise AssertionError("Texture allocation changed native candidate/correlation budget")
            for key in ("candidate_ids0","candidate_ids1"):
                for ids in allocated_c[key]:
                    if ids.unique().numel()!=ids.numel():raise AssertionError("Texture introduced duplicate candidate IDs")
            evidence["pairs"].append({"pair_id":record["pair_id"],"status":"pass","physical_runs":[meta0,meta1,meta2]})
            atomic_json(out_dir/"texture_checks.json",evidence)
        evidence["status"]="pass"
    except Exception as exc:
        evidence["status"]="fail";evidence["error"]=repr(exc)
        atomic_json(out_dir/"texture_checks.json",evidence);raise
    finally:
        for key,value in saved.items():setattr(model,key,value)
        gc.collect()
    runtime.check_frozen()
    atomic_json(out_dir/"texture_checks.json",evidence)
    return {name:{"status":"pass","evidence":str(out_dir/"texture_checks.json"),"real_pairs":len(records)}
            for name in ("texture_zero_restores_native","texture_fixed_candidate_budget")}
