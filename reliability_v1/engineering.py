"""Real-data engineering acceptance; never launches formal epochs."""
import gc
import copy
import json
from pathlib import Path
import time
import os
import shutil
import subprocess
import sys
import uuid
import numpy as np
import torch
from .runtime import (atomic_json,append_json,digest,now,sha,state_hash,save_checkpoint,load_checkpoint,resource_sample,code_identity,rng_state)
from .trainer import train_step,validate,permutation,heartbeat

def assert_state_equal(expected,actual,atol=0.,rtol=0.):
    if torch.is_tensor(expected):
        torch.testing.assert_close(expected,actual,atol=atol,rtol=rtol)
    elif isinstance(expected,np.ndarray):
        np.testing.assert_array_equal(expected,actual)
    elif isinstance(expected,dict):
        if expected.keys()!=actual.keys():raise AssertionError("Restored state keys differ")
        for k in expected:assert_state_equal(expected[k],actual[k],atol,rtol)
    elif isinstance(expected,(list,tuple)):
        if len(expected)!=len(actual):raise AssertionError("Restored sequence length differs")
        for a,b in zip(expected,actual):assert_state_equal(a,b,atol,rtol)
    elif expected!=actual:raise AssertionError("Restored scalar differs")

def cpu_contracts(config,run):
    run=Path(run);expected=digest(code_identity(config["repo_root"]))
    receipt=run/"cpu_contracts.json"
    if receipt.exists():
        current=json.loads(receipt.read_text())
        if current.get("status")=="pass" and current.get("code_sha256")==expected:return current
    log=run/("cpu_contracts_"+uuid.uuid4().hex[:8]+".log")
    with log.open("x") as f:
        result=subprocess.run([sys.executable,"-m","unittest","discover","-s","tests","-p","test_reliability*.py","-v"],
            cwd=config["repo_root"],stdout=f,stderr=subprocess.STDOUT,
            env=dict(os.environ,PYTHONDONTWRITEBYTECODE="1",OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1"))
    value=dict(status="pass" if result.returncode==0 else "fail",code_sha256=expected,log=str(log),returncode=result.returncode)
    atomic_json(receipt,value)
    if result.returncode:raise RuntimeError("CPU contracts failed: "+str(log))
    return value

def calibration(runtime_config,train_records,run):
    from reliability_data import load_pair
    from src.texture_budget import local_texture,TEXTURE_SCHEMA
    target=Path(run)/"texture_calibration.json"
    ids=train_records.scene_round_robin_indices(min(100,len(train_records)))
    records=[train_records[i] for i in ids]
    source=dict(pair_ids=[r["pair_id"] for r in records],schema=TEXTURE_SCHEMA,quantile=1/3,resolution=832,texture_code_sha256=sha(Path(runtime_config["repo_root"])/"src/texture_budget.py"),loader_code_sha256=sha(Path(runtime_config["repo_root"])/"reliability_data.py"),source_manifest_sha256=sha(getattr(train_records,"path",Path(run)/"train_manifest.jsonl")),selected_file_identity=digest([r.get("file_sha256",{}) for r in records]))
    if target.exists():
        value=json.loads(target.read_text())
        if value["source"]!=source:raise RuntimeError("Texture calibration identity differs")
        return value
    values=[];started=time.perf_counter();pixels=0
    torch.set_num_threads(1)
    for record in records:
        heartbeat(run,"calibration","pair_start",pair_id=record["pair_id"])
        data=load_pair(record,device="cpu",resolution=832,include_gt=False)
        for side in (0,1):
            texture,count=local_texture(data[f"image{side}"],data[f"valid_mask{side}"],data[f"mask{side}"].shape[-2:])
            legal=data[f"mask{side}"].flatten(1)&(count>0)&torch.isfinite(texture)
            values.append(texture[legal].numpy().copy());pixels+=int(count[legal].sum())
        del data
    if not values or not sum(len(x) for x in values):raise RuntimeError("No legal train grids for texture calibration")
    pooled=np.concatenate(values)
    threshold=float(np.quantile(pooled,1/3))
    value=dict(source=source,threshold=threshold,grids=len(pooled),valid_window_pixel_visits=pixels,
               pair_count=len(records),seconds=time.perf_counter()-started,method="numpy linear quantile over all legal grids of selected training pairs; no GT",
               status="committed")
    atomic_json(target,value)
    return value

def engineering(runtime,train_records,val_records,run_identity,scope="full_manifest_engineering"):
    if scope not in ("full_manifest_engineering","engineering_core"):raise ValueError("Unknown acceptance scope")
    run=runtime.run;out=run/"engineering";out.mkdir(exist_ok=True)
    runtime.run_identity=run_identity
    accepted=run/("engineering_core_acceptance.json" if scope=="engineering_core" else "engineering_acceptance.json")
    if accepted.exists():
        old=json.loads(accepted.read_text())
        if old.get("status")=="pass":
            if old["identity"]!=run_identity or old.get("scope")!=scope:raise RuntimeError("Passed acceptance is stale; use a new run identity for changed code/data")
            return old
    train=[train_records[i] for i in train_records.scene_round_robin_indices(runtime.config["engineering_pairs"])]
    val=[val_records[i] for i in val_records.scene_round_robin_indices(runtime.config["engineering_val_pairs"])]
    if not train or not val:raise RuntimeError("No trustworthy disjoint train/validation pairs")
    if set(r["scene"] for r in train)&set(r["scene"] for r in val):raise AssertionError("Scene leakage")
    prior_updates=0
    prior_root=run/"engineering_attempts"
    candidates=[out]+list(prior_root.glob("*")) if prior_root.exists() else [out]
    for directory in candidates:
        ledger=directory/"optimizer_intents.jsonl"
        if ledger.exists():prior_updates+=len([x for x in ledger.read_text().splitlines() if x.strip()])
        else:
            for filename in ("physical_visits.jsonl","resume_physical_visits.jsonl"):
                p=directory/filename
                if p.exists():prior_updates+=sum(bool(json.loads(x).get("updated")) for x in p.read_text().splitlines() if x.strip())
    if prior_updates>39:raise RuntimeError("Engineering retry has insufficient remaining budget for a fresh eight-update plus recovery proof; preserve existing evidence")
    if any(p.name!="smoke_manifest.json" for p in out.iterdir()):
        # Preserve all previous engineering attempt artifacts before a retry.
        prior_root.mkdir(exist_ok=True)
        archive=prior_root/("attempt_"+str(len(list(prior_root.iterdir()))+1).zfill(3))
        if out.resolve().parent!=run.resolve() or archive.resolve().parent!=prior_root.resolve():raise RuntimeError("Unsafe archive path")
        shutil.move(str(out),str(archive));out.mkdir()
        atomic_json(out/"prior_attempts.json",dict(prior_physical_optimizer_updates=prior_updates,archive=str(archive)))
    atomic_json(out/"smoke_manifest.json",dict(train=train,val=val,scope="engineering only, not generalization"))
    checks={"cpu_contracts":{"status":"pass","evidence":"cpu_contracts.json"}}
    start=time.perf_counter()
    from .engineering_checks import check_model,check_texture
    checks.update(check_model(runtime,train[:3],out))
    checks.update(check_texture(runtime,train[:3],out))
    before=state_hash(runtime.head)
    parameters_before={n:p.detach().cpu().clone() for n,p in runtime.head.named_parameters()}
    state=dict(epoch=0,next_pos=0,global_step=0,attempted_pairs=0,order=[],physical_updates=0)
    resume_done=False;results=[];phys_updates=prior_updates
    checkpoint=out/"resume_probe.json"
    max_visits=min(len(train)*runtime.config["engineering_passes"],48-prior_updates)
    schedule=[]
    for epoch in range(runtime.config["engineering_passes"]):
        order=permutation(len(train),runtime.config["seed"],epoch).tolist()
        for pos,index in enumerate(order):schedule.append((epoch,pos,index,order))
    for ordinal,(epoch,pos,index,order) in enumerate(schedule[:max_visits]):
        result,grad=train_step(runtime,train[index],epoch,pos,out/"physical_visits.jsonl")
        results.append(result)
        if result["updated"]:phys_updates+=1
        state.update(epoch=epoch,next_pos=pos+1,order=order,global_step=state["global_step"]+int(result["updated"]),
                     attempted_pairs=ordinal+1,physical_updates=phys_updates)
        # Probe the same next real step from a committed optimizer/RNG/cursor state.
        if not resume_done and state["global_step"]>=8 and ordinal+1<len(schedule[:max_visits]):
            saved_rng=copy.deepcopy(rng_state());saved_optimizer=copy.deepcopy(runtime.optimizer.state_dict())
            save_checkpoint(checkpoint,runtime.head,runtime.optimizer,state.copy(),run_identity)
            next_epoch,next_pos,next_index,next_order=schedule[ordinal+1]
            continuous,gradient0=train_step(runtime,train[next_index],next_epoch,next_pos,out/"resume_physical_visits.jsonl")
            params0={n:p.detach().cpu().clone() for n,p in runtime.head.named_parameters()}
            optimizer0=copy.deepcopy(runtime.optimizer.state_dict())
            # Explicitly restore the whole training state, not just model tensors.
            restored=load_checkpoint(checkpoint,runtime.head,runtime.optimizer,run_identity)
            if restored["position"]!=state:raise AssertionError("Sampler/cursor restoration differs")
            assert_state_equal(saved_rng,rng_state())
            assert_state_equal(saved_optimizer,runtime.optimizer.state_dict())
            replay,gradient1=train_step(runtime,train[next_index],next_epoch,next_pos,out/"resume_physical_visits.jsonl")
            if continuous["updated"]!=replay["updated"]:raise AssertionError("Resume update decision differs")
            maxdiff=0.
            if continuous["updated"]:
                if not np.isclose(continuous["loss"],replay["loss"],atol=runtime.config["atol"],rtol=runtime.config["rtol"]):raise AssertionError("Resume loss differs")
                torch.testing.assert_close(gradient0,gradient1,atol=runtime.config["atol"],rtol=runtime.config["rtol"])
                for n,p in runtime.head.named_parameters():
                    torch.testing.assert_close(params0[n],p.detach().cpu(),atol=runtime.config["atol"],rtol=runtime.config["rtol"])
                    maxdiff=max(maxdiff,float((params0[n]-p.detach().cpu()).abs().max()))
                assert_state_equal(optimizer0,runtime.optimizer.state_dict(),runtime.config["atol"],runtime.config["rtol"])
                phys_updates+=2
            atomic_json(out/"resume_evidence.json",dict(status="pass",pair_id=train[next_index]["pair_id"],
                checkpoint_sha=sha(checkpoint),continuous_loss=continuous.get("loss"),restored_loss=replay.get("loss"),
                parameter_max_abs=maxdiff,gradient_max_abs=float((gradient0-gradient1).abs().max()) if gradient0 is not None else None,
                cursor=state.copy(),restored_rng=True,restored_optimizer=True,rng_exactly_compared=True,optimizer_state_compared=True,
                physical_branch_updates=int(continuous["updated"])+int(replay["updated"]),atol=runtime.config["atol"],rtol=runtime.config["rtol"]))
            # Return to checkpoint; normal schedule performs its next step once, recorded separately.
            load_checkpoint(checkpoint,runtime.head,runtime.optimizer,run_identity)
            del params0,optimizer0,gradient0,gradient1,saved_rng,saved_optimizer
            resume_done=bool(continuous["updated"])
        if phys_updates>50:raise AssertionError("Engineering update budget exceeded")
        if (ordinal+1)%len(train)==0:
            validate(runtime,val,out/f"validation_{ordinal+1:02d}",f"smoke:{ordinal+1}:"+state_hash(runtime.head),run_identity)
    if not state["global_step"]:raise RuntimeError("No real engineering updates; cannot accept")
    if not resume_done:raise RuntimeError("Not enough evaluable real steps to exercise resume")
    runtime.check_frozen()
    if before==state_hash(runtime.head):raise AssertionError("New head did not change")
    parameter_changes={n:dict(changed=not torch.equal(parameters_before[n],p.detach().cpu()),
        max_abs_difference=float((parameters_before[n]-p.detach().cpu()).abs().max())) for n,p in runtime.head.named_parameters()}
    atomic_json(out/"new_parameter_updates.json",parameter_changes)
    if not all(v["changed"] for v in parameter_changes.values()):raise AssertionError("A new trainable parameter did not update")
    del parameters_before
    state["physical_updates"]=phys_updates
    save_checkpoint(out/"smoke.json",runtime.head,runtime.optimizer,state,run_identity,
                    extra={"scope":"engineering_only_not_formal_training"})
    values=np.array([r["resources"]["allocated"] for r in results],np.int64)
    reserved=np.array([r["resources"]["reserved"] for r in results],np.int64)
    warm=max(4,len(train))
    after=values[warm:] if len(values)>warm else values
    # Persistent tensor allocation after released batches should remain bounded.
    growth=int(after[-1]-after[0]) if len(after) else 0
    if growth>128*2**20:raise RuntimeError("Persistent GPU allocation growth exceeds 128 MiB after warmup")
    measured=dict(load_seconds=runtime.load_seconds,physical_optimizer_updates=phys_updates,
        logical_optimizer_updates=state["global_step"],prior_attempt_physical_updates=prior_updates,training_visits=state["attempted_pairs"],
        train_pair_seconds=[r["seconds"] for r in results],median_pair_seconds=float(np.median([r["seconds"] for r in results])),
        pair_p95_seconds=float(np.quantile([r["seconds"] for r in results],.95)),
        postwarm_allocated_growth=growth,allocated_series=values.tolist(),reserved_series=reserved.tolist(),
        peak_allocated=max(r["resources"]["peak_allocated"] for r in results),peak_reserved=max(r["resources"]["peak_reserved"] for r in results),
        rss_peak=max(r["resources"]["rss_peak_bytes"] for r in results),
        host_memory_swap_series=[r["resources"] for r in results],elapsed_seconds=time.perf_counter()-start,calls=runtime.calls,
        limitation="At most 50 updates only; 10-epoch resource stability is not established")
    atomic_json(out/"resources.json",measured)
    atomic_json(run/"watchdog.json",dict(pair_timeout_seconds=max(120.,20*measured["pair_p95_seconds"]),
        cold_load_timeout_seconds=max(300.,10*runtime.load_seconds),source="engineering measured p95 plus generous cold-load margin",measured=True))
    checks.update(coordinate_label_production={"status":"pass","evidence":"engineering/label_cases","contract":"feature_label_contract.json"},real_training={"status":"pass","evidence":"engineering/physical_visits.jsonl"},
                  checkpoint_resume={"status":"pass","evidence":"engineering/resume_evidence.json"},
                  frozen_parameters_bn={"status":"pass","evidence":"parameter_audit.json","final_sha":state_hash(runtime.model)},
                  finite_loss_gradient_parameter={"status":"pass","evidence":"engineering/physical_visits.jsonl"},
                  new_parameters_updated={"status":"pass","before":before,"after":state_hash(runtime.head)},
                  continuous_resources={"status":"pass","evidence":"engineering/resources.json"},
                  independent_validation={"status":"pass","pairs":len(val),"scenes":sorted({r["scene"] for r in val})})
    for name,value in checks.items():
        if value.get("status")!="pass":raise RuntimeError("Engineering check did not pass: "+name)
    acceptance=dict(status="pass",scope=scope,formal_training_eligible=scope=="full_manifest_engineering",identity=run_identity,checks=checks,time=now(),
        physical_optimizer_updates=phys_updates,smoke_checkpoint="engineering/smoke.json",formal_training="not_started")
    atomic_json(accepted,acceptance)
    return acceptance
