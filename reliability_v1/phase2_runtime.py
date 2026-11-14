"""Second-task runtime: scene-balanced training, cumulative LR, fixed evaluation."""
import gc,json,math,os,time
from pathlib import Path
import numpy as np
import torch
from .runtime import Runtime,atomic_json,append_json,identity,sha,digest,save_checkpoint,load_checkpoint,state_hash,resource_sample
from .scene_protocol import PROTOCOL,parent_indices,skip_guard
from .scalar_lr import ScalarStore,scalar_features,fit_cumulative,lr_logits
from .comparison import evaluate,final_source_errors
from .phase1_prepare import FrozenRecords
from reliability_data import ManifestDataset

IDENTITY_FILES=("data_audit_summary.json","train_manifest.jsonl","val_manifest.jsonl","fixed_validation.json","fixed_final_epe.json",
                "statistics_train_pairs.json","lr_standardization.json","texture_calibration.json","sampling_freeze.json","core_inheritance.json","phase2_cost.json","label_probe_summary.json","label_scene_review.json","feature_label_contract.json","final_epe_contract.json")

REQUIRED_CHECKS={"audit_complete","split_and_access","validation_and_statistics","label_probe","implementation_tests","cost","launch_and_commit","phase1_report"}

def validate_readiness_evidence(receipt,summary,freeze,review,cost):
    if receipt.get("scope")!="phase1_full_data_scene_protocol" or set(receipt.get("checks",{}))!=REQUIRED_CHECKS:
        raise RuntimeError("Full Phase1 evidence required; smoke receipt cannot authorize formal training")
    if receipt.get("status")!="ready" or any(v.get("status")!="pass" for v in receipt["checks"].values()):
        raise RuntimeError("Phase1 is blocked")
    if not summary.get("audit_complete") or summary.get("committed_indices")!=368 or summary.get("expected_indices")!=368 or summary.get("counts",{}).get("missing_indices",0):
        raise RuntimeError("Incomplete formal audit")
    train=set(summary["train_scenes"]);val=set(summary["val_scenes"]);excluded=set(summary["excluded_test_scenes"])|{"0015","0022"}
    if not train or not val or train&val or (train|val)&excluded:raise RuntimeError("Invalid final physical-scene split")
    if freeze.get("status")!="pass" or freeze.get("protocol")!=PROTOCOL or len(freeze.get("epoch_plans",[]))!=10 or freeze.get("final_epe_pairs")!=100:
        raise RuntimeError("Final sampled protocol is not frozen")
    if review.get("status")!="pass" or review.get("unresolved_scenes"):raise RuntimeError("Label scene reviews unresolved")
    if cost.get("status")!="pass" or not math.isfinite(cost.get("remaining_storage_peak_bytes",float("nan"))) or cost["remaining_storage_peak_bytes"]<0:
        raise RuntimeError("Missing checked cost and storage configuration")

def make_identity(config,run):
    run=Path(run);value=identity(config,{name:sha(run/name) for name in IDENTITY_FILES})
    value["phase2_entry_sha256"]=sha(Path(config["repo_root"])/"reliability_phase2.py")
    value["launcher_sha256"]=sha(Path(config["repo_root"])/"launch_reliability_phase2.sh")
    value["audit_database_sha256"]=sha(run/"data_audit.sqlite")
    value["epoch_plan_sha256"]={p.name:sha(p) for p in sorted((run/"plans").glob("epoch_*.json"))}
    return value

def verify_ready(config,run):
    run=Path(run);receipt=json.loads((run/"phase1_ready.json").read_text())
    validate_readiness_evidence(receipt,*[json.loads((run/n).read_text()) for n in
        ("data_audit_summary.json","sampling_freeze.json","label_scene_review.json","phase2_cost.json")])
    ident=make_identity(config,run)
    if receipt["identity"]!=ident:raise RuntimeError("Preparation identity changed")
    if config["training_protocol"]!=PROTOCOL:raise RuntimeError("Unconfirmed scene-balanced training protocol")
    import shutil
    needed=json.loads((run/"phase2_cost.json").read_text())["remaining_storage_peak_bytes"]+2*2**30
    if min(shutil.disk_usage(run).free,shutil.disk_usage("/mnt/c").free)<needed:raise RuntimeError("Insufficient host/guest storage reserve")
    return ident

def build_runtime(config,run):
    runtime=Runtime(config,run)
    runtime.model.texture_threshold=json.loads((Path(run)/"texture_calibration.json").read_text())["threshold"]
    return runtime

class Predictor:
    def __init__(self,runtime,stats,fit,full=False):self.runtime=runtime;self.stats=stats;self.fit=fit;self.full=full
    def __call__(self,record):
        r=self.runtime;r.head.eval()
        try:data,labels=r.forward(record,full=self.full,texture=False)
        except (ValueError,RuntimeError,OSError) as error:
            atomic_json(r.run/"validation_identity_failure.json",dict(pair_id=record["pair_id"],direction=record["direction"],
                reason=repr(error),expected_file_sha256=record["file_sha256"],paths={k:record[k] for k in ("image0","image1","depth0","depth1")},
                action="pause affected fixed validation; no replacement or resampling",time=time.time()))
            raise
        with torch.no_grad():head=r.logits(data).cpu().numpy()
        raw=data["coarse_match_id"].cpu().numpy()
        conf=data["coarse_confidence"].cpu().numpy()
        x=scalar_features(conf,data["coarse_descriptor0"].float().norm(dim=-1).cpu().numpy(),data["coarse_descriptor1"].float().norm(dim=-1).cpu().numpy())
        from src.texture_budget import local_texture
        texture,count=local_texture(data["image0"],data["valid_mask0"],data["hw0_c"])
        low=((texture[0]<=r.model.texture_threshold)&data["mask0"].flatten()&(count[0]>0))[data["i_idx_c"]].cpu().numpy()
        y=np.asarray(labels["labels"],np.int8)
        result=dict(raw_id=raw,head=head,lr=lr_logits(x,self.stats,self.fit),mconf=conf,labels=y,low=low)
        if self.full:
            error,valid,gt=final_source_errors(data)
            result.update(child_parent_id=data["fine_parent_id"].cpu().numpy(),child_epe=error,child_valid=valid,
                          final_source_input=data["fine_coord_0"].cpu().numpy(),final_target_input=data["fine_coord_1"].cpu().numpy(),
                          final_gt_target_input=gt["points1_gt_input"],final_gt_reasons=gt["invalid_reason"],
                          input_to_orig0=np.asarray(data["_gt"]["plan0"]["scale_input_to_orig_xy"]),
                          input_to_orig1=np.asarray(data["_gt"]["plan1"]["scale_input_to_orig_xy"]))
        del data,labels,texture,count;gc.collect();return result

def supervised_visit(runtime,record,epoch,position,store,invalid_dir):
    invalid=Path(invalid_dir)/f"{epoch:02d}_{position:06d}.json"
    if invalid.exists():return json.loads(invalid.read_text())
    try:data,labels=runtime.forward(record,full=False,texture=False)
    except OSError as e:
        if e.errno in (5,6,19):raise
        result=dict(updated=False,reason="deep_check:"+repr(e),pair_id=record["pair_id"])
        atomic_json(invalid,result);return result
    except RuntimeError as e:
        if not str(e).startswith(("Frozen data checksum mismatch:","Frozen file changed during checksum verification:")):raise
        result=dict(updated=False,reason="deep_check:"+repr(e),pair_id=record["pair_id"])
        atomic_json(invalid,result);return result
    except (ValueError,AssertionError) as e:
        result=dict(updated=False,reason="GT_contract:"+repr(e),pair_id=record["pair_id"])
        atomic_json(invalid,result);return result
    raw=data["coarse_match_id"].cpu().numpy()
    selected=parent_indices(labels["valid"].detach().cpu().numpy(),raw,runtime.config["seed"],epoch,record["pair_id"])
    if not len(selected):
        result=dict(updated=False,reason="no_supervision",pair_id=record["pair_id"]);atomic_json(invalid,result);del data,labels;return result
    ids=torch.as_tensor(selected,device="cuda")
    small={k:data[k][ids] for k in ("coarse_descriptor0","coarse_descriptor1","coarse_confidence")}
    x=scalar_features(small["coarse_confidence"].cpu().numpy(),small["coarse_descriptor0"].float().norm(dim=-1).cpu().numpy(),small["coarse_descriptor1"].float().norm(dim=-1).cpu().numpy())
    target=labels["y"].detach().cpu().numpy().astype(np.float32)[selected]
    block=store.commit(epoch,position,record["pair_id"],record["direction"],raw[selected],x,target)
    runtime.head.train();runtime.optimizer.zero_grad(set_to_none=True)
    logits=runtime.logits(small);y=torch.as_tensor(target,device="cuda")
    loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,y)
    if not torch.isfinite(loss):raise FloatingPointError("Nonfinite head BCE")
    loss.backward()
    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in runtime.head.parameters()):raise FloatingPointError("Head gradient invalid")
    if any(p.grad is not None for p in runtime.model.parameters()):raise AssertionError("Frozen model received gradient")
    runtime.optimizer.step()
    if any(not torch.isfinite(p).all() for p in runtime.head.parameters()):raise FloatingPointError("Head parameter invalid")
    result=dict(updated=True,reason=None,pair_id=record["pair_id"],loss=float(loss.detach()),sampled_parents=len(selected),scalar_block=block,resources=resource_sample())
    del data,labels,small,logits,loss,ids,y;runtime.optimizer.zero_grad(set_to_none=True);return result

def train(config,run,resume=False):
    run=Path(run);ident=verify_ready(config,run);out=run/"formal_train";out.mkdir(exist_ok=True)
    runtime=build_runtime(config,run);stats=json.loads((run/"lr_standardization.json").read_text());dataset=ManifestDataset(run/"train_manifest.jsonl")
    validation=FrozenRecords(run/"fixed_validation.json").records
    store=ScalarStore(out/"scalar_records",ident);last=out/"last.json";best=out/"best.json"
    state=dict(epoch=0,next_pos=0,phase="train",updates=0,guard={},best_bce=None,best_epoch=None,auroc_best=None,auroc_best_epoch=None)
    if resume:
        state=load_checkpoint(last,runtime.head,runtime.optimizer,ident)["position"]
    elif last.exists():raise RuntimeError("Formal run already exists; use --resume")
    else:save_checkpoint(last,runtime.head,runtime.optimizer,state,ident)
    visit_seconds=[]
    try:
        for epoch in range(state["epoch"],10):
            plan=json.loads((run/"plans"/f"epoch_{epoch:02d}.json").read_text())
            if plan["content_sha256"]!=digest({k:v for k,v in plan.items() if k!="content_sha256"}):raise ValueError("Sampling plan checksum changed")
            if state["phase"]=="train":
                for pos in range(state["next_pos"],len(plan["visits"])):
                    v=plan["visits"][pos];record=dataset[v["manifest_position"]]
                    if record["pair_id"]!=v["pair_id"]:raise AssertionError("Sampling cursor identity mismatch")
                    atomic_json(out/"heartbeat.json",dict(stage="train",epoch=epoch,position=pos,pair_id=record["pair_id"],time=time.time(),pid=os.getpid()))
                    started=time.monotonic();result=supervised_visit(runtime,record,epoch,pos,store,out/"invalid")
                    state["guard"]=skip_guard(state["guard"],result["reason"]);state["updates"]+=int(result["updated"]);state["next_pos"]=pos+1
                    seconds=time.monotonic()-started;visit_seconds.append(seconds)
                    if len(visit_seconds)>100:visit_seconds.pop(0)
                    append_json(out/"physical_visits.jsonl",dict(epoch=epoch,position=pos,seconds=seconds,**result))
                    if len(visit_seconds)>=50 and (pos+1)%50==0:
                        remaining=len(plan["visits"])-pos-1+sum(len(json.loads((run/"plans"/f"epoch_{e:02d}.json").read_text())["visits"]) for e in range(epoch+1,10))
                        atomic_json(out/"eta.json",dict(measured_recent_visits=len(visit_seconds),mean_seconds=float(np.mean(visit_seconds)),
                            remaining_train_visits=remaining,estimated_remaining_train_seconds=remaining*float(np.mean(visit_seconds)),
                            note="Formal measured training ETA; LR, validation and EPE remain separate costs",time=time.time()))
                    if (pos+1)%50==0 or state["guard"]["pause"]:
                        runtime.check_frozen();save_checkpoint(last,runtime.head,runtime.optimizer,state,ident)
                    if state["guard"]["pause"]:raise RuntimeError("More than20 abnormal skips among first100 planned visits; saved cursor, no resampling")
                state["phase"]="lr_fit";save_checkpoint(last,runtime.head,runtime.optimizer,state,ident)
            lrpath=out/f"lr_epoch_{epoch:02d}.json"
            if state["phase"]=="lr_fit":
                atomic_json(out/"heartbeat.json",dict(stage="lr_fit",epoch=epoch,time=time.time(),pid=os.getpid()))
                fit_cumulative(store,epoch,stats,lrpath)
                state["phase"]="validation";save_checkpoint(last,runtime.head,runtime.optimizer,state,ident)
            fit=json.loads(lrpath.read_text())
            if state["phase"]=="validation":
                atomic_json(out/"heartbeat.json",dict(stage="validation",epoch=epoch,time=time.time(),pid=os.getpid()))
                eval_identity=dict(training=ident,head_sha=state_hash(runtime.head),lr_sha=sha(lrpath),epoch=epoch)
                summary=evaluate(validation,Predictor(runtime,stats,fit),out/f"validation_{epoch:02d}",eval_identity)
                bce=summary["head_pair_equal_BCE"];auc=summary["ranking"]["scene_equal"]["head"]["score_AUROC"]
                if bce is None or not math.isfinite(bce):raise RuntimeError("Validation BCE undefined")
                # Strict improvement retains the earlier epoch on exact ties.
                if state["best_bce"] is None or bce<state["best_bce"]:
                    state["best_bce"]=bce;state["best_epoch"]=epoch
                    save_checkpoint(best,runtime.head,runtime.optimizer,state,ident,extra=dict(epoch=epoch,lr_path=str(lrpath),lr_sha=sha(lrpath),validation_summary=str(out/f"validation_{epoch:02d}"/"summary.json")))
                if auc is not None and (state["auroc_best"] is None or auc>state["auroc_best"]):state["auroc_best"]=auc;state["auroc_best_epoch"]=epoch
                state.update(epoch=epoch+1,next_pos=0,phase="train")
                runtime.check_frozen();save_checkpoint(last,runtime.head,runtime.optimizer,state,ident)
                atomic_json(out/"status.json",dict(status="committed" if epoch==9 else "running",state=state))
        return state
    except BaseException as e:
        atomic_json(out/"failure.json",dict(reason=repr(e),last_valid_checkpoint=str(last),phase=state["phase"],epoch=state["epoch"],resume_command="python reliability_phase2.py full_train --config "+str(run/"phase2_config.json")+" --resume"))
        raise
    finally:dataset.close();store.close()

def final_epe(config,run,resume=False):
    run=Path(run);ident=verify_ready(config,run);out=run/"formal_train"
    status=json.loads((out/"status.json").read_text())
    if status["status"]!="committed" or status["state"]["epoch"]!=10:raise RuntimeError("Final EPE requires ten committed sampling epochs")
    runtime=build_runtime(config,run);obj=load_checkpoint(out/"best.json",runtime.head,None,ident,restore_rng=False)
    extra=obj["extra"];fitpath=Path(extra["lr_path"])
    if sha(fitpath)!=extra["lr_sha"]:raise ValueError("Best-epoch LR identity changed")
    stats=json.loads((run/"lr_standardization.json").read_text());fit=json.loads(fitpath.read_text())
    records=FrozenRecords(run/"fixed_final_epe.json").records
    return evaluate(records,Predictor(runtime,stats,fit,full=True),run/"final_epe",dict(training=ident,checkpoint_sha=sha(out/"best.json"),lr_sha=sha(fitpath)),kind="final_epe")

def report(config,run):
    run=Path(run);training=json.loads((run/"formal_train/status.json").read_text())
    final=json.loads((run/"final_epe/summary.json").read_text())
    if training["status"]!="committed" or final["status"]!="committed":raise RuntimeError("Only committed inputs can be reported")
    best=json.loads((run/"formal_train/best.json").read_text())
    if sha(run/"formal_train"/best["payload"])!=best["sha256"]:raise ValueError("Best checkpoint checksum changed")
    expected=json.loads((run/"formal_train"/f'validation_{training["state"]["best_epoch"]:02d}'/"summary.json").read_text())
    if expected["status"]!="committed":raise RuntimeError("Selected validation is not committed")
    auc_epoch=training["state"].get("auroc_best_epoch",training["state"]["best_epoch"])
    auc_summary=json.loads((run/"formal_train"/f"validation_{auc_epoch:02d}"/"summary.json").read_text())
    if auc_summary["status"]!="committed":raise RuntimeError("AUROC selection validation is not committed")
    selection=dict(bce_best_epoch=training["state"]["best_epoch"],auroc_best_epoch=auc_epoch,
        epochs_differ=auc_epoch!=training["state"]["best_epoch"],
        BCE_at_bce_best=expected["head_pair_equal_BCE"],BCE_at_auroc_best=auc_summary["head_pair_equal_BCE"],
        AUROC_at_bce_best=expected["ranking"]["scene_equal"]["head"]["score_AUROC"],
        AUROC_at_auroc_best=auc_summary["ranking"]["scene_equal"]["head"]["score_AUROC"])
    result=dict(status="committed",training=training,best_checkpoint_pointer=best,final_epe=final,selection_comparison=selection,
                coarse_validation_at_selected_epoch=expected,
                interpretation="Head minus LR tests a descriptor-based nonlinear model against three-scalar logistic regression; descriptor and nonlinearity contributions require scalar-MLP ablation.",
                selected_by_validation=True,single_training_seed=2026)
    atomic_json(run/"phase2_report.json",result);return result
