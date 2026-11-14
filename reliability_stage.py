"""Explicit engineering/full_train/evaluate/all entry; engineering is default."""
import argparse
import gc
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",default="configs/reliability_v1.json")
    parser.add_argument("--run-stage","--run_stage",choices=("engineering","full_train","evaluate","all"),default=None)
    parser.add_argument("--resume",action="store_true",help="Resume the same formal training checkpoint/epoch cursor")
    parser.add_argument("--audit-only",action="store_true")
    parser.add_argument("--worker",action="store_true",help=argparse.SUPPRESS)
    args=parser.parse_args()
    cfg=json.loads(Path(args.config).read_text());stage=args.run_stage or cfg.get("run_stage","engineering")
    if not args.worker:
        from reliability_v1.supervisor import supervise
        raise SystemExit(supervise(sys.argv[1:],cfg))
    expected_parent=os.environ.get("RELIABILITY_SUPERVISOR_PID")
    if expected_parent:
        import ctypes,signal
        if os.getppid()!=int(expected_parent):raise RuntimeError("Supervisor disappeared before worker startup")
        if ctypes.CDLL(None,use_errno=True).prctl(1,signal.SIGTERM,0,0,0)!=0:raise OSError(ctypes.get_errno(),"prctl parent-death signal failed")
        if os.getppid()!=int(expected_parent):raise RuntimeError("Supervisor disappeared while arming parent-death signal")
    root=Path(cfg["repo_root"]).resolve();run=Path(cfg["run_dir"]).resolve()
    os.chdir(root);sys.dont_write_bytecode=True
    if not run.is_relative_to(root/"runs"):raise ValueError("Outputs must remain inside SLiM ext4 runs")
    if cfg["epochs"]!=10 or cfg["resolution"]!=832 or cfg["batch_size"]!=1 or cfg["num_workers"]!=0:
        raise ValueError("This registered experiment requires 10 epochs, R832, batch1, workers0")
    if args.resume and stage not in ("full_train","all"):raise ValueError("--resume applies to formal training")
    from reliability_v1.runtime import (run_lock,atomic_json,identity,sha,digest,Runtime,load_checkpoint,state_hash)
    from reliability_v1.report import write_engineering,write_training,write_evaluation,commands,read
    from reliability_data import audit_data,ManifestDataset
    run.mkdir(parents=True,exist_ok=True)
    tasks=read(run/"stage_tasks.json",{"engineering":"planned","full_train":"planned","evaluate":"planned"})
    with run_lock(run):
        atomic_json(run/"commands.json",commands(cfg))
        atomic_json(run/"resolved_config.json",cfg)
        try:
            summary=read(run/"data_audit_summary.json")
            if summary is None:
                if (run/"data_audit.incomplete.sqlite").exists():
                    raise RuntimeError("Full data audit is still running/incomplete; inspect data_audit_progress.json, do not replace it")
                summary=audit_data(cfg,run)
            if summary["status"]!="committed" or not summary["train_count"] or not summary["val_count"]:
                raise RuntimeError("No committed trustworthy independent data split")
            for split in ("train","val"):
                if sha(run/(split+"_manifest.jsonl"))!=summary["manifest_sha256"][split]:raise RuntimeError("Frozen manifest checksum changed")
            if sha(run/"data_audit.sqlite")!=summary["data_audit_sha256"]:raise RuntimeError("Frozen audit database checksum changed")
            if args.audit_only:
                print(json.dumps(summary,indent=2));return
            train=ManifestDataset(run/"train_manifest.jsonl");val=ManifestDataset(run/"val_manifest.jsonl")
            from reliability_v1.engineering import calibration,engineering,cpu_contracts
            if stage in ("engineering","all") and not args.resume:cpu_contracts(cfg,run)
            threshold=calibration(cfg,train,run)
            data_ident={key:summary[key] for key in ("manifest_sha256","data_audit_sha256","candidate_audit_sha256","train_list_sha256","train_count","val_count","train_scenes","val_scenes")}
            data_ident["texture_calibration_sha256"]=sha(run/"texture_calibration.json")
            run_id=identity(cfg,data_ident)
            old_id=read(run/"run_identity.json")
            if old_id and old_id!=run_id and read(run/"engineering_acceptance.json",{}).get("status")=="pass":
                raise RuntimeError("Code/config/data changed after accepted engineering; use a new run")
            atomic_json(run/"run_identity.json",run_id)
            if shutil.disk_usage(run).free<cfg["minimum_free_gib"]*2**30:raise RuntimeError("Insufficient free disk reserve")
            if stage in ("engineering","all") and not args.resume:
                tasks["engineering"]="running";atomic_json(run/"stage_tasks.json",tasks)
                runtime=Runtime(cfg,run);runtime.model.texture_threshold=threshold["threshold"]
                engineering(runtime,train,val,run_id)
                tasks["engineering"]="committed";atomic_json(run/"stage_tasks.json",tasks)
                resources=read(run/"engineering/resources.json")
                from reliability_v1.metrics import estimate_validation_storage
                mean_parents=__import__("numpy").mean([json.loads(line)["valid_parents"] for line in (run/"engineering/physical_visits.jsonl").read_text().splitlines()])
                storage=estimate_validation_storage(len(val),float(mean_parents),epochs=cfg["epochs"])
                budget=dict(estimated=True,train_pairs=len(train),validation_pairs=len(val),train_scenes=len(summary["train_scenes"]),validation_scenes=len(summary["val_scenes"]),
                    epochs=cfg["epochs"],expected_training_pair_visits=len(train)*cfg["epochs"],
                    expected_training_backbone_forwards=len(train)*cfg["epochs"],expected_validation_backbone_forwards=len(val)*(cfg["epochs"]+1),
                    expected_updates_upper_bound=len(train)*cfg["epochs"],
                    training_seconds=resources["median_pair_seconds"]*len(train)*cfg["epochs"],
                    validation_seconds="See engineering validation summaries; independent I/O/valid-parent mix may change",
                    storage=storage,free_disk_bytes=shutil.disk_usage(run).free,
                    no_pair_or_step_cap=True,scope="full_available; estimates are not completed visits")
                atomic_json(run/"formal_budget.json",budget)
                write_engineering(cfg)
                del runtime;gc.collect();__import__("torch").cuda.empty_cache()
            if stage in ("full_train","all"):
                accepted=read(run/"engineering_acceptance.json",{})
                if accepted.get("status")!="pass" or accepted.get("identity")!=run_id:raise RuntimeError("Matching engineering acceptance required")
                # A fresh head/optimizer is mandatory even when 'all' follows smoke.
                budget=read(run/"formal_budget.json",{})
                needed=budget.get("storage",{}).get("conservative_peak_bytes")
                if needed is None:raise RuntimeError("Engineering disk budget missing")
                if shutil.disk_usage(run).free<needed:raise RuntimeError("Insufficient estimated full-training disk budget; do not shorten data or epochs")
                print(json.dumps(budget,indent=2),flush=True)
                runtime=Runtime(cfg,run);runtime.model.texture_threshold=threshold["threshold"]
                tasks["full_train"]="running";atomic_json(run/"stage_tasks.json",tasks)
                from reliability_v1.trainer import full_train
                full_train(runtime,train,val,run_id,resume=args.resume)
                tasks["full_train"]="committed";atomic_json(run/"stage_tasks.json",tasks);write_training(cfg)
                del runtime;gc.collect();__import__("torch").cuda.empty_cache()
            if stage in ("evaluate","all"):
                status=read(run/"full_train/status.json",{})
                if status.get("status")!="committed" or status.get("state",{}).get("epoch")!=cfg["epochs"]:
                    raise RuntimeError("Stage3 requires all 10 formal epochs committed")
                tasks["evaluate"]="running";atomic_json(run/"stage_tasks.json",tasks)
                runtime=Runtime(cfg,run);runtime.model.texture_threshold=threshold["threshold"]
                checkpoint=load_checkpoint(run/"full_train/best.json",runtime.head,None,run_id,restore_rng=False)
                best_epoch=checkpoint["position"]["best_epoch"]
                checkpoint_id=checkpoint["extra"]["checkpoint_id"]
                from reliability_v1.trainer import Predictor,validate
                # Reuse the complete same-checkpoint best validation committed during training.
                score=validate(runtime,val,run/"full_train"/f"validation_epoch_{best_epoch:02d}",checkpoint_id,run_id)
                del runtime;gc.collect();__import__("torch").cuda.empty_cache()
                from reliability_v1.evaluation import run_pilot,measure_costs
                from reliability_v1.exports import FilePredictor
                from reliability_v1.watchdog import geometry_timeout
                watchdog=geometry_timeout(run)
                out=run/"evaluation"
                exports=FilePredictor(str(Path(args.config).resolve()),run_id,out/"exports",run/"full_train/best.json")
                pilot=run_pilot(val,exports,out/"pilot",identity=run_id,
                                geometry_watchdog_seconds=watchdog["seconds"])
                costs=measure_costs(val,exports,out/"costs",__import__("torch").cuda.synchronize)
                result=dict(status=pilot["status"],checkpoint_id=checkpoint_id,score_metrics=score,pilot=pilot,costs=costs,
                    initialized_validation="full_train/validation_init",last_validation=f"full_train/validation_epoch_{cfg['epochs']-1:02d}",
                    training_seed=cfg["seed"],geometry_seeds=cfg["geometry"]["seeds"])
                atomic_json(out/"evaluation_summary.json",result);write_evaluation(cfg,result)
                tasks["evaluate"]=result["status"];atomic_json(run/"stage_tasks.json",tasks)
            train.close();val.close()
            print(json.dumps(dict(stage=stage,status=tasks,run_dir=str(run)),indent=2))
        except BaseException as exc:
            active=next((x for x in ("evaluate","full_train","engineering") if tasks.get(x)=="running"),stage)
            tasks[active]="failed"
            atomic_json(run/"stage_tasks.json",tasks)
            atomic_json(run/"failure.json",dict(stage=active,type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()))
            if active=="engineering":write_engineering(cfg,error=str(exc))
            if active=="full_train":write_training(cfg)
            raise

if __name__=="__main__":main()
