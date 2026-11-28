"""Serial, restartable orchestration. Every scientific stage has a recorded command and receipt."""
import hashlib
import os
import shlex
import shutil
import subprocess
import sys
import traceback
from pathlib import Path
from .options import parser,plan
from ..utils.run_io import EXP,ROOT,PKG,config,locations,lock,read,write,append,digest,sha,now


def source_identity():
    paths = []
    for name in ("configs","models","losses","data","utils","training","evaluation","tests","scripts","manifests"):
        paths.extend(p for p in (EXP/name).rglob("*") if p.is_file() and p.suffix in {".py",".sh",".yaml",".ini",".json"})
    for name in ("src","configs","utils"):
        paths.extend((ROOT/name).rglob("*.py"))
    paths.append(EXP/"protocol_pro5000.md")
    return digest({str(p.relative_to(ROOT)):sha(p) for p in sorted(paths)})


class Pipeline:
    def __init__(self,args):
        self.args=args
        self.sides,self.models,self.budgets=plan(args)
        _,self.report=locations(args.mode,args.run_id)
        self.progress=self.report/"progress.json"
        self.opts={k:str(v.resolve()) if isinstance(v,Path) else v for k,v in vars(args).items()}
        self.source=source_identity()
        settings={k:v for k,v in self.opts.items() if k not in ("resume_pipeline","monitor","python")}
        self.weight=sha(args.weights) if args.weights.is_file() else None
        self.identity=digest(dict(options=settings,source=self.source,weights=self.weight,budgets=self.budgets))
        self.cache=EXP/"outputs"/"shared"/"indices"/digest(dict(source=self.source,base=str(args.data_root.resolve())))[:24]
        self.errors=[]
        self.blocked=[]
        self.ready={}
        self.state={}
        self.command=["bash",str((EXP/"scripts/run_full_pipeline.sh").relative_to(ROOT)),*sys.argv[1:]]
        self.precision="bfloat16"
        self.eval_precision=config("P1")["eval_precision"]

    def save(self,name,status):
        self.state.update(stage=name,status=status,updated=now())
        write(self.progress,self.state)

    def out(self,model):
        return locations(self.args.mode,self.args.run_id,model)[0]

    def error(self,name,cmd,job,code,parent_trace=None):
        report=self.report
        worker=Path(job.get("directory",report))/"worker_error.json"
        details=read(worker) if worker.is_file() else {}
        current=read(report/"live_status.json") if (report/"live_status.json").is_file() else {}
        logs=(report/"pipeline.log")
        tail=""
        if logs.is_file():
            with logs.open("rb") as f:
                f.seek(max(0,logs.stat().st_size-16000))
                tail=f.read().decode(errors="replace")
        batch_path=report/"current_batch.json"
        batch=read(batch_path) if batch_path.exists() else {}
        ckpts=sorted(self.out(job.get("model","P0")).rglob("checkpoint_last.ckpt"),key=lambda p:p.stat().st_mtime)
        resume=self.command if "--resume-pipeline" in self.command else [*self.command,"--resume-pipeline"]
        bundle=report/"debug_bundle"
        bundle.mkdir(parents=True,exist_ok=True)
        data=dict(stage=name,model=job.get("model"),side=job.get("side"),step=current.get("step"),exit_code=code,
                  job=job,config=config(job.get("model","P0")),failed_command=shlex.join(cmd),
                  resume_command=shlex.join(resume),latest_checkpoint=str(ckpts[-1]) if ckpts else None,
                  last_observed=current,traceback=parent_trace or details.get("traceback",tail),gpu=details.get("memory",current.get("memory")),
                  last_observed_shapes=batch.get("shapes",current.get("shapes")),batch=batch,tail=tail,time=now())
        write(report/"latest_error.json",data)
        write(bundle/(name+"_"+now().replace(":","-")+".json"),data)
        (report/"latest_error.md").write_text(f"# {name}\n\nExit {code}.\n\n```text\n{data['traceback']}\n```\n\n```bash\n{data['resume_command']}\n```\n")
        self.errors.append(name)

    def stage(self,name,job=None,cmd=None,expected=None):
        job=dict(options=self.opts,identity=self.identity,source=self.source,report=str(self.report),
                 cache=str(self.cache),**(job or {}))
        job.setdefault("directory",str(self.report))
        entry=self.state["stages"].get(name,{})
        if entry.get("status")=="PASS" and (expected is None or expected.exists()):
            return True
        if cmd is None:
            path=self.report/"jobs"/(name+".json")
            write(path,job)
            cmd=[sys.executable,"-m",PKG+".training.worker",str(path)]
        item=dict(status="RUNNING",started=now(),command=shlex.join(cmd),job=job)
        self.state["stages"][name]=item
        self.save(name,"RUNNING")
        print(f"\n{name}  {item['command']}",flush=True)
        with (self.report/"pipeline.log").open("a",buffering=1) as logfile:
            logfile.write(f"\n{item['started']} {name} {item['command']}\n")
            proc=subprocess.Popen(cmd,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,errors="replace",bufsize=1)
            for line in proc.stdout:
                print(line,end="",flush=True)
                logfile.write(line)
            code=proc.wait()
        status="PASS" if code==0 and (expected is None or expected.exists()) else "FAIL"
        item.update(status=status,exit_code=code,ended=now())
        self.save(name,"RUNNING")
        if status!="PASS":
            self.error(name,cmd,job,code)
            return False
        if expected is not None and expected.suffix==".json" and read(expected).get("status")=="BLOCKED":
            self.blocked.append(name)
            item["status"]="BLOCKED"
            self.save(name,"RUNNING")
        return True

    def evaluate(self,model,side,domain,split,out,checkpoint,precision):
        name=f"{model}_R{side}_{split}_{domain}"
        return self.stage(name,dict(action="eval",model=model,side=side,domain=domain,split=split,directory=str(out),
                          checkpoint=str(checkpoint) if checkpoint else None,precision=precision),expected=out/(domain+".json"))

    def train_model(self,model):
        args=self.args
        start=0
        prev=None
        for side,updates in zip(self.sides,self.budgets):
            out=self.out(model)/f"R{side}"
            probe=out/"probe"
            base=dict(model=model,side=side,precision=self.precision)
            if not self.stage(f"{model}_R{side}_probe",dict(base,action="probe",directory=str(probe)),expected=probe/"selected.json"):
                return False
            best=read(probe/"selected.json")
            runtime=dict(base,**{k:best[k] for k in ("micro","checkpointing","correlation_chunk","graph_chunk_size")})
            runtime["precision"]=best["precision"]
            smoke_sig=digest(dict(source=self.source,weights=self.weight,cfg=config(model),runtime=runtime,
                                  data=self.opts["data_root"],cap=args.vram_cap_gib,
                                  device=read(self.report/"server_preflight.json")["device"]["uuid"],
                                  packages=read(self.report/"server_preflight.json")["dependencies"]))
            receipt=EXP/"outputs/shared/smoke"/(smoke_sig+".json")
            def train(name,path,first,end,total,previous,formal):
                return self.stage(name,dict(runtime,action="train",directory=str(path),start=first,end=end,total=total,
                                  previous=str(previous) if previous else None,formal=formal),expected=path/"training_summary.json")
            def accepted(path):
                summary=read(path/"training_summary.json")
                development=read(path/"after/megadepth.json")
                if summary["step"]-summary["job"]["start"]!=20 or development["status"]!="PASS":
                    raise RuntimeError(f"smoke acceptance lacks 20 completed updates and MegaDepth dev metrics: {path}")
                write(receipt,dict(signature=smoke_sig,model=model,side=side,run_id=args.run_id,
                                   summary=str(path/"training_summary.json"),sha256=sha(path/"training_summary.json")))
            valid=False
            if receipt.is_file():
                saved=read(receipt)
                p=Path(saved["summary"])
                valid=p.is_file() and saved["signature"]==smoke_sig and sha(p)==saved["sha256"]
            if args.mode in ("short","medium","full") and not valid:
                eng=out/"engineering_smoke"
                if not train(f"{model}_R{side}_smoke",eng,0,20,20,None,False):
                    return False
                accepted(eng)
            if not train(f"{model}_R{side}_{args.mode}_train",out,start,start+updates,sum(self.budgets),prev,True):
                return False
            for domain,status in read(out/"training_summary.json").get("development_status",{}).items():
                if status=="BLOCKED":
                    self.blocked.append(f"{model}_R{side}_{domain}_development")
            if args.mode=="smoke":
                accepted(out)
            prev=out/"checkpoint_last.ckpt"
            start+=updates
        out=self.out(model)/f"R{self.sides[-1]}"
        selected=out/"checkpoint_best.ckpt"
        if not selected.is_file():
            selected=out/"checkpoint_last.ckpt"
        self.ready[model]=(selected,self.eval_precision)
        return True

    def work(self):
        args=self.args
        if self.progress.exists():
            self.state=read(self.progress)
            if not args.resume_pipeline:
                raise FileExistsError("run ID already exists; choose a new ID or add --resume-pipeline")
            if self.state["identity"]!=self.identity:
                raise ValueError("code, config, weights, data paths or budget changed; choose a new run ID")
        else:
            self.state=dict(identity=self.identity,source=self.source,mode=args.mode,run_id=args.run_id,stages={},
                            options=self.opts,budgets=self.budgets,command=shlex.join(self.command))
            self.save("init","RUNNING")
        print("[1/10] Preflight",flush=True)
        if not self.stage("preflight",dict(action="preflight"),expected=self.report/"server_preflight.json"):
            return
        self.precision=read(self.report/"server_preflight.json")["precision"]
        print("[2/10] CPU / CUDA compatibility and data freeze",flush=True)
        env=os.environ
        env.update(CCS_WEIGHTS=str(args.weights),CCS_VRAM_CAP_GIB=str(args.vram_cap_gib))
        tests=["bash",str(EXP/"scripts/run_tests.sh"),"--python",sys.executable]
        if not self.stage("server_tests",cmd=tests):
            return
        scaling=[640,832,1024,1216] if args.mode!="smoke" and len(self.sides)>1 else self.sides
        ok=self.stage("prepare",dict(action="prepare",calibration_sides=sorted(set(scaling+self.sides))),expected=self.report/"training_data.json")
        if not ok:
            return
        data_ready=read(self.report/"training_data.json")["status"]=="PASS"
        print("[3/10] P0 official high-resolution development baseline",flush=True)
        if "P0" in self.models:
            success=True
            for side in self.sides:
                for domain in ("megadepth","texturepoor"):
                    ok=self.evaluate("P0",side,domain,"dev",self.out("P0")/f"R{side}/after",None,self.eval_precision)
                    success=ok and success
            if success:
                self.ready["P0"]=(None,self.eval_precision)
        failed=set()
        for i,model in enumerate(("P1","P2","P3","P4"),4):
            print(f"[{i}/10] {model} probe, smoke and training",flush=True)
            if model not in self.models:
                continue
            missing=({"P2"} if model=="P3" else {"P2","P3"} if model=="P4" else set()) & failed
            if missing or not data_ready:
                self.blocked.append(model)
                self.state["stages"][model]=dict(status="BLOCKED",reason=f"training data={data_ready}, dependencies={sorted(missing)}")
                failed.add(model)
            elif not self.train_model(model):
                failed.add(model)
        print("[8/10] Development selection and frozen formal evaluation",flush=True)
        chosen={m:dict(checkpoint=str(p or args.weights),sha256=sha(p or args.weights),precision=d,
                       resolution=self.sides[-1]) for m,(p,d) in self.ready.items()}
        freeze=dict(identity=self.identity,source=self.source,manifests=sha(EXP/"manifests/hashes.json"),
                    selection="Full-development LT recall; earliest trained step on tie; fixed last if development unavailable",
                    models=chosen)
        freeze_path=self.report/"model_freeze.json"
        if freeze_path.exists() and read(freeze_path)!=freeze:
            tested=any("_test_" in k and v.get("status")=="PASS" for k,v in self.state["stages"].items())
            previous=read(freeze_path)
            if tested and any(chosen.get(k)!=v for k,v in previous["models"].items()):
                raise ValueError("weights changed after formal evaluation")
        write(freeze_path,freeze)
        complete_fit=any(m!="P0" for m in self.ready) and not (set(self.models)-self.ready.keys())
        if args.mode=="full" and not args.dev_only and not complete_fit:
            self.blocked.append("formal_test_waiting_for_completed_full_training")
        if args.mode=="full" and not args.dev_only and complete_fit:
            for m,(p,prec) in self.ready.items():
                for domain in ("texturepoor","megadepth","scannet","hpatches"):
                    self.evaluate(m,self.sides[-1],domain,"test",self.out(m)/"test",p,prec)
        print("[9/10] Inference efficiency and resolution scaling",flush=True)
        for m,(p,prec) in self.ready.items():
            for side in scaling:
                out=self.out(m)/"efficiency"/f"R{side}"
                self.stage(f"{m}_R{side}_benchmark",dict(action="benchmark",model=m,side=side,
                           precision=prec,directory=str(out),checkpoint=str(p) if p else None),expected=out/"benchmark.json")

    def finish(self):
        print("[10/10] Research report",flush=True)
        status="FAIL" if self.errors else "PARTIAL" if self.blocked else "PASS"
        self.state.update(errors=self.errors,blocked=self.blocked)
        self.save("report",status)
        from ..evaluation.summarize_results import summarize
        summarize(self.args.mode,self.args.run_id)
        return 1 if status=="FAIL" else 2 if status=="PARTIAL" else 0


def main():
    args=parser().parse_args()
    if args.monitor:
        from .monitor import monitor
        monitor(args)
        return 0
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    p=Pipeline(args)
    with lock(p.report/"pipeline.lock",blocking=False):
        if p.progress.exists() and (not args.resume_pipeline or read(p.progress)["identity"]!=p.identity):
            raise RuntimeError("existing run differs or --resume-pipeline omitted; state preserved")
        try:
            p.work()
        except Exception:
            traceback.print_exc()
            stage=p.state.get("stage","pipeline")
            job=p.state.get("stages",{}).get(stage,{}).get("job",{})
            p.error(stage,p.command,job,1,parent_trace=traceback.format_exc())
        return p.finish()


if __name__=="__main__":
    sys.exit(main())
