import datetime, fcntl, os, subprocess, time
from .data import EXP,OUT,read,save
from .report import MODELS,render,run_path

def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def state(**info):
    info.update(utc=utc(),owner_pid=os.getpid())
    save(OUT/"state.json",info)
    parent=EXP/"reports/state.json";s=read(parent)
    s.update(current_phase="fast_screening",budget="screening",gate_status=info.get("status","RUNNING"),fast_screening=info)
    save(parent,s)

def cli(args,label,run=None,details=None):
    logdir=EXP/"logs/fast_screening";logdir.mkdir(exist_ok=True)
    path=logdir/(label+"_"+str(time.time_ns())+".log")
    command=["bash",str(EXP/"scripts/screen_python.sh"),*args]
    def tick(pid):
        prog=read(run/"progress.json") if run and (run/"progress.json").exists() else {}
        state(status="RUNNING",stage=label,worker_pid=pid,progress=prog,log=str(path.relative_to(EXP)),**(details or {}))
        (EXP/"reports/handoff.md").write_text(f"快速筛选 v1。当前 {label}，已提交 {prog.get('step',0)} 步。阶段状态、配置及进度在 reports/fast_screening/state.json。恢复入口 bash experiments/lowtex_e2e_v1/scripts/run_workflow.sh。R832 已归档；不自动启动长期训练。\n")
    if run and (run/"writer.lock").exists():
        with (run/"writer.lock").open("a") as lock:
            while True:
                try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
                except BlockingIOError:tick(None);time.sleep(30)
    with path.open("x") as f:
        print(command,file=f,flush=True)
        proc=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT)
        while True:
            tick(proc.pid)
            try:rc=proc.wait(timeout=45);break
            except subprocess.TimeoutExpired:
                render()
    path.with_suffix(".exitcode").write_text(str(rc)+"\n")
    if rc:raise RuntimeError(f"exit={rc} log={path.relative_to(EXP)} tail={path.read_text(errors='replace')[-1800:]}")
    return path

def exercise(model,mode,until=None):
    cfg=read(OUT/"selected_config.json")
    d=run_path(model) if mode=="screening" else EXP/"outputs/fast_screening_v1/smoke"/cfg["id"]/model
    def finish():
        if not (d/"timing.json").is_file():
            runarg=["-m","experiments.lowtex_e2e_v1.screening.timing","--run",str(d.relative_to(EXP))]
            if mode=="smoke":runarg.append("--smoke")
            cli(runarg,f"timing_{mode}_{model}",details=dict(experiment=model,mode=mode))
        return read(d/"complete.json")
    if (d/"complete.json").is_file():return finish()
    args=["-m","experiments.lowtex_e2e_v1.screening.train","--exp",model,"--mode",mode]
    if until:args+=["--until",str(until)]
    cli(args,f"{mode}_{model}",d,dict(experiment=model,mode=mode,run=str(d.relative_to(EXP))))
    return read(d/"progress.json") if until else finish()

def execute():
    render()
    receipt=OUT/"acceptance.json"
    if not receipt.exists():
        path=cli(["-m","unittest","-v","experiments.lowtex_e2e_v1.tests.test_texturepoor",
                    "experiments.lowtex_e2e_v1.tests.test_training","experiments.lowtex_e2e_v1.tests.test_screening"],"cli_tests")
        save(receipt,dict(log=str(path.relative_to(EXP)),status="PASS"))
    failures=read(OUT/"failures.json") if (OUT/"failures.json").exists() else {}
    def fail(name,error):
        failures[name]=str(error);save(OUT/"failures.json",failures);render()
    def passed(name):
        if name in failures:
            del failures[name];save(OUT/"failures.json",failures)
    models=MODELS[:-1]
    config=read(OUT/"selected_config.json")
    b2_verified=OUT/"batch_equivalence.json"
    if not b2_verified.exists():
        try:cli(["-m","experiments.lowtex_e2e_v1.screening.batch_check"],"batch_equivalence")
        except Exception as err:
            if not b2_verified.exists():save(b2_verified,{"pass":False,"error":str(err)})
    options=[config]+sorted([read(p) for p in (OUT/"bench").glob("*/result.json")
             if read(p).get("status")=="PASS" and read(p).get("min_free",0)>=1024**3 and read(p)["id"]!=config["id"]],
             key=lambda c:-c["pairs_s"])
    if not read(b2_verified)["pass"]:options=[o for o in options if o["bs"]==1]
    # All engineering smokes precede the first retained run so a memory fallback applies to every group.
    if (OUT/"training_profile.json").exists():
        frozen=read(OUT/"training_profile.json")
        if frozen!=config:raise ValueError("Frozen shared profile changed")
    else:
        for option in options:
            save(OUT/"selected_config.json",option)
            memory_failure=False
            for name in models:
                try:
                    exercise(name,"smoke",until=1)
                    prog=exercise(name,"smoke")
                    if prog["min_free_bytes"]<1024**3:raise RuntimeError("memory headroom below 1 GiB")
                    passed(name)
                except Exception as e:
                    fail(name,e)
                    if "out of memory" in str(e).lower() or "memory headroom" in str(e):
                        memory_failure=True;break
            if not memory_failure:
                save(OUT/"training_profile.json",option);break
        else:raise RuntimeError("No common safe tested configuration for screening")
    for name in models:
        if not (EXP/"outputs/fast_screening_v1/smoke"/read(OUT/"selected_config.json")["id"]/name/"complete.json").exists():continue
        try:exercise(name,"screening");passed(name)
        except Exception as e:fail(name,e)
        render()
    gates=render()
    eligible=all(g["advance"] for g in gates[:3])
    save(OUT/"joint_decision.json",dict(run=eligible,dependencies=gates[:3],utc=utc()))
    if eligible:
        try:exercise("joint","smoke",until=1);exercise("joint","smoke");exercise("joint","screening");passed("joint")
        except Exception as e:fail("joint",e)
    render()
    final="BLOCKED" if failures else "COMPLETE"
    state(status=final,stage="screening_report",failures=failures,report="reports/fast_screening/FAST_SCREENING_REPORT.md")
    (EXP/"reports/handoff.md").write_text("快速筛选状态 "+final+"。见 reports/fast_screening/FAST_SCREENING_REPORT.md 及失败证据。extended/full 不会自动启动。\n")
