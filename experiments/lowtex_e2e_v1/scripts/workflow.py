"""Local v3 supervisor. Resume committed training and require evidence before advancing."""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

EXP=Path(__file__).resolve().parents[1]
ROOT=EXP.parents[1]
REPORT=EXP/"reports"
STATE=REPORT/"state.json"
BUDGET=EXP/"manifests/full_budget.json"
ENTRY=EXP/"scripts/run_all.sh"


def read(path):
    return json.loads(path.read_text())


def replace(path,text):
    tmp=path.with_name(path.name+f".{os.getpid()}.tmp")
    tmp.write_text(text)
    tmp.replace(path)


def save(path,data):
    replace(path,json.dumps(data,ensure_ascii=False,indent=2)+"\n")


def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1048576),b""):h.update(block)
    return h.hexdigest()


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def run_dir():
    receipt=REPORT/"full_run.json"
    if not receipt.exists():return None
    info=read(receipt)
    d=(EXP/info["directory"]).resolve()
    if EXP not in d.parents or sha(d/"run_manifest.json")!=info["identity"]:
        raise ValueError("invalid run identity")
    return d


def track(pid,status,reason=None):
    state=read(STATE)
    run=run_dir()
    row=state["experiments"]["exp0"]
    progress={}
    if run and (run/"progress.json").is_file():
        progress=read(run/"progress.json")
        row.update(directory=str(run.relative_to(EXP)),checkpoint=str((run/"checkpoint_last.pt").relative_to(EXP)),
            completed_optimizer_updates=progress["step"],cursor=progress["cursor"],
            unique_ordered_rows=progress["unique_rows"],unique_unordered=progress.get("unique_unordered"),coverage=progress["coverage"])
    row["full_training"]=status
    state["phases"]["02"]["full_training"]=status
    state["phases"]["02"]["blocker"]=reason
    state["gate_status"]="BLOCKED" if reason else "RUNNING"
    row["updated_utc"]=now()
    save(STATE,state)
    save(REPORT/"workflow_status.json",dict(supervisor_pid=os.getpid(),worker_pid=pid,updated=now(),stage="02",status=status,
                                          run=str(run) if run else None,reason=reason,checkpoint_updates=progress.get("step")))
    handoff=f"""# 当前进度

Phase 02。Exp-0 full 状态 {status}，已提交 {row.get('completed_optimizer_updates',0)} / {row['planned_optimizer_updates']} 个 optimizer updates。最新 run 为 {run}。逐对训练访问、checkpoint、采样游标与两路开发验证在该目录。官方权重的开发结果与两次 smoke 见 phase_02.md。最终测试保持封存。

WSL 仓库根目录执行 bash experiments/lowtex_e2e_v1/scripts/run_workflow.sh 可重新取得锁并从最新完整 checkpoint 续跑。正在执行的同一工作流持有锁时，第二次启动只返回状态。运行环境中使用现有 slim 解释器，终端前台运行到退出。\n"""
    if reason:handoff+="\n实际阻塞："+reason+"\n"
    replace(REPORT/"handoff.md",handoff)


def baseline():
    budget=read(BUDGET)
    run=run_dir()
    cmd=["bash",str(ENTRY),"exp0","--mode","full","--resolution",str(budget["resolution"]),"--seed","66"]
    if run and (run/"checkpoint_last.pt").exists():cmd+=["--resume","--checkpoint",str(run/"checkpoint_last.pt")]
    track(None,"RUNNING")
    log_dir=EXP/"logs/workflow"
    log_dir.mkdir(parents=True,exist_ok=True)
    path=log_dir/(datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")+".log")
    with path.open("x",buffering=1) as log:
        print(json.dumps(cmd),file=log,flush=True)
        p=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        while True:
            track(p.pid,"RUNNING")
            try:
                rc=p.wait(timeout=45)
                break
            except subprocess.TimeoutExpired:pass
    path.with_suffix(".exitcode").write_text(str(rc)+"\n")
    if rc:
        tail=path.read_text(errors="replace")[-1300:]
        track(None,"BLOCKED",f"exit {rc}; {path.relative_to(EXP)}\n{tail}")
        raise SystemExit(rc)
    run=run_dir()
    result=read(run/"summary.json")
    progress=read(run/"progress.json")
    last=run/"checkpoint_last.pt"
    if (result["status"]!="PASS" or result["step"]!=budget["optimizer_updates"] or progress["cursor"]!=budget["micro_batches"]
        or progress["unique_rows"]!=budget["ordered_rows"] or progress["unique_unordered"]!=budget["unordered_unique"]
        or not (run/"validation"/f"update_{result['step']:08d}"/"accepted.json").exists()):
        track(None,"BLOCKED","Incomplete coverage, budget or final development receipt")
        raise SystemExit(2)
    track(None,"PASS")
    state=read(STATE)
    receipt=dict(full_run=str(run.relative_to(EXP)),checkpoint_sha256=sha(last),result=result,coverage=progress["coverage"])
    save(REPORT/"exp0_completion.json",receipt)
    state["experiments"]["exp0"]["engineering"]="PASS"
    state["phases"]["02"].update(engineering="PASS",science="NOT_EVALUATED")
    state.update(current_phase="03",next_prompt="03_texture_loss/prompt.md",gate_status="READY")
    save(STATE,state)
    with (REPORT/"phase_02.md").open("a") as f:
        f.write("\n完整预算及两个覆盖分母已通过，真实结果在 "+str(run.relative_to(EXP))+"，最终权重 SHA-256 "+receipt["checkpoint_sha256"]+"。\n")
    replace(REPORT/"handoff.md","Phase 02 已通过。从 prompts/03_texture_loss/prompt.md 继续，读取公共协议和 reports/exp0_completion.json。\n")


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--status",action="store_true")
    parser.add_argument("--phase02-only",action="store_true")
    args=parser.parse_args()
    if args.status:
        print(json.dumps(read(STATE),ensure_ascii=False,indent=2))
        return
    lock=(REPORT/"workflow.lock").open("a")
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        print("Workflow already active",flush=True);return
    state=read(STATE)
    if state["authorized_through_batch"]!=4 or state["budget"]!="full":raise ValueError("authorization identity changed")
    if state["current_phase"]=="02":baseline()
    if not args.phase02_only:
        print("The Windows stage supervisor is launched by scripts/run_workflow.sh",flush=True)


if __name__=="__main__":
    try:main()
    except Exception as e:
        save(REPORT/"workflow_error.json",dict(time=now(),error=repr(e),phase=read(STATE)["current_phase"]))
        raise
