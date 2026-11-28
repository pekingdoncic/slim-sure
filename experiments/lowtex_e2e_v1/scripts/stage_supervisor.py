"""Run in the Linux experiment directory. Launch each phase under the verified Windows CLI approval policy."""
import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

from workflow import EXP,ROOT,REPORT,STATE,now,read,save


def status(phase,action,detail):
    save(REPORT/"host_status.json",dict(pid=os.getpid(),phase=phase,action=action,detail=detail,utc=now()))


def worker_active():
    path=REPORT/"workflow_status.json"
    if not path.is_file():
        return False
    w=read(path)
    if w["status"]!="RUNNING":
        return False
    age=time.time()-datetime.datetime.fromisoformat(w["updated"]).timestamp()
    return age<150 and Path("/proc",str(w["supervisor_pid"])).is_dir()


def stage_session(phase,config):
    executable=subprocess.check_output(["wslpath","-u",config["cli"]],text=True).strip()
    if not Path(executable).is_file():
        raise RuntimeError("Verified local CLI executable is unavailable")
    prompt=f"""Execute current Phase {phase} of the authorized v3 research workflow in Ubuntu WSL {ROOT}. Windows cwd is only a trusted non-Git process host. Request normal narrow WSL automatic approval review when required. All file changes belong in {EXP}. First read the complete prompts/START_HERE.md and the three common protocols, reports/state.json, reports/handoff.md and only the current phase prompt. Follow those documents and the original all-batches, full-budget authorization. Preserve original code/data/weights and earlier runs. No large downloads, system alterations, paid compute, or final tests before Phase 08 freeze. For Phase 02 recovery, preserve the in-progress run, log the error and verify restored coverage and budget. Do not replace a full run by a smoke or shrink the legal pair pool. Execute real CLI, training/validation, checkpointing and acceptance, preserve per-stage details and real negative outcomes. Continue independent tasks when a specific dependency fails. The stage supervisor owns later phase sessions. Do not launch another agent or recursively call the whole workflow. If current-phase training has a durable long-running worker, you may hand it to this supervisor by atomically recording reports/phase_worker.json with phase, pid, Linux boot_id, start_ticks from /proc/PID/stat field 22, the path to an experiment-local status/log, and a complete resume command. The supervisor will wait for that exact worker and return to the current stage for acceptance. Otherwise finish the current stage or record an evidenced blocker. Advance state only after engineering/full/scientific dispositions and preserve the frozen model gate. Respond in Chinese."""
    log=EXP/"logs/workflow"/f"phase_{phase}_{time.time_ns()}.log"
    status(phase,"STAGE_SESSION",str(log))
    with log.open("x",buffering=1) as stream:
        # WSL interop uses a live stdin pipe for Windows native commands.
        proc=subprocess.Popen([executable,*config["verified_argv"]],stdin=subprocess.PIPE,stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT,cwd=EXP)
        proc.stdin.write((prompt+"\n").encode("utf-8"));proc.stdin.close()
        for line in proc.stdout:
            stream.write(line.decode("utf-8",errors="replace"))
        proc.wait()
    log.with_suffix(".exitcode").write_text(str(proc.returncode)+"\n")
    return proc.returncode,log


def delegated_worker(phase):
    receipt=REPORT/"phase_worker.json"
    if not receipt.exists():return None
    r=read(receipt)
    if str(r["phase"])!=phase or r["boot_id"]!=Path("/proc/sys/kernel/random/boot_id").read_text().strip():return None
    proc=Path("/proc",str(int(r["pid"])),"stat")
    try:fields=proc.read_text().rsplit(")",1)[1].split()
    except FileNotFoundError:return None
    if fields[0]=="Z" or int(fields[19])!=int(r["start_ticks"]):return None
    evidence=(EXP/r["evidence"]).resolve()
    if EXP not in evidence.parents or not evidence.exists() or not r["resume_command"]:
        raise RuntimeError("Active stage worker is missing its evidence or recovery command")
    return r


def monitor():
    lock=(REPORT/"host_workflow.lock").open("a")
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        print("Stage supervisor already active",flush=True);return
    cfg=read(REPORT/"host_config.json")
    receipt=(EXP/cfg["receipt"]).read_text(encoding="utf-8-sig").splitlines()
    if len(receipt)!=2 or receipt[1]!=read(STATE)["protocol_sha256"]:
        raise RuntimeError("CLI workspace probe identity mismatch")
    prior=None
    failures=0
    while True:
        state=read(STATE);phase=state["current_phase"]
        if state["authorized_through_batch"]!=4 or state["budget"]!="full":
            raise RuntimeError("Authorization record changed")
        if state["gate_status"]=="COMPLETE":
            status(phase,"COMPLETE","See Phase 08 report and frozen test artifacts");return
        if phase!=prior:failures=0;prior=phase
        if phase=="02" and state["phases"]["02"]["full_training"]!="BLOCKED":
            if worker_active():
                w=read(REPORT/"workflow_status.json")
                status(phase,"MONITORING_FULL",dict(checkpoint_updates=w["checkpoint_updates"],run=w["run"]))
                time.sleep(40)
                continue
            status(phase,"RESUMING_FULL","Acquire Linux worker lock and verify checkpoint")
            rc=subprocess.run(["bash",str(EXP/"scripts/run_workflow.sh"),"--phase02-worker"],cwd=ROOT).returncode
            if rc:failures+=1
            if failures>=2:raise RuntimeError(f"Repeated full worker exit {rc}")
            time.sleep(4)
            continue
        work=delegated_worker(phase)
        if work:
            status(phase,"MONITORING_STAGE_WORKER",work)
            time.sleep(45)
            continue
        rc,log=stage_session(phase,cfg)
        after=read(STATE)
        if rc==0 and (after["current_phase"]!=phase or after["gate_status"]=="COMPLETE"):
            completed=after["phases"][phase]
            if completed["engineering"] not in ("PASS","SKIPPED","BLOCKED") or completed["full_training"] not in ("PASS","SKIPPED","BLOCKED","NOT_APPLICABLE"):
                raise RuntimeError(f"Unresolved phase advanced; check {log}")
            if phase!="08" and after["final_test"]!="NOT_RUN":
                raise RuntimeError("Premature final test disposition")
            if not completed["report"] or not (EXP/completed["report"]).is_file():
                raise RuntimeError(f"Phase {phase} report missing")
            failures=0
        elif delegated_worker(phase):
            failures=0
        else:
            failures+=1
            if failures>=2 or after["gate_status"]=="BLOCKED":
                raise RuntimeError(f"Phase {phase} did not advance: exit {rc}, {log}")


if __name__=="__main__":
    try:monitor()
    except Exception as err:
        status(read(STATE)["current_phase"],"BLOCKED",repr(err))
        raise
