"""Capture a new ordinary v3 checkpoint, then retire only processes belonging to that run."""
from pathlib import Path
import datetime, hashlib, json, os, shutil, signal, statistics, time
import numpy as np
import torch

E=Path(__file__).resolve().parents[1]
R=E/"reports"
A=R/"fast_screening"/"legacy_r832"
def read(p):return json.loads(p.read_text())
def save(p,j):
    temp=p.with_suffix(".json.tmp")
    temp.write_text(json.dumps(j,ensure_ascii=False,indent=2)+"\n")
    temp.replace(p)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()

def main():
    A.mkdir(parents=True,exist_ok=True)
    if (A/"transition.json").exists():return
    prev=read(R/"full_run.json")
    run=(E/prev["directory"]).resolve()
    assert E in run.parents and sha(run/"run_manifest.json")==prev["identity"]
    h=read(R/"host_status.json"); w=read(R/"workflow_status.json"); a=read(R/"development_archive.json")
    assert w["run"]==str(run) and h["phase"]=="02" and a["run"]==prev["directory"]
    for n in ("state.json","handoff.md","host_status.json","workflow_status.json","development_archive.json","full_run.json"):
        if not (A/n).exists():shutil.copy2(R/n,A/n)
    def cmd(pid):return Path("/proc",str(pid),"cmdline").read_bytes().split(b"\0")
    def children(pid):return list(map(int,Path(f"/proc/{pid}/task/{pid}/children").read_text().split()))
    training=[i for i in children(w["worker_pid"]) if b"experiments.lowtex_e2e_v1.training.train" in cmd(i)]
    assert len(training)==1 and w["worker_pid"] in children(w["supervisor_pid"])
    pid=training[0]
    owned=[(h["pid"],str(E/"scripts/stage_supervisor.py")),(w["supervisor_pid"],str(E/"scripts/workflow.py")),
           (a["pid"],str(E/"scripts/archive_development.py")),(pid,"experiments.lowtex_e2e_v1.training.train")]
    assert cmd(pid)[-8:-1]==[b"exp0",b"--mode",b"full",b"--resolution",b"832",b"--seed",b"66"]
    import ctypes
    libc=ctypes.CDLL(None,use_errno=True)
    handles={}
    for i,tag in owned:
        handles[i]=libc.pidfd_open(i,0)
        if handles[i]<0:raise OSError(ctypes.get_errno(),"pidfd_open")
        assert tag.encode() in cmd(i)
    def send(i,s):
        if libc.pidfd_send_signal(handles[i],int(s),None,0):raise OSError(ctypes.get_errno(),"pidfd_send_signal")
    start=read(run/"progress.json")["step"]
    print("awaiting normal checkpoint newer than",start,flush=True)
    while True:
        progress=read(run/"progress.json")
        if progress["step"]>start:break
        time.sleep(.04)
    send(pid,signal.SIGSTOP)
    try:
        checkpoint=A/f"checkpoint_{progress['step']}.pt"
        if not checkpoint.exists():os.link(run/"checkpoint_last.pt",checkpoint)
        saved=torch.load(checkpoint,map_location="cpu")
        assert saved["step"]==progress["step"] and saved["cursor"]==progress["cursor"] and saved["identity"]==prev["identity"]
        for k in ("model","optimizer","scheduler","rng","bn","unique","observed","physical_bits"):assert k in saved
        assert int(np.unpackbits(saved["physical_bits"]).sum())==progress["unique_unordered"]
        assert int(saved["unique"].sum())==progress["unique_rows"] and saved["scheduler"]["last_epoch"]==saved["step"]
        assert all(torch.equal(saved["model"]["feature_backbone."+k],v) for k,v in saved["bn"].items())
        vals={v.parent.name:{name:domain["scene_equal"] for name,domain in read(v).items()} for v in sorted((run/"validation").glob("*/accepted.json"))}
        rates=[];peaks=[0,0];invalid=0
        for f in sorted(run.glob("train_metrics_*.jsonl")):
            for l in f.open():
                z=json.loads(l)
                if z["step"]>saved["step"]:continue
                rates.append(z["seconds"])
                peaks=list(map(max,zip(peaks,z["peak_bytes"])))
                invalid+=sum(bool(m.get("skip_reason")) for m in z["micro"])
        verified=dict(time_utc=now(),checkpoint=str(checkpoint.relative_to(E)),sha256=sha(checkpoint),
                      lr=[g["lr"] for g in saved["optimizer"]["param_groups"]],progress=progress,validation=vals,
                      optimizer_updates_per_second=len(rates)/sum(rates),pairs_per_second=4*len(rates)/sum(rates),
                      peak_bytes=peaks,invalid_geometry_visits=invalid,committed_steps=saved["step"],processes=owned)
        save(A/"verified.json",verified)
    except BaseException:
        send(pid,signal.SIGCONT);raise
    for i,tag in owned[:-1]:send(i,signal.SIGTERM)
    send(pid,signal.SIGTERM);send(pid,signal.SIGCONT)
    import select
    wait=select.poll()
    for fd in handles.values():wait.register(fd,select.POLLIN)
    for fd in handles.values():
        if not select.select([fd],[],[],30)[0]:raise RuntimeError("Owned process did not exit")
    st=read(R/"state.json");item=st["experiments"]["exp0"]
    item.update(full_training="PAUSED_BY_USER_REVISION",completed_optimizer_updates=saved["step"],checkpoint=verified["checkpoint"],
                unique_ordered_rows=progress["unique_rows"],unique_unordered=progress["unique_unordered"],cursor=progress["cursor"],coverage=progress["coverage"])
    st["phases"]["02"]["full_training"]="PAUSED_BY_USER_REVISION"
    st.update(current_phase="fast_screening",gate_status="RUNNING",budget="screening",authorized_through_batch=0,
              full_auto_start=False,final_test_authorized=False,next_prompt=None,active_protocol="protocols/fast_screening_v1.md")
    st["authorization_history"].append(dict(at=now(),scope="RTX4080 screening; no automatic extended/full; 1D and 2A–2E deferred"))
    save(R/"state.json",st)
    save(R/"workflow_status.json",dict(status="SUPERSEDED",run=str(run),checkpoint_updates=saved["step"],evidence=str(A.relative_to(E))))
    save(R/"host_status.json",dict(action="V3_RETIRED",utc=now(),evidence=str(A.relative_to(E))))
    (R/"handoff.md").write_text(f"Fast screening v1。R832 在正常写入第 {saved['step']} 步 checkpoint 后暂停。恢复材料在 reports/fast_screening/legacy_r832。当前筛选见 reports/fast_screening/state.json；入口 scripts/run_workflow.sh。\n")
    save(A/"transition.json",dict(ended_utc=now(),verified=verified["checkpoint"],step=saved["step"],pidfd_exit_verified=True))
    print("verified and retired",saved["step"],flush=True)

if __name__=="__main__":main()
