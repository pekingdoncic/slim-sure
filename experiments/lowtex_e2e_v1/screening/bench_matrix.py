import csv, json, os, subprocess, sys
from pathlib import Path
from .data import EXP,OUT,read,save

def matrix():
    rows=[]
    def run(name,bs,cp,workers=2,pin=1,persist=1,prefetch=2,steps=50):
        d=OUT/"bench"/name
        d.mkdir(parents=True,exist_ok=True)
        p=d/"result.json"
        cfg=dict(bs=bs,cp=cp,workers=workers,pin=pin,persist=persist,prefetch=prefetch,steps=steps)
        if not p.exists():
            cmd=["bash",str(EXP/"scripts/screen_python.sh"),"-m","experiments.lowtex_e2e_v1.screening.bench","--id",name]
            for k,v in cfg.items():cmd.extend(("--"+k,str(v)))
            save(OUT/"state.json",dict(status="RUNNING",stage="throughput",case=name,pid=os.getpid(),command=cmd))
            with (d/"cli.log").open("a") as log:
                print(cmd,file=log,flush=True)
                rc=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT).returncode
            (d/"exitcode").write_text(str(rc)+"\n")
            if rc:
                with (d/"cli.log").open() as f: reason=f.read()[-3000:]
                save(p,dict(**cfg,id=name,status="OOM" if "out of memory" in reason.lower() else "FAIL",error=reason))
        row=read(p);rows.append(row)
        fields=["id","status","bs","cp","workers","pin","persist","prefetch","steps","pairs_s","step_s","load_s","forward_s","backward_s","gpu_util","allocated","reserved","min_free","error"]
        dest=OUT/"TRAINING_THROUGHPUT.csv";tmp=dest.with_suffix(".tmp")
        with tmp.open("w",newline="") as out:
            w=csv.DictWriter(out,fields,extrasaction="ignore");w.writeheader();w.writerows(rows)
        tmp.replace(dest)
        return row
    def best(cases):
        candidates=[r for r in cases if r.get("status")=="PASS" and r["min_free"]>=1024**3]
        if not candidates:raise RuntimeError("No memory-safe measured candidate")
        return max(candidates,key=lambda r:r["pairs_s"])
    base=[run(f"b{bs}_cp{cp}",bs,cp,steps=100 if bs==1 and cp==1 else 50) for bs,cp in ((1,1),(1,0),(2,1),(2,0))]
    winner=best(base)
    alts=[winner]
    for workers,pin,persist,prefetch in [(4,1,1,2),(8,1,1,2),(2,0,1,2),(2,1,0,2),(2,1,1,1)]:
        alts.append(run(f"loader_w{workers}_pin{pin}_persist{persist}_pf{prefetch}",winner["bs"],winner["cp"],workers,pin,persist,prefetch))
    for selected in sorted([r for r in alts if r.get("status")=="PASS" and r["min_free"]>=1024**3],key=lambda r:-r["pairs_s"]):
        confirm=run("confirm_"+selected["id"],*[selected[k] for k in ("bs","cp","workers","pin","persist","prefetch")],steps=100)
        if confirm.get("status")=="PASS" and confirm["min_free"]>=1024**3:
            save(OUT/"selected_config.json",confirm);return
    raise RuntimeError("No confirmed throughput configuration")

if __name__=="__main__":matrix()
