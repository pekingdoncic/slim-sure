import argparse, csv, datetime, json, os, statistics, subprocess, threading, time
import torch
from .data import OUT, prepare, loader, save
from .engine import update
from .model import components

class Monitor:
    def __init__(self):self.values=[];self.stop=threading.Event()
    def run(self):
        while not self.stop.is_set():
            p=subprocess.run(["nvidia-smi","-i","0","--query-gpu=utilization.gpu,memory.free","--format=csv,noheader,nounits"],capture_output=True,text=True)
            if p.returncode==0:
                try:self.values.append(tuple(map(int,p.stdout.strip().split(", "))))
                except ValueError:pass
            self.stop.wait(.5)

def main():
    parser=argparse.ArgumentParser()
    for k in ("bs","cp","workers","pin","persist","prefetch","steps"):parser.add_argument("--"+k,type=int,required=True)
    parser.add_argument("--id",required=True)
    a=parser.parse_args();prepare();torch.set_num_threads(2)
    p=OUT/"bench"/a.id
    p.mkdir(parents=True,exist_ok=True)
    args=vars(a)
    if (p/"result.json").exists():raise FileExistsError(p)
    warm=10
    dataset=loader(0,(warm+a.steps)*4,a.bs,a.workers,bool(a.pin),bool(a.persist),a.prefetch)
    it=iter(dataset)
    model,cfg,objective,policy=components("exp0",bool(a.cp),2215669)
    bn=model.bn_snapshot()
    rows=[];m=Monitor();thread=None
    try:
        for step in range(warm+a.steps):
            if step==warm:
                thread=threading.Thread(target=m.run,daemon=True);thread.start();start=time.perf_counter()
            v=update(it,4,a.bs,model,cfg,objective,policy,step,check_grad=step==0)
            if step>=warm:rows.append({k:val for k,val in v.items() if k!="samples"})
        elapsed=time.perf_counter()-start
        if not all(torch.equal(bn[k],v) for k,v in model.bn_snapshot().items()):raise RuntimeError("BN drift")
    finally:
        m.stop.set()
        if thread:thread.join()
    if not m.values:raise RuntimeError("GPU telemetry unavailable")
    save(p/"samples.json",{"updates":rows,"gpu":m.values})
    res=dict(**args,status="PASS",warmup=warm,pairs_s=4*a.steps/elapsed,step_s=elapsed/a.steps,
             gpu_util=statistics.mean(v[0] for v in m.values),allocated=max(v["allocated"] for v in rows),
             reserved=max(v["reserved"] for v in rows),min_free=min(min(v[1]<<20 for v in m.values),min(v["min_free_now"] for v in rows)),
             gradient_and_bn_verified=True)
    for k in ("forward_s","backward_s","load_s"):res[k]=statistics.mean(v[k] for v in rows)
    save(p/"result.json",res);print(json.dumps(res),flush=True)

if __name__=="__main__":main()
