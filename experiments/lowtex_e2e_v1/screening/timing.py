"""Isolated endpoint inference timing using held-out development pairs and no optimizer."""
import argparse, hashlib, statistics, time
import numpy as np
import torch
from .data import EXP,OUT,read,save,sha
from .model import components
from ..evaluation.development import texture_input
from configs.cfg_outdoor_test import get_cfg_defaults

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--run",required=True);p.add_argument("--smoke",action="store_true")
    a=p.parse_args()
    d=(EXP/a.run).resolve()
    if EXP not in d.parents:raise ValueError(d)
    complete=read(d/"complete.json")
    model,_,_,_=components(read(d/"identity.json")["exp"],False,2215669)
    assert sha(d/"checkpoint_last.pt")==complete["checkpoint_sha256"]
    model.load_state_dict(torch.load(d/"checkpoint_last.pt",map_location="cpu")["model"])
    model.eval();model.fine_match_thres=get_cfg_defaults().MODEL.FINE_THRES
    data=read(EXP/"manifests/texturepoor_val_pairs.json")
    scenes=sorted({r["scene"] for r in data})
    groups=[sorted((r for r in data if r["scene"]==scene),key=lambda r:hashlib.sha256(("timing66/"+r["pair_id"]).encode()).digest()) for scene in scenes]
    rows=[r for triple in zip(*groups) for r in triple][:3 if a.smoke else 100]
    original=model.refine_iters
    out=[]
    for side in (640,832):
        routes=[("native",original)]
        if read(d/"identity.json")["exp"] in ("exp0","rru_matched"):routes.append(("rru1",1))
        for label,it in routes:
            model.refine_iters=it;timings=[];refine=[];peaks=[]
            warm=2 if a.smoke else 30
            for seq in range(warm+len(rows)*(1 if a.smoke else 3)):
                rec=rows[seq%len(rows)]
                b=texture_input(rec,side);torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                start=time.perf_counter()
                with torch.autocast("cuda",dtype=torch.float16):model(b,training=False)
                torch.cuda.synchronize();elapsed=1000*(time.perf_counter()-start)
                if seq>=warm:
                    timings.append(elapsed);refine.append(1000*b["refine_time"])
                    peaks.append([torch.cuda.max_memory_allocated(),torch.cuda.max_memory_reserved()])
                model.pyramid=None
                del b
            out.append(dict(side=side,route=label,median_ms=statistics.median(timings),mean_ms=statistics.mean(timings),
                p95_ms=float(np.percentile(timings,95)),refine_ms=statistics.median(refine),allocated=max(x[0] for x in peaks),
                reserved=max(x[1] for x in peaks),samples=len(timings),distinct_pairs=len(rows),warmup=warm))
    save(d/"timing.json",dict(checkpoint=complete["checkpoint_sha256"],pairs=[r["pair_id"] for r in rows],results=out))
    print(out)
if __name__=="__main__":
    import subprocess,sys
    if "--smoke" in sys.argv:
        arg=sys.argv[sys.argv.index("--run")+1]
        subprocess.run([sys.executable,"-B","-m","experiments.lowtex_e2e_v1.screening.gradient_audit","--run",arg],check=True)
    main()
