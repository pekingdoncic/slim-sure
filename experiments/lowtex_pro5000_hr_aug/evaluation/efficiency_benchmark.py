from collections import defaultdict
from time import perf_counter
import numpy as np
import torch
from tqdm import tqdm
from .datasets import records,load_record
from .eval_hr import upload,matched_original
from .pose import pair_metrics,aggregate
from .lowtex_metrics import metrics
from ..data.texture_adapter import threshold
from ..utils.run_io import write,append


def distribution(x):
    return dict(mean=float(np.mean(x)),median=float(np.median(x)),p95=float(np.percentile(x,95)))


def flops_audit(model,d,precision):
    # PyTorch only defines FLOP formulas for a subset of operators. Unknowns are exported.
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],
                                with_flops=True) as prof,torch.autocast("cuda",dtype=getattr(torch,precision)):
        model(d)
    events = prof.key_averages()
    return dict(covered_flops=sum(e.flops or 0 for e in events),
                uncounted_operators=sorted(e.key for e in events if not e.flops),
                convention="Partial profiler count: multiply and add are two FLOPs; unsupported ops uncounted.")


@torch.no_grad()
def benchmark(model,args,side,directory,calibration,precision,guard,smoke):
    cfg = model.cfg
    model.eval()
    selected = records(args,"megadepth","dev")
    take = min(len(selected), 2 if smoke else cfg["benchmark_pairs"])
    selected = [selected[i] for i in np.linspace(0,len(selected)-1,take,dtype=int)]
    warm,repeats = (1,1) if smoke else (cfg["benchmark_warmup"],cfg["benchmark_repeats"])
    timing = defaultdict(list)
    rows = []
    peak_a=peak_r=0.
    units = []
    poses = {}
    for ix in range(warm):
        d,_ = load_record(args,"megadepth","dev",selected[ix%take],side,point_gt=False)
        with torch.autocast("cuda",dtype=getattr(torch,precision)):
            model(upload(d))
        guard.check()
    torch.cuda.synchronize()
    guard.reset_observations()
    for r in tqdm(selected,desc=f"benchmark {cfg['model']} R{side}"):
        for repeat in range(repeats):
            guard.check()
            torch.cuda.reset_peak_memory_stats()
            start = perf_counter()
            d,meta = load_record(args,"megadepth","dev",r,side,point_gt=False)
            pre = (perf_counter()-start)*1000
            gpu = upload({k:v for k,v in d.items() if k.startswith(("image","mask","scale","valid_mask"))},True)
            torch.cuda.synchronize()
            transfer = perf_counter()
            stages = {}
            with torch.autocast("cuda",dtype=getattr(torch,precision)):
                model(gpu,diagnostics=False,times=stages)
            torch.cuda.synchronize()
            stop = perf_counter()
            p,q = matched_original(gpu,meta["plans"])
            result = pair_metrics(meta,p,q)
            wall = (perf_counter()-start)*1000
            stages.update(Total=(stop-transfer)*1000,read=meta["io_ms"],preprocess=max(0.,pre-meta["io_ms"]),
                          h2d=(transfer-start)*1000-pre,ransac=result["ransac_ms"],pipeline=wall)
            peak_a=max(peak_a,torch.cuda.max_memory_allocated()/1024**3)
            peak_r=max(peak_r,torch.cuda.max_memory_reserved()/1024**3)
            for k,v in stages.items():
                timing[k].append(v)
            units.append(dict(pair_id=r["pair_id"],repeat=repeat,timing_ms=stages))
            del gpu
        poses[r["pair_id"]] = result
    guard.poll()
    measured_driver=guard.snapshot()
    # Depth I/O, ranking diagnostics and profiler instrumentation have their own untimed pass.
    for r in tqdm(selected,desc="untimed point-GT diagnostics"):
        d,meta=load_record(args,"megadepth","dev",r,side)
        gpu = upload(d)
        result=poses[r["pair_id"]]
        with torch.autocast("cuda",dtype=getattr(torch,precision)):
            model(gpu,diagnostics=True)
        result.update(metrics(gpu,model,threshold(calibration,side),meta["plans"][1]))
        result["edges"] = (len(gpu["keys"]) if "keys" in gpu else
                           len(gpu["candidate_ids0"][0])*len(gpu["candidate_ids1"][0]))
        rows.append(result)
        del gpu
        guard.check()
    try:
        count = flops_audit(model,upload(d),precision)
    except (RuntimeError,NotImplementedError) as exc:
        if isinstance(exc,torch.OutOfMemoryError):
            raise
        count = dict(covered_flops=None,uncounted_operators=[],profiler_error=str(exc))
    out = dict(status="PASS",model=cfg["model"],side=side,run_id=args.run_id,precision=precision,batch=1,
               warmup=warm,repeats=repeats,params=sum(p.numel() for p in model.parameters()),
               new_params=sum(p.numel() for n,p in model.named_parameters() if not n.startswith("core.")),
               **count,flops_pair_id=selected[-1]["pair_id"],timing_ms={k:distribution(x) for k,x in timing.items()},peak_allocated_gib=peak_a,
               peak_reserved_gib=peak_r,driver=measured_driver,cohort="fixed equally spaced MegaDepth development rows",
               environment="shared GPU, synchronized stage boundaries",**aggregate(rows))
    write(directory/"benchmark.json",out)
    write(directory/"samples.json",units)
    return out
