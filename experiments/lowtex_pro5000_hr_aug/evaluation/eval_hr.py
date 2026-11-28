from pathlib import Path
from time import perf_counter
import numpy as np
import torch
from tqdm import tqdm
from .datasets import records,load_record
from .pose import aggregate,pair_metrics
from .lowtex_metrics import metrics
from ..data.texture_adapter import threshold
from ..utils.run_io import atomic_path,write,read
import json


def upload(d,pin=False):
    return {k:(v.pin_memory() if pin else v)[None].to("cuda",non_blocking=pin) for k,v in d.items() if isinstance(v,torch.Tensor)}


def matched_original(d,plans,return_ids=False):
    points=[d[f"fine_coord_{i}"].float().cpu().numpy() for i in (0,1)]
    live=np.ones(len(points[0]),dtype=bool)
    for p,plan in zip(points,plans):
        if not np.isfinite(p).all():
            raise FloatingPointError("nonfinite fine match")
        live &= ((p>=0)&(p<np.array([plan.rw,plan.rh]))).all(1)
    original=[plan.to_original(p[live]) for p,plan in zip(points,plans)]
    return (*original,np.flatnonzero(live)) if return_ids else original


def capped_indices(d,live_ids):
    if len(live_ids)<=1000:
        return np.arange(len(live_ids))
    ids=torch.as_tensor(live_ids,device=d["m_idx_it"].device)
    m,i,j=[d[key][ids] for key in ("m_idx_it","i_idx_it","j_idx_it")]
    sim=d["sim_matrix_f"]
    ar=torch.arange(len(ids),device=ids.device)
    row=sim[m,i,:].float().softmax(1)[ar,j]
    col=sim[m,:,j].float().softmax(1)[ar,i]
    return (row*col).argsort(descending=True,stable=True)[:1000].cpu().numpy()


@torch.no_grad()
def evaluate(model,args,side,domain,split,directory,calibration,precision,guard,limit=None):
    old = model.training
    model.eval()
    result = directory/f"{domain}.json"
    try:
        rows = records(args,domain,split)
        if limit:
            rows = rows[:limit]
        if not rows:
            raise ValueError(f"empty {domain} split")
        point_gt = domain=="megadepth" and split!="test"
        mechanism = "PASS" if point_gt else "NOT_APPLICABLE"
        try:
            thr = threshold(calibration,side) if point_gt else None
        except FileNotFoundError:
            point_gt,thr,mechanism = False,None,"BLOCKED_MISSING_TRAIN_CALIBRATION"
        output,capped = [],[]
        with atomic_path(directory/f"{domain}_pairs.jsonl") as tmp, tmp.open("w") as fp:
            for r in tqdm(rows,desc=f"{model.cfg['model']} R{side} {domain} {split}",mininterval=2):
                guard.check()
                d,meta = load_record(args,domain,split,r,side,point_gt=point_gt)
                d = upload(d)
                with torch.autocast("cuda",dtype=getattr(torch,precision)):
                    model(d,diagnostics=point_gt)
                p,q,live_ids = matched_original(d,meta["plans"],return_ids=True)
                row = pair_metrics(meta,p,q)
                if len(p)>1000:
                    selected=capped_indices(d,live_ids)
                    subset=pair_metrics(meta,p[selected],q[selected])
                else:
                    subset=dict(row)
                capped.append(subset)
                row["capped_1000"]=subset
                row["physical_pair"] = json.dumps([r["scene"],meta["physical_pair"]])
                row["raw_matches"] = len(d["fine_coord_0"])
                if point_gt:
                    row.update(metrics(d,model,thr,meta["plans"][1]))
                if "keys" in d:
                    row["edges"] = d["keys"].numel()
                    row["routing"] = d["route_stats"]
                fp.write(json.dumps(row,allow_nan=False)+"\n")
                output.append(row)
                del d
        res = dict(status="PASS",domain=domain,split=split,side=side,model=model.cfg["model"],mechanism_status=mechanism,
                   run_id=args.run_id,unique_physical_pairs=len({r["physical_pair"] for r in output}),
                   repeated_rows=len(output)-len({r["physical_pair"] for r in output}),
                   capped_1000=aggregate(capped,domain=="hpatches"),**aggregate(output,domain=="hpatches"))
    except FileNotFoundError as exc:
        res = dict(status="BLOCKED",domain=domain,split=split,side=side,reason=str(exc),run_id=args.run_id)
    finally:
        model.train(old)
    write(result,res)
    print(f"{domain} R{side} {split}",res.get("scene_equal",res.get("reason")),flush=True)
    return res


def double_dev(model,args,side,directory,calibration,precision,guard,quick=False,probe=False):
    return {domain:evaluate(model,args,side,domain,"quick" if quick else "dev",directory,calibration,
                            precision,guard,1 if probe else None) for domain in ("megadepth","texturepoor")}
