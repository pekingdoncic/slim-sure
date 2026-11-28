import json
from pathlib import Path
import time

import numpy as np
import torch
from configs.cfg_outdoor_test import get_cfg_defaults as test_config
from reliability_data import project_visible
from diagnostics_coarse_rank.common import grid_contract

from .texturepoor_sfm.dataset import EXP,preprocess,read_image
from .texturepoor_sfm.metrics import aggregate,evaluate_pair
from .texturepoor_sfm.visualize import draw
from ..models.slim_wrapper import THRESHOLD
from ..training.data import load_pair


def texture_input(row,side):
    batch={}
    for i in (0,1):
        s=str(i)
        gray=read_image(row["image"+s])
        image,mask,scale=preprocess(gray,side)
        batch["image"+s]=torch.from_numpy(image)[None,None].cuda()
        batch["valid_mask"+s]=torch.from_numpy(mask)[None].cuda()
        batch["mask"+s]=batch["valid_mask"+s][:,::8,::8]
        batch["scale"+s]=torch.from_numpy(scale)[None].cuda()
    return batch


def low_metrics(batch,p,q,temperature):
    contract=grid_contract(batch,THRESHOLD)
    source=project_visible(p/batch["scale0"][0].cpu().numpy(),batch)
    valid=source["valid_gt"]
    error=np.linalg.norm(q-source["points1_gt_orig"],axis=1)
    parents=batch["i_idx_c"].cpu().numpy()
    targets=batch["j_idx_c"].cpu().numpy()
    child=parents[batch["m_idx_it"].cpu().numpy()]
    correct_parents=parents[targets==contract["correct"][parents]]
    inpool=np.isin(np.arange(len(contract["den"])),batch["candidate_ids0"][0].cpu().numpy()) & np.isin(contract["correct"],batch["candidate_ids1"][0].cpu().numpy())
    candidate0,candidate1=(batch["candidate_ids"+str(i)][0] for i in (0,1))
    ranked=np.flatnonzero(contract["den"] & inpool)
    ranks=np.zeros(len(contract["den"]),np.int32)
    if len(ranked):
        with torch.autocast("cuda",dtype=torch.float16):
            f0,f1=(batch["feat"+str(i)+"_c"].flatten(2).transpose(1,2)[0] for i in (0,1))
            dim=f0.shape[-1]**0.5
            sim=torch.einsum("lc,sc->ls",(f0/dim)[candidate0].contiguous(),(f1/dim)[candidate1].contiguous())/(temperature+1e-8)
            conf=sim.softmax(0)*sim.softmax(1)
        # Candidate IDs are not assumed sorted. Use the original raw target grid ID for ties.
        rev0={v:i for i,v in enumerate(candidate0.tolist())}
        rev1={v:i for i,v in enumerate(candidate1.tolist())}
        for ids in np.array_split(ranked,(len(ranked)+63)//64):
            rows=torch.tensor([rev0[i] for i in ids],device=conf.device)
            raw_gt=contract["correct"][ids]
            cols=torch.tensor([rev1[i] for i in raw_gt],device=conf.device)
            values=conf[rows,cols,None]
            alternatives=conf[rows]
            ranks[ids]=(1+((alternatives>values)|((alternatives==values)&(candidate1[None]<candidate1[cols,None]))).sum(1)).cpu().numpy()
    result={}
    for name,texture in (("lt",contract["grid"][0]["low"]),("other",~contract["grid"][0]["low"])):
        den=texture & contract["den"]
        valid_child=texture[child]&valid
        source_ids=np.flatnonzero(den)
        rescued=np.intersect1d(source_ids,child[valid & (error<2)])
        count=int(den.sum())
        result[name]=dict(den=count,success=len(rescued),recall=len(rescued)/count if count else None,
                          both_in_pool=int((den&inpool).sum()),
                          rank_histogram={str(k):int((den&(ranks>0)&(ranks<=k)).sum()) for k in (1,3,5,10)},
                          mean_rank_conditional=float(ranks[den&inpool].mean()) if (den&inpool).any() else None,
                          strict_coarse=len(np.intersect1d(source_ids,correct_parents)),
                          covered=len(np.intersect1d(source_ids,child[valid])),valid_children=int(valid_child.sum()),
                          epe=float(error[valid_child].mean()) if valid_child.any() else None,
                          precision=float((error[valid_child]<2).mean()) if valid_child.any() else None)
    return result


@torch.no_grad()
def validate(model,side,base,output,smoke=False,domains=("texturepoor","megadepth")):
    output.mkdir(parents=True,exist_ok=False)
    prev=model.training
    capture=model.capture_reliability
    thr=model.fine_match_thres
    model.eval()
    model.fine_match_thres=test_config().MODEL.FINE_THRES
    summaries={}
    try:
        for domain in domains:
            name=("texturepoor_smoke_val" if smoke else "texturepoor_val_pairs") if domain=="texturepoor" else ("megadepth_smoke_val" if smoke else "megadepth_dev_pairs")
            records=json.loads((EXP/"manifests"/(name+".json")).read_text())
            model.capture_reliability=domain=="megadepth"
            rows=[]
            pictures=set()
            with (output/(domain+".jsonl")).open("x") as f:
                for row in records:
                    if domain=="texturepoor":
                        batch=texture_input(row,side)
                        record=row
                    else:
                        batch,wh=load_pair(base,Path(row["index"]).stem,row["frame_indices"],side,"cuda",row["pair_id"])
                        record={**row,**{k:batch["_gt"][k] for k in ("K0","K1","T_0to1")},"wh0":wh[0],"wh1":wh[1]}
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    start=time.perf_counter()
                    with torch.autocast("cuda",dtype=torch.float16):
                        model(batch,training=False)
                    torch.cuda.synchronize()
                    ms=(time.perf_counter()-start)*1000
                    memory=[torch.cuda.max_memory_allocated(),torch.cuda.max_memory_reserved()]
                    p,q=(batch["fine_coord_"+str(i)].float().cpu().numpy() for i in (0,1))
                    result=evaluate_pair(record,p,q)
                    result.update(matcher_ms=ms,peak_allocated_bytes=memory[0],peak_reserved_bytes=memory[1],
                                  module_times_raw={k:float(batch[k]) for k in ("feat_extract_time","upsample_time","coarse_match_time","fine_match_time","refine_time")})
                    if domain=="megadepth":
                        result.update(low_metrics(batch,p,q,model.coarse_temperature.detach()))
                    else:
                        key=(row["scene"],result["pose_recovered"])
                        if key not in pictures:
                            draw(row,dict(xy0=p,xy1=q),output/(row["pair_id"].replace("/","_")+".jpg"))
                            pictures.add(key)
                    f.write(json.dumps(result,allow_nan=False)+"\n");f.flush()
                    rows.append(result)
                    del batch
            summaries[domain]=aggregate(rows)
            if domain=="megadepth":
                for group in ("lt","other"):
                    byscene=[]
                    for scene in sorted({r["scene"] for r in rows}):
                        v=[r[group]["recall"] for r in rows if r["scene"]==scene and r[group]["recall"] is not None]
                        if v:
                            byscene.append(float(np.mean(v)))
                    summaries[domain]["scene_equal"][group+"_recall"]=float(np.mean(byscene)) if byscene else None
            (output/(domain+"_summary.json")).write_text(json.dumps(summaries[domain],indent=2,allow_nan=False))
            print(domain,len(rows),summaries[domain]["scene_equal"],flush=True)
    finally:
        model.capture_reliability=capture
        model.fine_match_thres=thr
        model.train(prev)
    return summaries
