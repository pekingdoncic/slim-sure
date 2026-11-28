from collections import defaultdict

import cv2
import numpy as np
import torch
from utils.metrics import estimate_pose, relative_pose_error, symmetric_epipolar_distance


def auc(values, thresholds=(5,10,20)):
    if not len(values):
        raise ValueError("empty evaluation")
    errors=np.r_[0.,np.sort([np.inf if e is None else e for e in values])]
    recall=np.arange(len(errors))/(len(errors)-1)
    result={}
    for t in thresholds:
        idx=np.searchsorted(errors,t)
        result[f"auc@{t}"]=float(100*np.trapz(np.r_[recall[:idx],recall[idx-1]],np.r_[errors[:idx],t])/t)
    return result


def coverage(points, wh):
    w,h=wh
    cells=np.floor(points/32).astype(int)
    ok=((points>=0)&(points<np.array([w,h]))).all(axis=1)
    return len(np.unique(cells[ok],axis=0))/(np.ceil(w/32)*np.ceil(h/32))


def evaluate_pair(row, pts0, pts1):
    p,q=map(lambda x:np.asarray(x,dtype=np.float64).reshape(-1,2),(pts0,pts1))
    if len(p)!=len(q) or not (np.isfinite(p).all() and np.isfinite(q).all()):
        raise ValueError("invalid matcher coordinates")
    k,l,t=map(np.asarray,(row["K0"],row["K1"],row["T_0to1"]))
    tx,ty,tz=t[:3,3]
    e=np.array([[0.,-tz,ty],[tz,0.,-tx],[-ty,tx,0.]])@t[:3,:3]
    epi=symmetric_epipolar_distance(*map(torch.from_numpy,(p,q,e,k,l))).numpy()
    cv2.setRNGSeed(66)
    try:
        pose=estimate_pose(p,q,k,l,thresh=0.5,conf=0.99999)
    except cv2.error:
        pose=None
    err_t=err_r=err=None
    n=0
    if pose is not None:
        r,v,mask=pose
        a,b=relative_pose_error(t,r,v)
        if np.isfinite([a,b]).all():
            err_t,err_r,err=float(a),float(b),float(max(a,b))
            n=int(mask.sum())
    return dict(pair_id=row["pair_id"],scene=row["scene"],rotation_deg=err_r,translation_deg=err_t,pose_deg=err,
                pose_recovered=err is not None,success_at_10=err is not None and err<10,
                matches=len(p),inliers=n,inlier_ratio=n/len(p) if len(p) else 0.,
                epi_correct=int((epi<1e-4).sum()),epi_precision=float((epi<1e-4).mean()) if len(p) else None,
                coverage0=float(coverage(p,row["wh0"])),coverage1=float(coverage(q,row["wh1"])))


def aggregate(rows):
    def one(items):
        result=auc([r["pose_deg"] for r in items])
        result["pairs"]=len(items)
        result["failed"]=sum(r["pose_deg"] is None for r in items)
        for key in ("success_at_10","pose_recovered","matches","inliers","inlier_ratio","coverage0","coverage1",
                    "rotation_deg","translation_deg","epi_precision","matcher_ms","peak_allocated_bytes","peak_reserved_bytes"):
            v=[r[key] for r in items if r.get(key) is not None]
            result[key+"_mean"]=float(np.mean(v)) if v else None
        result["epi_precision_pooled"]=sum(r["epi_correct"] for r in items)/sum(r["matches"] for r in items) if sum(r["matches"] for r in items) else None
        return result
    groups=defaultdict(list)
    for row in rows:
        groups[row["scene"]].append(row)
    per_scene={s:one(group) for s,group in groups.items()}
    summary=one(rows)
    equal={}
    for k in summary:
        if k not in ("pairs","failed"):
            v=[r[k] for r in per_scene.values() if r[k] is not None]
            equal[k]=float(np.mean(v)) if v else None
    return dict(global_metrics=summary,scene_equal=equal,per_scene=per_scene)
