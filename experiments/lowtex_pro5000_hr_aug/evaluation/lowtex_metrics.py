"""Fixed-denominator grid outcomes. Ranks use full-pool log Dual-Softmax and raw-ID tie breaks."""
import torch
from ..data.supervision_adapter import coarse_truth, project
from ..data.texture_adapter import energy
from ..models.block_router import block_ids,halo_membership,locate
from ..models.sparse_dual_softmax import dual_logp


def ratio(a,b):
    return a/b if b else None


def sparse_rank(x,keys,gt,n):
    i,j = gt["i"],gt["j"]
    ok,at = locate(keys,i*n+j)
    ranks = torch.zeros(n,device=i.device,dtype=torch.long)
    margin = x.new_full((n,),torch.nan)
    if len(keys):
        row,col = (keys//n).long(),(keys%n).long()
        target = gt["target"]
        positive = x.new_full((n,),-torch.inf)
        positive[i[ok]] = x[at[ok]]
        outrank = (x>positive[row])|((x==positive[row])&(col<target[row]))
        ranks[i[ok]] = 1+torch.bincount(row[outrank],minlength=n)[i[ok]]
        neg = x.new_full((n,),-torch.inf).scatter_reduce_(0,row,x.masked_fill(col==target[row],-torch.inf),reduce="amax")
        eligible = positive.isfinite()&neg.isfinite()
        margin[eligible] = positive[eligible]-neg[eligible]
    return ranks,margin


def native_rank(d,gt,temp):
    a,b = [d[f"feat{k}_c"][0].flatten(1).T for k in (0,1)]
    ia,ib = [d[f"candidate_ids{k}"][0] for k in (0,1)]
    n = len(a)
    ranks = torch.zeros(n,dtype=torch.long,device=a.device)
    margin = torch.full((n,),torch.nan,device=a.device)
    if not len(ia) or not len(ib):
        return ranks,margin
    # Column LSE across every retained source. Row LSE cancels within a row rank/margin.
    denominator = temp.float()*a.shape[-1]+1e-8*a.shape[-1]
    aa,bb = a.float(),b[ib].float()
    norm = bb.new_full((len(ib),),-torch.inf)
    for ids in ia.split(64):
        norm = torch.logaddexp(norm,((aa[ids]@bb.T)/denominator).logsumexp(0))
    target_index = ranks.new_full((len(b),),-1)
    target_index[ib] = torch.arange(len(ib),device=a.device)
    for ids in ia.split(64):
        j = gt["target"][ids]
        jj = target_index[j]
        ok = gt["valid"][ids] & (jj>=0)
        if ok.any():
            ids,j,jj = ids[ok],j[ok],jj[ok]
            x = 2*(aa[ids]@bb.T)/denominator-norm
            pos = x.gather(1,jj[:,None])
            ranks[ids] = 1+((x>pos)|((x==pos)&(ib[None]<j[:,None]))).sum(1)
            margin[ids] = (pos[:,0]-x.masked_fill(ib[None]==j[:,None],-torch.inf).amax(1))
    return ranks,margin


def metrics(d,model,threshold,plan):
    gt = coarse_truth(d,model.cfg)
    n = len(gt["valid"])
    w = d["hw0_c"][1]
    v,legal = energy(d,"0")
    den = legal & gt["valid"]
    pool_rank = rank = margin = None
    top = route = halo = None
    if "keys" in d:
        key = d["keys"]
        rank,margin = sparse_rank(d["logp"],key,gt,n)
        pool_rank=rank
        if d["raw_top"] is not None:
            pre = dual_logp(d["base_scores"],key,n,n)
            pool_rank,_ = sparse_rank(pre,key,gt,n)
        all_i = torch.arange(n,device=v.device)
        route = d["route_pred"][block_ids(all_i,w),block_ids(gt["target"],w)]
        halo=halo_membership(d["route_pred"],all_i,gt["target"],d["hw0_c"],model.cfg["halo_radius_coarse_grid"])
        halo &= d["route_masks"][0].flatten() & d["route_masks"][1].flatten()[gt["target"]]
        if d["raw_top"] is not None:
            hit,at = locate(key,all_i*n+gt["target"])
            top = hit & (d["raw_top"]==at[:,None]).any(1)
    else:
        rank,margin = native_rank(d,gt,model.core.coarse_temperature.detach())
        pool_rank = rank
    parents = d["i_idx_c"]
    strict = torch.zeros_like(den)
    strict[parents[d["j_idx_c"]==gt["target"][parents]]] = True
    child = parents[d["m_idx_it"]]
    uv,good = project(d,d["fine_coord_0"].float(),rel=model.cfg["depth_relative_tolerance"],
                      absolute=model.cfg["depth_absolute_tolerance"])
    scale = uv.new_tensor(plan.scale)
    epe = ((uv-d["fine_coord_1"].float())/scale).norm(dim=-1)
    success = torch.zeros_like(den)
    target_inside = ((d["fine_coord_1"]>=0)&(d["fine_coord_1"]<uv.new_tensor([plan.rw,plan.rh]))).all(-1)
    good_match = good & target_inside & (epe<2)
    success[child[good_match]] = True
    any_child=torch.zeros_like(den)
    valid_child=torch.zeros_like(den)
    any_child[child]=True
    valid_child[child[good]]=True
    desc = d["feat0_c"][0].float().flatten(1).norm(dim=0)
    groups = {}
    for name,band in (("lt",v<=threshold),("other",v>threshold)):
        valid = den&band
        inpool = valid&(rank>0)
        pts = good & valid[child]
        m = valid&margin.isfinite()
        counts = dict(den=int(valid.sum()),in_pool=int(inpool.sum()),strict=int((valid&strict).sum()),
                      success=int((valid&success).sum()),valid_children=int(pts.sum()),
                      good_children=int((pts&good_match).sum()),epe_sum=float(epe[pts].sum()),
                      rank_sum=int(rank[valid].sum()),before_rank_sum=int(pool_rank[valid].sum()),
                      margin_sum=float(margin[m].sum()),margin_count=int(m.sum()),
                      norm_sum=float(desc[valid].sum()),competition_failure=int((inpool&~strict).sum()),
                      refinement_failure=int((valid&strict&valid_child&~success).sum()),
                      no_child_after_strict=int((valid&strict&~any_child).sum()),
                      no_valid_child_after_strict=int((valid&strict&any_child&~valid_child).sum()),pool_miss=int((valid&~inpool).sum()))
        for k in (1,5,10):
            counts[f"recall{k}_hits"] = int((inpool&(rank<=k)).sum())
        counts["before_recall1_hits"] = int((inpool&(pool_rank==1)).sum())
        if route is not None:
            counts["route_hits"] = int((valid&route).sum())
            counts["halo_hits"] = int((valid&halo).sum())
            counts["route_misses"] = int((valid&~route).sum())
            counts["truncated_gt"] = int((valid&halo&~inpool).sum())
        if top is not None:
            counts.update(top_hits=int((valid&top).sum()),top_out=int((valid&~top).sum()),
                          top_converted=int((valid&top&strict).sum()))
        groups[name] = counts
    return groups


def rates(c):
    spec = dict(recall=("success","den"),fine_recall=("success","den"),precision=("good_children","valid_children"),
                epe=("epe_sum","valid_children"),coarse_recall=("strict","den"),pool_rate=("in_pool","den"),
                rank=("rank_sum","in_pool"),before_rank=("before_rank_sum","in_pool"),
                margin=("margin_sum","margin_count"),feature_norm=("norm_sum","den"),
                route_recall=("route_hits","den"),halo_recall=("halo_hits","den"),topm_recall=("top_hits","den"),
                topm_conversion=("top_converted","top_hits"),before_recall1=("before_recall1_hits","den"))
    spec.update({f"recall{k}":(f"recall{k}_hits","den") for k in (1,5,10)})
    return {k:ratio(c[x],c[y]) for k,(x,y) in spec.items() if x in c and y in c}
