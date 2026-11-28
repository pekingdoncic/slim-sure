import torch


def segment_lse(x, index, n):
    index = index.long()
    peak = x.new_full((n,),-torch.inf)
    peak.scatter_reduce_(0,index,x.detach(),reduce="amax")
    total = x.new_zeros(n).scatter_add_(0,index,(x-peak[index]).exp())
    return peak + total.clamp_min(1e-30).log()


def dual_logp(scores, keys, n0, n1):
    """One pair at a time. keys are unique and sorted by BlockRouter."""
    if len(keys)>1 and (keys[1:]<=keys[:-1]).any():
        raise ValueError("duplicate or unsorted candidate edges")
    x = scores.float()
    row,col = (keys//n1).long(),(keys%n1).long()
    return 2*x-segment_lse(x,row,n0)[row]-segment_lse(x,col,n1)[col]


@torch.no_grad()
def select_mnn(logp, keys, n0, n1, threshold, cap):
    if not len(keys):
        return keys.long()
    row,col = (keys//n1).long(),(keys%n1).long()
    edge = torch.arange(len(keys),device=keys.device)
    def winners(idx,n):
        peak = logp.new_full((n,),-torch.inf).scatter_reduce_(0,idx,logp,reduce="amax")
        ids = torch.full((n,),len(keys),dtype=torch.long,device=keys.device)
        ids.scatter_reduce_(0,idx,torch.where(logp==peak[idx],edge,len(keys)),reduce="amin")
        return ids[idx]==edge
    ok = winners(row,n0)&winners(col,n1)
    if threshold>0:
        import math
        ok &= logp > logp.max()+math.log(threshold)
    keep = edge[ok]
    return keep[logp[keep].argsort(descending=True,stable=True)[:cap]]
