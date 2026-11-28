import torch
from ..models.block_router import locate


def ranking(scores,keys,table,gt,hw,cfg):
    """Highest-scoring competing hypothesis with reliable forward AND reverse geometric evidence."""
    h,w = hw
    n = h*w
    if not len(keys):
        return scores.sum()*0,0
    i,j = gt["i"],gt["j"]
    present,pos = locate(keys,i*n+j)
    candidates = table[i]
    other = keys[candidates.clamp_min(0)].long()%n
    def far(a,b):
        return torch.maximum((a%w-b%w).abs(),(a//w-b//w).abs())>=cfg["negative_distance_cells"]
    known = (candidates>=0)&gt["reverse"][other]&far(other,j[:,None])&far(gt["back"][other],i[:,None])
    minus = scores[candidates.clamp_min(0)].masked_fill(~known,-torch.inf).amax(1)
    ok = present & (candidates==pos[:,None]).any(1) & known.any(1)
    if not ok.any():
        return scores.sum()*0,0
    return (cfg["ranking_margin"]+minus[ok]-scores[pos[ok]]).relu().mean(),int(ok.sum())
