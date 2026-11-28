import torch
from torch.utils.checkpoint import checkpoint
from ..models.block_router import locate


def focal(logp,gamma):
    p = logp.exp().clamp(max=1)
    return -(1-p).pow(gamma)*p.clamp_min(1e-6).log()


def sparse_focal(logp,keys,gt,n,gamma):
    ok,at = locate(keys,gt["i"]*n+gt["j"])
    return (focal(logp[at[ok]],gamma).mean() if ok.any() else logp.sum()*0),int(ok.sum())


def native_focal(a,b,mask0,mask1,i,j,temperature,loss,chunk,generator):
    """Original sampled positive focal objective per pair, streaming the two normalizers."""
    num = int(len(i)*loss.coarse_percent)
    if not num:
        return (a.sum()+temperature)*0
    keep = torch.randperm(len(i),device=i.device,generator=generator)[:num]
    i,j = i[keep],j[keep]
    def part(a,b,ii,jj,t):
        with torch.autocast(device_type=a.device.type,enabled=False):
            aa,bb = a.float()/a.shape[1]**.5,b.float()/b.shape[1]**.5
            r = ((aa[ii]@bb.T)/(t+1e-8)).masked_fill(~(mask0[ii,None]&mask1[None,:]),-1e9)
            c = ((bb[jj]@aa.T)/(t+1e-8)).masked_fill(~(mask1[jj,None]&mask0[None,:]),-1e9)
            ar = torch.arange(len(ii),device=i.device)
            lp = r[ar,jj]-r.logsumexp(1)+c[ar,ii]-c.logsumexp(1)
            return (focal(lp,loss.coarse_gamma)*(mask0[ii]&mask1[jj])).sum()
    total = sum(checkpoint(part,a,b,ii,jj,temperature,use_reentrant=False)
                for ii,jj in zip(i.split(chunk),j.split(chunk)))
    return total/num
