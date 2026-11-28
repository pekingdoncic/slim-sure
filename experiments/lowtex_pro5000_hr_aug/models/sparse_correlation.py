import torch


class _EdgeDot(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b, keys, chunk):
        ctx.save_for_backward(a,b,keys)
        ctx.chunk = chunk
        out = torch.empty(keys.numel(),device=a.device,dtype=torch.float32)
        for start in range(0,len(keys),chunk):
            e = keys[start:start+chunk].long()
            out[start:start+len(e)] = (a[e//len(b)].float()*b[e%len(b)].float()).sum(-1)
        return out

    @staticmethod
    def backward(ctx, grad):
        a,b,keys = ctx.saved_tensors
        ga = torch.zeros_like(a,dtype=torch.float32)
        gb = torch.zeros_like(b,dtype=torch.float32)
        for start in range(0,len(keys),ctx.chunk):
            e = keys[start:start+ctx.chunk].long()
            i,j = e//len(b),e%len(b)
            g = grad[start:start+len(e),None].float()
            ga.index_add_(0,i,g*b[j].float())
            gb.index_add_(0,j,g*a[i].float())
        return ga.to(a.dtype),gb.to(b.dtype),None,None


def edge_logits(a,b,keys,temperature,chunk):
    if chunk<=0:
        raise ValueError("positive correlation chunk required")
    return _EdgeDot.apply(a,b,keys,chunk)/(a.shape[-1]*(temperature.float()+1e-8))
