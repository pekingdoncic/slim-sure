import torch
from torch import nn
from torch.nn import functional as F


def masked_softmax(scores, valid):
    scores = scores.float().masked_fill(~valid, -torch.inf)
    peak = scores.amax(-1,keepdim=True)
    weights = torch.exp(scores-torch.where(peak.isfinite(),peak,0))
    return weights/weights.sum(-1,keepdim=True).clamp_min(1e-20)


class CCFI(nn.Module):
    def __init__(self, channels, cfg):
        super().__init__()
        dim = cfg["context_dim"]
        self.temperature = cfg["context_temperature"]
        self.project = nn.Sequential(nn.Conv2d(channels,dim,1),nn.GroupNorm(1,dim),nn.GELU())
        self.query,self.key,self.value = (nn.Linear(dim,dim) for _ in range(3))
        self.out = nn.Conv2d(dim,channels,1)
        self.gate = nn.Conv2d(channels+dim+2,1,1)
        nn.init.normal_(self.gate.weight,std=.002)
        nn.init.constant_(self.gate.bias,-4.0)

    def forward(self, f0, f1, m0, m1):
        def tokens(f,m):
            weight = F.avg_pool2d(m[:,None].float(),4)
            deep = F.avg_pool2d(f*m[:,None],4)/weight.clamp_min(1/16)
            x = self.project(deep).flatten(2).transpose(1,2)
            return x,weight.flatten(1)>0,deep.shape[-2:]
        a,ma,ha = tokens(f0,m0)
        b,mb,hb = tokens(f1,m1)
        q,k = F.normalize(self.query(a).float(),dim=-1),F.normalize(self.key(b).float(),dim=-1)
        with torch.autocast(device_type=q.device.type,enabled=False):
            s = q @ k.transpose(1,2) / self.temperature
        valid = ma[:,:,None] & mb[:,None,:]
        def fuse(f,m,hw,v,logits,ok):
            p = masked_softmax(logits,ok)
            context = p.to(v.dtype) @ self.value(v)
            top = p.topk(min(2,p.shape[-1]),dim=-1).values
            margin = top[...,0] - (top[...,1] if top.shape[-1]==2 else 0)
            entropy = -(p*p.clamp_min(1e-30).log()).sum(-1)
            def up(x):
                return F.interpolate(x.transpose(1,2).reshape(f.shape[0],-1,*hw),
                                     size=f.shape[-2:],mode="bilinear",align_corners=False)
            c = up(context)
            info = up(torch.stack((margin,entropy),-1)).to(c.dtype)
            gate = self.gate(torch.cat((f,c,info),1)).sigmoid()
            active = ok.any(-1).reshape(f.shape[0],1,*hw).float()
            active = F.interpolate(active,size=f.shape[-2:],mode="nearest")
            return f + m[:,None]*active*gate*self.out(c)
        return fuse(f0,m0,ha,b,s,valid),fuse(f1,m1,hb,a,s.transpose(1,2),valid.transpose(1,2)),s,ma,mb
