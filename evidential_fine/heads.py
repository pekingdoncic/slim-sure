"""Original implementations of the preregistered heads and normalized NIG objective."""
import math,torch
from torch import nn
from torch.nn import functional as F
FLOORS={'nu':1e-6,'beta':1e-6,'alpha_minus_one':1e-4}
def block(a,b):return nn.Sequential(nn.Linear(a,b),nn.LayerNorm(b),nn.GELU())
class EvidentialHead(nn.Module):
    def __init__(self,route,coarse=192,fine=48):
        super().__init__();self.route=route;self.R=8 if route=='A' else 3;self.K=32 if route=='A' else 12
        if route=='A':
            self.coarse_projection=nn.Linear(coarse,128);self.patch_projection=nn.Linear(4*fine,128)
            self.side=nn.Sequential(block(128,128),block(128,128))
            self.merge=nn.Sequential(block(256,128),block(128,128))
        else:
            self.windows=nn.Sequential(nn.Conv2d(2*fine,64,3,padding=1),nn.GroupNorm(8,64),nn.GELU(),
                nn.Conv2d(64,64,3,padding=1),nn.GroupNorm(8,64),nn.GELU())
            self.merge=block(64*6*6,128)
        self.x=nn.Sequential(nn.Linear(128,64),nn.GELU(),nn.Linear(64,self.K+4))
        self.y=nn.Sequential(nn.Linear(128,64),nn.GELU(),nn.Linear(64,self.K+4))
        self.register_buffer('bins',torch.linspace(-self.R,self.R,self.K+1))
    def forward(self,inputs):
        if self.route=='A':
            c0,c1,p0,p1=inputs
            a=self.side(self.coarse_projection(c0)+self.patch_projection(p0))
            b=self.side(self.coarse_projection(c1)+self.patch_projection(p1))
            z=self.merge(torch.cat([a,b],dim=1))
        else:z=self.merge(self.windows(inputs[0]).flatten(1))
        raw=torch.stack([self.x(z),self.y(z)],1).float()
        with torch.autocast(device_type=raw.device.type,enabled=False):
            mu=(raw[:,:,:self.K+1].softmax(-1)*self.bins.float()).sum(-1)
            evidence=F.softplus(raw[:,:,self.K+1:])
            nu=evidence[:,:,0].clamp_min(FLOORS['nu'])
            delta=evidence[:,:,1].clamp_min(FLOORS['alpha_minus_one'])
            beta=evidence[:,:,2].clamp_min(FLOORS['beta']);alpha=1+delta
            ua=beta/(alpha-1).clamp_min(FLOORS['alpha_minus_one'])*self.R**2;ue=ua/nu
            floors=torch.stack([(evidence[:,:,0]<FLOORS['nu']).float().mean(),(evidence[:,:,1]<FLOORS['alpha_minus_one']).float().mean(),(evidence[:,:,2]<FLOORS['beta']).float().mean()]) if len(raw) else raw.new_zeros(3)
        return dict(mu=mu,nu=nu,alpha=alpha,beta=beta,aleatoric=ua.mean(1),epistemic=ue.mean(1),floor_rates=floors)
def nig_loss(out,offset,R):
    with torch.autocast(device_type=offset.device.type,enabled=False):
        u=out['mu'].float()/R;y=offset.float()/R
        nu=out['nu'].float();alpha=out['alpha'].float();beta=out['beta'].float()
        omega=2*beta*(1+nu)
        nll=.5*torch.log(torch.pi/nu)-alpha*torch.log(omega)+(alpha+.5)*torch.log(nu*(u-y).square()+omega)+torch.lgamma(alpha)-torch.lgamma(alpha+.5)
        reg=(u-y).abs()*(2*nu+alpha)
        return (nll+reg).mean(),dict(nll=nll.mean(),regularizer=reg.mean())
def numerical_checks():
    import numpy as np
    from scipy.stats import t
    torch.manual_seed(2026)
    u=torch.randn(37,2,requires_grad=True);y=torch.randn(37,2)
    raw=torch.randn(37,2,3,requires_grad=True)*3
    nu=F.softplus(raw[:,:,0]).clamp_min(1e-6);a=1+F.softplus(raw[:,:,1]).clamp_min(1e-4);b=F.softplus(raw[:,:,2]).clamp_min(1e-6)
    out=dict(mu=u,nu=nu,alpha=a,beta=b);loss,parts=nig_loss(out,y,1)
    scale=torch.sqrt(b*(1+nu)/(a*nu)).detach().numpy()
    oracle=-t.logpdf(y.numpy(),df=2*a.detach().numpy(),loc=u.detach().numpy(),scale=scale)
    diff=abs(float(parts['nll'].detach())-float(np.mean(oracle)))
    assert diff<1e-5,(diff,float(loss));loss.backward();assert torch.isfinite(u.grad).all()
    extremes=[]
    for value in [-1000.,1000.]:
        z=torch.full((8,2,3),value,requires_grad=True)
        ev=F.softplus(z);v=ev[:,:,0].clamp_min(1e-6);aa=1+ev[:,:,1].clamp_min(1e-4);bb=ev[:,:,2].clamp_min(1e-6)
        pred=torch.zeros(8,2,requires_grad=True);l,_=nig_loss(dict(mu=pred,nu=v,alpha=aa,beta=bb),torch.ones(8,2),1);l.backward()
        assert torch.isfinite(l) and torch.isfinite(z.grad).all() and torch.isfinite(pred.grad).all()
        extremes.append(dict(raw=value,loss=float(l.detach()),backward_finite=True))
    return dict(status='pass',student_t_independent_nll_max_difference=diff,normal_backward_finite=True,extreme_cases=extremes,floors=FLOORS,reference='User formula; equivalent independent scipy Student-t logpdf; SURE f702f618 source inspected, no code copied')
