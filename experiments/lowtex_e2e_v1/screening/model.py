import math
import numpy as np
import torch
from torch import nn
from src.loss import SLiM_Loss
from src.texture_budget import local_texture
from configs.cfg_outdoor_train import get_cfg_defaults
from ..models.slim_wrapper import ControlledSLiM, THRESHOLD
from ..training.optimization import UpdatePolicy
from .data import ROOT

class ScreeningLoss(SLiM_Loss):
    def compute_fine_loss(self,data):
        if data["spv_b_ids_it"].numel()==0:
            return data["sim_matrix_f"].reshape(-1)[:0].sum()
        return super().compute_fine_loss(data)

class BalancedLoss(ScreeningLoss):
    def compute_coarse_loss(self,d):
        sim=d["sim_matrix"].float()
        b,i,j=torch.where(d["conf_matrix_gt"]==1)
        keep=torch.randperm(len(i),device=i.device)[:int(len(i)*self.coarse_percent)]
        b,i,j=b[keep],i[keep],j[keep]
        if not len(i):return sim.flatten()[:0].sum()
        ri,inv_i=i.unique(return_inverse=True)
        cj,inv_j=j.unique(return_inverse=True)
        confidence=sim[0,ri].softmax(1)[inv_i,j]*sim[0,:,cj].softmax(0)[i,inv_j]
        terms=-(1-confidence).pow(self.coarse_gamma)*confidence.clamp_min(1e-6).log()
        valid=d["mask0"].flatten()[i]&d["mask1"].flatten()[j]
        texture,_=local_texture(d["image0"],d["valid_mask0"],d["mask0"].shape[-2:])
        low=(texture.flatten()[i]<=THRESHOLD)&valid
        other=(~low)&valid
        nl,no=int(low.sum()),int(other.sum())
        if not nl+no:return terms.sum()*0
        frac=nl/(nl+no)
        alpha=min(4*frac,max(1-4*(1-frac),.4)) if nl and no else float(bool(nl))
        l=terms[low].mean() if nl else terms.new_zeros(())
        o=terms[other].mean() if no else terms.new_zeros(())
        d["balanced_stats"]={"lt":nl,"other":no,"lt_weight":alpha,"lt_loss":float(l.detach()),"other_loss":float(o.detach())}
        return alpha*l+(1-alpha)*o

class CropHead(nn.Module):
    def __init__(self,multiscale):
        super().__init__()
        self.multi=multiscale
        self.high=nn.Conv2d(48,12,1)
        self.middle=nn.Conv2d(96,12,1)
        self.regress=nn.Sequential(nn.Conv2d(48,32,3,padding=1),nn.GroupNorm(4,32),nn.GELU(),
                                  nn.Conv2d(32,16,3,padding=1),nn.GroupNorm(4,16),nn.GELU(),
                                  nn.AdaptiveAvgPool2d((2,2)),nn.Flatten(),nn.Linear(64,2),nn.Tanh())
    def forward(self,d,pyramid,extract):
        idx=d["b_idx_it"]; xy=[d["intermediate_coord_"+str(i)].float()/d["scale"+str(i)][idx] for i in (0,1)]
        projected=[]
        for side in (0,1):
            high=self.high(d["feat"+str(side)+"_f"])
            middle=self.middle(pyramid[side][1]) if self.multi else None
            projected.append((high,middle))
        parts=[]
        for start in range(0,len(idx),128):
            sl=slice(start,start+128);feats=[]
            for side,(high,middle) in enumerate(projected):
                h=extract(high,idx[sl],xy[side][sl],6)
                mid=extract(middle,idx[sl],xy[side][sl]/4,6) if self.multi else torch.zeros_like(h)
                feats.extend((h,mid))
            parts.append(self.regress(torch.cat(feats,dim=1)).float())
        offset=torch.cat(parts) if parts else xy[1].new_empty((0,2))
        d.update(fine_coord_0=d["intermediate_coord_0"],fine_coord_1=(xy[1]+offset*3)*d["scale1"][idx],all_offset_1=offset[None])

class ScreenModel(ControlledSLiM):
    def __init__(self,cfg,cp):
        super().__init__(cfg.MODEL,freeze_bn=True,checkpointing=cp,fine_cap=1024)
        self.compute_features=self.feature_backbone.forward
        self.precomputed=None
        self.pyramid=None
        self.crop=None
        def features(a,b):
            result=self.precomputed if self.precomputed is not None else self.compute_features(a,b)
            if self.crop is not None:self.pyramid=result
            return result
        self.feature_backbone.forward=features
    def _refinement(self,data,*args,**kwargs):
        if self.crop is None:return super()._refinement(data,*args,**kwargs)
        return self.crop(data,self.pyramid,self._extract_feat_window_bilinear)
    def micro_loss(self,batch,cfg,objective):
        shared=None
        if len(batch)>1:
            with torch.autocast("cuda",dtype=torch.bfloat16):
                shared=self.compute_features(torch.cat([d["image0"] for _,d in batch]),torch.cat([d["image1"] for _,d in batch]))
        losses=[];stats=[]
        try:
            for i,(ident,d) in enumerate(batch):
                if shared is not None:self.precomputed=tuple([level[i:i+1] for level in side] for side in shared)
                loss,a=self.train_pair(d,cfg,objective,torch.bfloat16)
                if not loss.requires_grad:
                    if a["valid_gt"] or (not a.get("skip_reason") and any(p.requires_grad for p in self.feature_backbone.parameters())):
                        raise RuntimeError("Supervised loss unexpectedly lost its gradient")
                    loss=loss+next(p for p in self.parameters() if p.requires_grad).flatten()[0]*0
                losses.append(loss);stats.append(dict(**a,identity=ident,loss=float(loss.detach()),balanced=d.get("balanced_stats")))
            return torch.stack(losses).sum(),stats
        finally:self.precomputed=None;self.pyramid=None

def components(variant,cp,total):
    import random
    random.seed(66);np.random.seed(66);torch.manual_seed(66);torch.cuda.manual_seed_all(66)
    cfg=get_cfg_defaults()
    model=ScreenModel(cfg,cp)
    weights=torch.load(ROOT/"ckpt/megadepth_19epochs.ckpt",map_location="cpu")["state_dict"]
    model.load_state_dict(weights,strict=True)
    head=None
    torch.manual_seed(6166)
    if variant in ("single","crop","joint"):
        head=model.crop=CropHead(variant!="single")
    elif variant=="rru_matched":
        head=model.recurrent_refine_unit
        for m in head.modules():
            if hasattr(m,"reset_parameters"):m.reset_parameters()
    elif variant not in ("exp0","exp1a"):raise ValueError(variant)
    model.cuda()
    policy=UpdatePolicy(model,head,dict(new_head_warmup=.1,new_head_lr=1e-4,backbone_lr=2e-5,weight_decay=.01,
                                       milestones=[6,10,14,16]),total,"end_to_end")
    objective=(BalancedLoss if variant in ("exp1a","joint") else ScreeningLoss)(cfg.LOSS)
    torch.manual_seed(66);torch.cuda.manual_seed_all(66)
    return model,cfg,objective,policy
