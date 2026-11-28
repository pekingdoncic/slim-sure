from contextlib import contextmanager

import torch
from torch.utils.checkpoint import checkpoint

from src.slim import SLiM
from src.texture_budget import local_texture
from src.utils.supervision import spvs_coarse, spvs_intermediate, compute_supervision_fine

THRESHOLD = 0.0001920461654663086


class ControlledSLiM(SLiM):
    def __init__(self, config, *, freeze_bn=False, checkpointing=False, fine_cap=None):
        super().__init__(config)
        self.freeze_bn=freeze_bn
        self.checkpointing=checkpointing
        self.fine_cap=fine_cap
        backbone=self.feature_backbone
        original=backbone.forward

        def features(a,b):
            if self.training and self.checkpointing and any(p.requires_grad for p in backbone.parameters()):
                return checkpoint(original,a,b,use_reentrant=False)
            return original(a,b)
        backbone.forward=features

    def train(self,mode=True):
        super().train(mode)
        if mode and self.freeze_bn:
            for m in self.feature_backbone.modules():
                if isinstance(m,torch.nn.modules.batchnorm._BatchNorm):
                    m.eval()
        return self

    def bn_snapshot(self):
        return {k:v.clone() for k,v in self.feature_backbone.named_buffers()
                if k.endswith(("running_mean","running_var","num_batches_tracked"))}

    @contextmanager
    def training_cap(self):
        original=self.max_fine_matches
        if self.fine_cap is not None:
            self.max_fine_matches=min(original,self.fine_cap)
        try:
            yield
        finally:
            self.max_fine_matches=original

    def train_pair(self,data,cfg,loss_fn,amp_dtype):
        spvs_coarse(data,cfg.MODEL.COARSE_SCALE)
        ids=data["spv_i_ids"]
        if data["image0"].shape[0]!=1:
            raise ValueError("single-pair microbatch required by this profile")
        h,w=data["mask0"].shape[-2:]
        texture,_=local_texture(data["image0"],data["valid_mask0"],(h,w))
        low=texture.flatten() <= THRESHOLD
        stats={"parents_before":len(ids),"lt_parents_before":int(low[ids].sum())}
        if self.fine_cap is not None and len(ids)>self.fine_cap:
            selection=torch.randperm(len(ids),device=ids.device)[:self.fine_cap].sort().values
            for key in ("spv_b_ids","spv_i_ids","spv_j_ids"):
                data[key]=data[key][selection]
        stats.update(parents_after=len(data["spv_i_ids"]),lt_parents_after=int(low[data["spv_i_ids"]].sum()))
        if len(ids) < 2:
            stats.update(children_before=0,children_after=0,lt_children_before=0,lt_children_after=0,valid_gt=0,skip_reason="fewer than two coarse positives; native 90% sample empty")
            return self.coarse_temperature*0.,stats
        spvs_intermediate(data,cfg)
        stats["children_before"]=len(data["spv_b_ids_it"])
        src=data["spv_i_ids"][data["spv_m_ids_it"]]
        stats["lt_children_before"]=int(low[src].sum())
        with self.training_cap(),torch.autocast("cuda",dtype=amp_dtype):
            self(data,training=True)
        compute_supervision_fine(data,cfg)
        for key in ("sim_matrix","sim_matrix_f","all_offset_1"):
            if key in data:
                data[key]=data[key].float()
        loss_fn(data)
        stats["valid_gt"]=int(data["correct_mask"].sum())
        n=len(data["m_idx_it"])
        if self.fine_cap is not None and n>self.fine_cap:
            raise ValueError("fine cap exceeded")
        stats.update(children_after=n,lt_children_after=int(low[data["spv_i_ids"][data["m_idx_it"]]].sum()))
        return data["loss"],stats
