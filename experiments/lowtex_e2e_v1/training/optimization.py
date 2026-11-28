import math
import torch


class UpdatePolicy:
    def __init__(self,model,head,settings,total,mode):
        self.model=model
        self.head=head
        self.mode=mode
        self.warmup=math.ceil(settings["new_head_warmup"]*total) if head is not None else 0
        new_ids={id(p) for p in head.parameters()} if head is not None else set()
        old=[p for p in model.parameters() if id(p) not in new_ids]
        groups=[dict(params=old,lr=settings["backbone_lr"],initial_lr=settings["backbone_lr"])]
        if head is not None:
            groups.append(dict(params=list(head.parameters()),lr=settings["new_head_lr"],initial_lr=settings["new_head_lr"]))
        self.optimizer=torch.optim.AdamW(groups,weight_decay=settings["weight_decay"])
        self.scheduler=torch.optim.lr_scheduler.MultiStepLR(self.optimizer,
                milestones=[math.ceil(total*n/27) for n in settings["milestones"]],gamma=.5)
        self.apply(0)

    def apply(self,step):
        head_only=self.mode=="head_only" or step<self.warmup
        if head_only and self.head is None:
            raise ValueError("head_only requires an explicit trainable head")
        for p in self.model.parameters():p.requires_grad_(not head_only)
        if self.mode=="partial_unfreeze" and hasattr(self.model,"feature_backbone"):
            for p in self.model.feature_backbone.first_stage.parameters():p.requires_grad_(False)
        if self.head is not None:
            for p in self.head.parameters():p.requires_grad_(True)
        self.model.train()
        if head_only and hasattr(self.model,"feature_backbone"):
            self.model.feature_backbone.eval()
