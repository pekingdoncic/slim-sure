import math
import random
import numpy as np
import torch


def freeze_phase(model, step, total):
    warm = step < math.ceil(total*model.cfg["new_module_warmup_ratio"])
    for p in model.core.feature_backbone.parameters():
        p.requires_grad_(not warm)
    for m in model.core.feature_backbone.modules():
        if isinstance(m,torch.nn.modules.batchnorm._BatchNorm) and m.affine:
            for p in (m.weight,m.bias):
                p.requires_grad_(not warm and model.cfg["bn_affine_trainable"])
    model.train()
    return warm


def optimizer(model,total):
    cfg = model.cfg
    native = list(model.core.parameters())
    old = {id(p) for p in native}
    new = [p for p in model.parameters() if id(p) not in old]
    groups = [dict(params=native,lr=cfg["backbone_lr"],name="native")]
    if new:
        groups.append(dict(params=new,lr=cfg["new_module_lr"],name="new"))
    opt = torch.optim.AdamW(groups,weight_decay=cfg["weight_decay"],foreach=False)
    return opt,torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=total)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def rng_state():
    return (random.getstate(),np.random.get_state(),torch.get_rng_state(),torch.cuda.get_rng_state())


def restore_rng(state):
    random.setstate(state[0])
    np.random.set_state(state[1])
    torch.set_rng_state(state[2])
    torch.cuda.set_rng_state(state[3])
