"""Real-data GPU comparison with all experimental controls disabled."""
import argparse
import random
import numpy as np
import torch
from configs.cfg_outdoor_train import get_cfg_defaults
from src.slim import SLiM
from src.loss import SLiM_Loss
from src.utils.supervision import compute_supervision_coarse,compute_supervision_fine
from experiments.lowtex_e2e_v1.evaluation.texturepoor_sfm.dataset import EXP,ROOT
from experiments.lowtex_e2e_v1.training.data import load_pair
from experiments.lowtex_e2e_v1.models.slim_wrapper import ControlledSLiM

cfg=get_cfg_defaults()
from experiments.lowtex_e2e_v1.training.sampling import Pool
pool=Pool(EXP,ROOT/"data/megadepth",66)
bi,ordinal,frames,_=pool.at(0)
name=pool.bins[bi]["bin"]
weights=torch.load(ROOT/"ckpt/megadepth_19epochs.ckpt",map_location="cpu")["state_dict"]


def batch():
    d,_=load_pair(ROOT/"data/megadepth",name,frames,320,"cuda",f"{name}/{ordinal}")
    return d


def probe(cls):
    m=cls(cfg.MODEL).cuda().eval()
    m.load_state_dict(weights);m.capture_reliability=True
    with torch.no_grad(),torch.autocast("cuda",dtype=torch.float16):
        d=batch();m(d,training=False)
        results={k:d[k].detach().cpu() for k in ("i_idx_c","j_idx_c","i_idx_it","j_idx_it","fine_coord_0","fine_coord_1","coarse_confidence")}
    sim=d["sim_matrix_f"].float()
    conf=sim.softmax(1)*sim.softmax(2)
    results["fine_confidence"]=conf[d["m_idx_it"],d["i_idx_it"],d["j_idx_it"]].cpu()
    m.capture_reliability=False
    random.seed(66);np.random.seed(66)
    torch.manual_seed(66);torch.cuda.manual_seed_all(66)
    d=batch();m.train()
    loss_fn=SLiM_Loss(cfg.LOSS)
    if cls==SLiM:
        compute_supervision_coarse(d,cfg.MODEL.COARSE_SCALE,cfg)
        with torch.autocast("cuda",dtype=torch.bfloat16):
            m(d,training=True)
        compute_supervision_fine(d,cfg)
        for key in ("all_offset_1","sim_matrix_f","sim_matrix"):d[key]=d[key].float()
        loss_fn(d);loss=d["loss"]
    else:
        loss,_=m.train_pair(d,cfg,loss_fn,torch.bfloat16)
    loss.backward()
    results["loss"]=loss.detach().cpu()
    for key in ("coarse_temperature","fine_temperature","feature_backbone.first_stage.patch_embed.conv.weight"):
        results[key]=m.get_parameter(key).grad.detach().float().cpu()
    return results


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--repeatability",action="store_true")
    parser.add_argument("--deterministic",action="store_true")
    args=parser.parse_args()
    if args.deterministic:
        torch.use_deterministic_algorithms(True,warn_only=True)
        torch.backends.cudnn.benchmark=False
        torch.backends.cudnn.deterministic=True
    ref=probe(SLiM)
    key="feature_backbone.first_stage.patch_embed.conv.weight"
    if args.repeatability:
        torch.cuda.empty_cache()
        repeated=probe(SLiM)
        norm=float((ref[key]-repeated[key]).norm()/ref[key].norm().clamp_min(1e-8))
        print("native_vs_native",key,"relative_gradient_norm",norm,flush=True)
    torch.cuda.empty_cache()
    other=probe(ControlledSLiM)
    for key,a in ref.items():
        b=other[key]
        if key in ("coarse_temperature","fine_temperature","feature_backbone.first_stage.patch_embed.conv.weight"):
            # Parallel reductions in custom kernels use reduced precision; compare normalized gradient norm.
            ratio=float((a-b).norm()/a.norm().clamp_min(1e-8))
            print(key,"relative_gradient_norm",ratio,flush=True)
            assert ratio<=0.05 and torch.isfinite(b).all()
        else:
            torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-5)
        print(key,tuple(a.shape),"PASS",flush=True)
