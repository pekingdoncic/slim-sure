"""Server-only real Conv-Mamba, native fine matcher and RRU tests."""
import os
from pathlib import Path
import pytest
import torch
from experiments.lowtex_pro5000_hr_aug.utils.run_io import ROOT,config

pytestmark=pytest.mark.gpu


@pytest.fixture(scope="module",autouse=True)
def allocator():
    from experiments.lowtex_pro5000_hr_aug.training.runtime import Guard
    guard=Guard(float(os.environ.get("CCS_VRAM_CAP_GIB","30")))
    yield guard
    guard.close()


def weights():
    return Path(os.environ.get("CCS_WEIGHTS",ROOT/"ckpt/megadepth_19epochs.ckpt"))


def synthetic(side=96):
    gen=torch.Generator().manual_seed(71)
    image=torch.rand((1,1,side,side),generator=gen).cuda()
    k=torch.tensor([[[80.,0.,side/2],[0.,80.,side/2],[0.,0.,1.]]],device="cuda")
    d=dict(T_0to1=torch.eye(4,device="cuda")[None],T_1to0=torch.eye(4,device="cuda")[None],
           position=torch.tensor([19],device="cuda"))
    for i in (0,1):
        d.update({f"image{i}":image.clone(),f"mask{i}":torch.ones((1,side//8,side//8),dtype=torch.bool,device="cuda"),
                 f"valid_mask{i}":torch.ones_like(image[:,0],dtype=torch.bool),f"depth{i}":torch.ones_like(image[:,0]),
                 f"K{i}":k.clone(),f"scale{i}":torch.ones(1,2,device="cuda")})
    return d


def test_all_off_equivalent_to_native():
    from configs.cfg_outdoor_test import get_cfg_defaults
    from src.slim import SLiM
    from experiments.lowtex_pro5000_hr_aug.models.slim_wrapper import CCSMatcher
    cfg=config("P4")
    cfg.update(ccfi=False,bbr=False,lhg=False)
    official=SLiM(get_cfg_defaults().MODEL).cuda().eval()
    official.load_state_dict(torch.load(weights(),map_location="cpu",weights_only=False)["state_dict"])
    wrapped=CCSMatcher(cfg,weights()).cuda().eval()
    left=synthetic()
    right={k:v.clone() for k,v in left.items()}
    with torch.no_grad(),torch.autocast("cuda",dtype=torch.float16):
        official(left)
        wrapped(right)
    for k in ("b_idx_c","i_idx_c","j_idx_c","b_idx_it","m_idx_it","i_idx_it","j_idx_it","fine_coord_0","fine_coord_1","all_offset_1"):
        torch.testing.assert_close(left[k],right[k],rtol=0,atol=0)


def test_zero_residual_real_p3_p4():
    from experiments.lowtex_pro5000_hr_aug.models.slim_wrapper import CCSMatcher
    p3=CCSMatcher(config("P3"),weights()).cuda().eval()
    p4=CCSMatcher(config("P4"),weights()).cuda().eval()
    p4.ccfi.load_state_dict(p3.ccfi.state_dict())
    with torch.no_grad(),torch.autocast("cuda",dtype=torch.float16):
        a,b=p3(synthetic()),p4(synthetic())
    for k in ("keys","base_scores","updated_scores","logp","i_idx_c","j_idx_c","fine_coord_0","fine_coord_1"):
        torch.testing.assert_close(a[k],b[k],rtol=0,atol=0)


def test_multiupdate_graph_backbone_checkpoint_bn_and_resume(tmp_path):
    from experiments.lowtex_pro5000_hr_aug.models.slim_wrapper import CCSMatcher
    from experiments.lowtex_pro5000_hr_aug.training.optimizer import optimizer,freeze_phase
    cfg=config("P4")
    cfg.update(max_fine_train_matches=32,router_topk=2)
    model=CCSMatcher(cfg,weights()).cuda()
    opt,scheduler=optimizer(model,5)
    bn=[m for m in model.core.feature_backbone.modules() if isinstance(m,torch.nn.modules.batchnorm._BatchNorm)]
    previous=[m.running_mean.clone() for m in bn]
    assert bn
    for step,ck in enumerate(("off","full","selective")):
        warm=freeze_phase(model,step,5)
        assert warm==(step==0)
        assert not any(m.training for m in bn)
        model.checkpointing=ck
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda",dtype=torch.float16):
            loss,_=model.training_loss(synthetic(),1.)
        assert torch.isfinite(loss)
        loss.backward()
        observed=(model.graph.out,model.ccfi) if warm else (model.graph.embed,model.graph.out,model.ccfi,model.core.feature_backbone)
        for module in observed:
            assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for p in module.parameters())
        opt.step()
        scheduler.step()
        model.eval().train()
    for before,m in zip(previous,bn):
        torch.testing.assert_close(before,m.running_mean,rtol=0,atol=0)
    from experiments.lowtex_pro5000_hr_aug.training.trainer import save_checkpoint
    path=tmp_path/"checkpoint.ckpt"
    save_checkpoint(path,dict(model=model.state_dict(),optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),step=3))
    restored=CCSMatcher(cfg,weights()).cuda()
    state=torch.load(path,map_location="cpu",weights_only=False)
    restored.load_state_dict(state["model"])
    op,sc=optimizer(restored,5)
    op.load_state_dict(state["optimizer"])
    sc.load_state_dict(state["scheduler"])
    assert sc.last_epoch==scheduler.last_epoch
    with torch.no_grad(),torch.autocast("cuda",dtype=torch.float16):
        a=model.eval()(synthetic())
        b=restored.eval()(synthetic())
    torch.testing.assert_close(a["updated_scores"],b["updated_scores"])


def test_checkpointed_backbone_gradient_matches_native_execution():
    from experiments.lowtex_pro5000_hr_aug.models.slim_wrapper import CCSMatcher
    model=CCSMatcher(config("P2"),weights()).cuda().train()
    d=synthetic()
    d={k:torch.cat((v,v),0) for k,v in d.items()}
    par=model.core.feature_backbone.first_stage.patch_embed.conv.weight
    def gradient(mode):
        model.checkpointing=mode
        with torch.autocast("cuda",dtype=torch.float16):
            a,b,*_=model.features(d)
            objective=a.float().square().mean()+b.float().square().mean()
        return torch.autograd.grad(objective,par)[0]
    reference=gradient("off")
    assert reference.norm()>0
    for mode in ("selective","full"):
        torch.testing.assert_close(gradient(mode),reference,rtol=.02,atol=5e-4)
