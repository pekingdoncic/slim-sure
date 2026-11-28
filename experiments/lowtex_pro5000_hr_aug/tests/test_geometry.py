import numpy as np
import pytest
import torch
from experiments.lowtex_pro5000_hr_aug.data.pad32_adapter import ResizePlan,prepare
from experiments.lowtex_pro5000_hr_aug.data.supervision_adapter import project,coarse_truth
from experiments.lowtex_pro5000_hr_aug.evaluation.pose import auc,pair_metrics
from experiments.lowtex_pro5000_hr_aug.utils.run_io import config


@pytest.mark.parametrize("side,active",[(1024,1024),(1216,1200)])
def test_resize_padding_intrinsics_and_half_pixel(side,active):
    raw=np.ones((601,1200),np.uint8)
    k=np.array([[1100.,0.,361.],[0.,980.,245.],[0.,0.,1.]])
    plan=ResizePlan.make(raw.shape,side)
    assert max(plan.rh,plan.rw)==active and plan.side%32==0
    assert plan.mask().sum()==plan.rh*plan.rw
    p=np.array([[0.,0.],[199.25,300.75],[1199.,600.]])
    np.testing.assert_allclose(plan.to_original(plan.to_network(p)),p,atol=1e-12)
    ray=np.linalg.solve(k,np.r_[p[1],1.])
    np.testing.assert_allclose((plan.intrinsics(k)@ray)[:2],plan.to_network(p[1]))
    depth=plan.canvas(np.ones(raw.shape,np.float32),True)
    assert not depth[~plan.mask()].count_nonzero()


def test_projection_translation_occlusion_and_identity_coarse_grid():
    raw=np.ones((64,64),np.uint8)
    k=np.array([[32.,0.,31.5],[0.,32.,31.5],[0.,0.,1.]])
    d,_=prepare([raw,raw],[k,k],64,[np.ones_like(raw),np.ones_like(raw)])
    d={k:v[None] for k,v in d.items()}
    d.update(T_0to1=torch.eye(4)[None],T_1to0=torch.eye(4)[None])
    cfg=config("P1")
    gt=coarse_truth(d,cfg)
    assert gt["valid"].all() and torch.equal(gt["i"],gt["j"])
    points=torch.tensor([[15.5,16.5],[25.5,27.5]])
    d["T_0to1"][0,0,3]=1/32
    uv,ok=project(d,points)
    torch.testing.assert_close(uv,points+torch.tensor([1.,0.]))
    assert ok.all()
    d["depth1"][0,16,16]=2.
    _,ok=project(d,points)
    assert ok.tolist()==[False,True]


def test_auc_failed_pose_stays_in_denominator():
    assert auc([None,None])["auc@10"]==0
    assert auc([0.,None])["auc@10"]==50
    k=np.eye(3)
    t=np.eye(4)
    t[0,3]=1.
    meta=dict(scene="s",pair_id="x",K0=k,K1=k,T_0to1=t,plans=[ResizePlan.make((48,64),64)]*2)
    row=pair_metrics(meta,np.empty((0,2)),np.empty((0,2)))
    assert row["error"] is None and row["matches"]==0


def test_pose_from_calibrated_synthetic_matches():
    rng=np.random.default_rng(91)
    xyz=rng.uniform([-1.,-.7,3.],[1.,.7,8.],(100,3))
    k=np.array([[520.,0.,310.],[0.,495.,242.],[0.,0.,1.]])
    t=np.eye(4)
    t[:3,3]=[.35,.08,.02]
    def pixels(x):
        uv=x@k.T
        return uv[:,:2]/uv[:,2:]
    meta=dict(scene="s",pair_id="synthetic",K0=k,K1=k,T_0to1=t,plans=[ResizePlan.make((480,640),640)]*2)
    row=pair_metrics(meta,pixels(xyz),pixels(xyz+t[:3,3]))
    assert row["error"] is not None and row["error"]<.1 and row["epi_correct"]==100
