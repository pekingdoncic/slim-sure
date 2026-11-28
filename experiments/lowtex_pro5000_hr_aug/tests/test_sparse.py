import pytest
import torch
from experiments.lowtex_pro5000_hr_aug.models.block_router import BlockRouter,gt_blocks,locate
from experiments.lowtex_pro5000_hr_aug.models.sparse_correlation import edge_logits
from experiments.lowtex_pro5000_hr_aug.models.sparse_dual_softmax import dual_logp,select_mnn
from experiments.lowtex_pro5000_hr_aug.losses.routing_loss import routing_loss
from experiments.lowtex_pro5000_hr_aug.utils.run_io import config


def test_chunked_forward_backward_reference():
    torch.manual_seed(8)
    keys=torch.tensor([0,2,8,9,14,15],dtype=torch.int32)
    cpu=[torch.randn(5,7,requires_grad=True),torch.randn(4,7,requires_grad=True),torch.tensor(.31,requires_grad=True)]
    ref=[x.detach().clone().requires_grad_() for x in cpu]
    sparse=edge_logits(*cpu[:2],keys,cpu[2],2)
    lp=dual_logp(sparse,keys,5,4)
    i,j=keys.long()//4,keys.long()%4
    r=ref[0]@ref[1].T/(7*(ref[2]+1e-8))
    mask=torch.zeros_like(r,dtype=torch.bool)
    mask[i,j]=True
    dense=r.masked_fill(~mask,-1e9)
    target=(torch.log_softmax(dense,0)+torch.log_softmax(dense,1))[i,j]
    torch.testing.assert_close(sparse,r[i,j])
    torch.testing.assert_close(lp,target)
    weights=torch.linspace(.2,1.2,len(lp))
    for got,want in zip(torch.autograd.grad(lp@weights,cpu),torch.autograd.grad(target@weights,ref)):
        torch.testing.assert_close(got,want,atol=3e-5,rtol=2e-5)


def test_equal_empty_ties_duplicates_and_independent_pairs():
    edges=torch.arange(4,dtype=torch.int32)
    a=dual_logp(torch.zeros(4),edges,2,2)
    assert torch.allclose(a.exp(),torch.full((4,),.25))
    assert select_mnn(a,edges,2,2,.1,100).tolist()==[0]
    assert not len(dual_logp(a[:0],edges[:0],2,2))
    with pytest.raises(ValueError,match="duplicate"):
        dual_logp(torch.zeros(2),torch.tensor([0,0]),2,2)
    second=dual_logp(torch.tensor([22.,-4.,-1.,9.]),edges,2,2)
    assert second[0].exp()>.99 and a[0].exp()==.25


def test_routing_halo_masks_bidirectionality_and_supervision():
    cfg=config("P3")
    cfg.update(router_topk=1,route_block_chunk=1)
    scores=torch.tensor([[4.,3.,2.,1.],[9.,1.,1.,1.],[2.,8.,6.,1.],[1.,7.,2.,0.]],requires_grad=True)
    mask=torch.ones(8,8,dtype=torch.bool)
    valid=torch.ones(4,dtype=torch.bool)
    gt=dict(i=torch.tensor([0,1]),j=torch.tensor([0,63]),reverse=torch.zeros(64,dtype=torch.bool),back=torch.zeros(64,dtype=torch.long))
    router=BlockRouter(cfg)
    edges,truth,stats,pred=router(scores,(valid,valid),(mask,mask),gt=gt)
    assert pred[0,0] and pred[2,2]  # reverse top choice for block 2
    assert (edges[1:]>edges[:-1]).all()
    assert locate(edges,torch.tensor([4]))[0].item()  # one-cell halo of block 0
    assert stats["raw_edges"]>=stats["deduplicated_edges"]
    assert {0,3}==set(truth.tolist())
    assert pred[0,3] or stats["missing_blocks"]>0
    injected,_,info,_=router(scores,(valid,valid),(mask,mask),gt,1.,True)
    assert locate(injected,torch.tensor([1*64+63]))[0].item()
    assert info["truncated_gt"]==0
    with pytest.raises(ValueError,match="injection"):
        router(scores,(valid,valid),(mask,mask),gt,1.,False)
    loss=routing_loss(scores,valid,valid,truth)
    loss.backward()
    assert scores.grad.abs().sum()>0
    pad=mask.clone()
    pad[:,7]=False
    masked,*_=router(scores,(valid,valid),(pad,pad))
    assert ((masked//64)%8!=7).all() and ((masked%64)%8!=7).all()
    cfg.update(edge_budget_per_pair=3)
    with pytest.raises(RuntimeError,match="budget"):
        router(scores,(valid,valid),(mask,mask))
    cfg["edge_budget_policy"]="truncate"
    truncated,_,s,_=router(scores,(valid,valid),(mask,mask),gt)
    assert len(truncated)==3 and s["truncated"]>0


def test_dense_native_focal_matches_original_single_pair():
    from configs.cfg_outdoor_test import get_cfg_defaults
    from src.loss import SLiM_Loss
    from experiments.lowtex_pro5000_hr_aug.losses.sparse_matching_loss import native_focal
    cfg=get_cfg_defaults().LOSS
    loss=SLiM_Loss(cfg)
    features=torch.Generator().manual_seed(42)
    a=torch.randn(8,5,generator=features,requires_grad=True)
    b=torch.randn(8,5,generator=features,requires_grad=True)
    t=torch.tensor(.37,requires_grad=True)
    i=torch.tensor([1,2,3,4,6])
    j=torch.tensor([5,2,4,1,0])
    mask=torch.tensor([True]*7+[False])
    gt=torch.zeros(1,8,8)
    gt[0,i,j]=1.
    sim=((a@b.T)/(5*(t+1e-8))).masked_fill(~(mask[:,None]&mask[None]),-1e9)
    # Native SLiM receives [batch, Hc, Wc] masks and flattens only the spatial axes.
    grid_mask=mask.reshape(1,2,4)
    d=dict(sim_matrix=sim[None],conf_matrix_gt=gt,mask0=grid_mask,mask1=grid_mask)
    torch.manual_seed(91)
    expected=loss.compute_coarse_loss(d)
    actual=native_focal(a,b,mask,mask,i,j,t,loss,2,torch.Generator().manual_seed(91))
    torch.testing.assert_close(actual,expected)
    for got,want in zip(torch.autograd.grad(actual,(a,b,t)),torch.autograd.grad(expected,(a,b,t))):
        torch.testing.assert_close(got,want,atol=3e-5,rtol=2e-5)
