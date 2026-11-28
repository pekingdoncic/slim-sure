import pytest
import torch
from experiments.lowtex_pro5000_hr_aug.models.ccfi import CCFI
from experiments.lowtex_pro5000_hr_aug.models.hypothesis_graph import HypothesisGraph
from experiments.lowtex_pro5000_hr_aug.models.sparse_dual_softmax import dual_logp
from experiments.lowtex_pro5000_hr_aug.losses.hard_negative_ranking import ranking
from experiments.lowtex_pro5000_hr_aug.utils.run_io import config


def test_ccfi_masks_identity_initialization_and_gradients():
    torch.manual_seed(9)
    cfg=config("P2")
    m=CCFI(16,cfg)
    a,b=[torch.randn(2,16,8,12,requires_grad=True) for _ in range(2)]
    ma=torch.ones(2,8,12,dtype=torch.bool)
    mb=ma.clone()
    mb[1]=False
    aa,bb,s,x,y=m(a,b,ma,mb)
    assert aa.shape==a.shape and s.shape==(2,6,6) and torch.isfinite(aa).all()
    assert (aa-a).norm()<.12*a.norm()
    torch.testing.assert_close(aa[1],a[1])
    (aa[0].square().mean()+bb[0].mean()).backward()
    for p in (a,b,m.query.weight,m.out.weight,m.gate.weight):
        assert p.grad is not None and p.grad.abs().sum()>0


def test_graph_topm_scatter_background_gradients_and_degree():
    torch.manual_seed(83)
    cfg=config("P4")
    cfg.update(graph_chunk_size=11,graph_max_degree=3,graph_layers=2)
    net=HypothesisGraph(12,cfg)
    n=16
    a,b=[torch.randn(n,12,requires_grad=True) for _ in range(2)]
    x=(a@b.T).flatten()
    keys=torch.arange(n*n,dtype=torch.int32)
    block=torch.zeros(1,1,requires_grad=True)
    out,raw,table,added=net(a,b,keys,x,block,(4,4))
    assert raw.shape==(16,5) and net.src.out_features==net.dst.out_features==64 and not added
    torch.testing.assert_close(out,x)
    torch.testing.assert_close(dual_logp(out,keys,n,n),dual_logp(x,keys,n,n))
    with torch.no_grad():
        net.out.weight.fill_(.01)
    out,*_=net(a,b,keys,x,block,(4,4))
    mask=torch.ones(len(x),dtype=torch.bool)
    mask[raw.flatten()]=False
    torch.testing.assert_close(out[mask],x[mask])
    out.sum().backward()
    assert net.embed[0].weight.grad.abs().sum()>0 and net.layers[0][0].weight.grad.abs().sum()>0
    net.eval()
    with pytest.raises(ValueError,match="GT"):
        net(a,b,keys,x,block,(4,4),injection=1.)
    out2,*_=net(a,b,keys,x,block,(4,4))
    cfg["graph_chunk_size"]=3
    out3,*_=net(a,b,keys,x,block,(4,4))
    torch.testing.assert_close(out2,out3)


def test_ranking_never_labels_unknown_or_occluded_negatives():
    cfg=config("P4")
    ids=torch.arange(64)
    gt=dict(i=ids[:1],j=ids[:1],back=ids,reverse=torch.zeros(64,dtype=torch.bool))
    keys=torch.tensor([0,1,63],dtype=torch.int32)
    x=torch.tensor([.3,2.,1.],requires_grad=True)
    table=torch.full((64,3),-1,dtype=torch.long)
    table[0]=torch.arange(3)
    loss,n=ranking(x,keys,table,gt,(8,8),cfg)
    assert n==0 and loss.item()==0
    gt["reverse"][63]=True
    loss,n=ranking(x,keys,table,gt,(8,8),cfg)
    assert n==1
    loss.backward()
    torch.testing.assert_close(x.grad,torch.tensor([-1.,0.,1.]))


def test_topm_replacement_keeps_predicted_recall_and_graph_degree():
    cfg=config("P4")
    cfg.update(hypothesis_topm=2,source_neighbor_degree=1,graph_max_degree=1,graph_layers=1)
    model=HypothesisGraph(4,cfg)
    a,b=torch.randn(16,4),torch.randn(16,4)
    scores=torch.arange(16,dtype=torch.float32).repeat(16)
    keys=torch.arange(256,dtype=torch.int32)
    gt=dict(i=torch.tensor([8]),j=torch.tensor([0]))
    result,raw,used,count=model(a,b,keys,scores,torch.zeros(1,1),(4,4),gt,1.)
    assert count==1 and 128 not in raw[8].tolist() and 128 in used[8].tolist()
    torch.testing.assert_close(result,scores)
    # With one left neighbor, changing a distant source descriptor cannot affect row 8.
    with torch.no_grad():
        model.out.weight.fill_(.01)
        model.eval()
        first=model(a,b,keys,scores,torch.zeros(1,1),(4,4))[0]
        other=a.clone()
        other[15]+=100
        second=model(other,b,keys,scores,torch.zeros(1,1),(4,4))[0]
        torch.testing.assert_close(first[128:144],second[128:144])
