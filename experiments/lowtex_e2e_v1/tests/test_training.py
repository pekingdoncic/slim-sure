import copy
import unittest

import numpy as np
import torch

from experiments.lowtex_e2e_v1.training.optimization import UpdatePolicy
from experiments.lowtex_e2e_v1.training.sampling import Pool


class TrainingControlTest(unittest.TestCase):
    def test_sampler_covers_each_original_row(self):
        def fixture():
            p=Pool.__new__(Pool);p.sizes=np.array([3,8]);p.quota=np.array([9,8]);p.seed=66
            p.ends=np.r_[0,np.cumsum([sum((e+1)*p.quota//27-e*p.quota//27) for e in range(27)])]
            p.epoch=-1;p.order={0:np.array([1,2,0]),1:np.arange(8)[::-1]}
            p.bin_rows=lambda b: np.arange(2*p.sizes[b]).reshape(-1,2)
            return p
        seq=[fixture().at(i) for i in range(17)]
        p=fixture()
        self.assertEqual(seq,[p.at(i) for i in range(17)])
        self.assertEqual(sum(r[3] for r in seq),11)
        self.assertEqual(len({(bi,row) for bi,row,_,_ in seq}),11)
        self.assertEqual(seq[6:],[fixture().at(i) for i in range(6,17)])

    def test_warmup_and_resume(self):
        torch.manual_seed(66)
        m=torch.nn.Sequential(torch.nn.Linear(2,3),torch.nn.Linear(3,1))
        cfg=dict(backbone_lr=.001,new_head_lr=.005,new_head_warmup=.15,weight_decay=.01,milestones=[6,10,14,16])
        policy=UpdatePolicy(m,m[1],cfg,20,"end_to_end")
        data=torch.tensor([[1.,2.]])
        def update(n,net,p):
            p.apply(n);p.optimizer.zero_grad(set_to_none=True)
            net(data).square().mean().backward()
            p.optimizer.step();p.scheduler.step()
        frozen=m[0].weight.detach().clone()
        for i in range(3):update(i,m,policy)
        self.assertTrue(torch.equal(frozen,m[0].weight))
        p2net=copy.deepcopy(m);p2=UpdatePolicy(p2net,p2net[1],cfg,20,"end_to_end")
        p2.optimizer.load_state_dict(copy.deepcopy(policy.optimizer.state_dict()))
        p2.scheduler.load_state_dict(copy.deepcopy(policy.scheduler.state_dict()))
        for i in (3,4,5):
            update(i,m,policy);update(i,p2net,p2)
        for a,b in zip(m.parameters(),p2net.parameters()):torch.testing.assert_close(a,b,rtol=0,atol=0)
        self.assertFalse(torch.equal(frozen,m[0].weight))
        self.assertAlmostEqual(policy.optimizer.param_groups[1]["lr"]/policy.optimizer.param_groups[0]["lr"],5.)
        self.assertEqual(policy.scheduler.last_epoch,p2.scheduler.last_epoch)
