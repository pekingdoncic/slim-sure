import json, math, unittest
import numpy as np
import torch
from src.slim import SLiM
from configs.cfg_outdoor_train import get_cfg_defaults
from experiments.lowtex_e2e_v1.screening.data import CACHE,EXP,N,OUT,U,Rows,read
from experiments.lowtex_e2e_v1.screening.model import BalancedLoss,CropHead
from experiments.lowtex_e2e_v1.screening.train import Coverage

class ScreeningTests(unittest.TestCase):
    def test_full_schedule_and_undirected_population(self):
        order=np.load(CACHE/"order_66.npy",mmap_mode="r")
        counts=np.bincount(order,minlength=N)
        self.assertEqual(len(counts),N);self.assertTrue(np.all(counts==1))
        self.assertEqual(math.ceil(N/4),2215669);self.assertEqual(N%4,1)
        data=Rows(N-1,N); ids,_=data[0]
        self.assertEqual(ids[0],N-1);self.assertEqual(ids[1],int(order[-1]))
        seen=np.zeros(U,dtype=bool)
        for b in data.bins:
            a=np.load(data.cache/(b["bin"]+".unordered.npy"),mmap_mode="r")
            self.assertEqual(len(a),b["legal_rows"]);seen[a]=True
        self.assertTrue(seen.all())
    def test_cursor_and_duplicate_roundtrip(self):
        x=Coverage(2)
        x.add([{"identity":(0,4,3,0)},{"identity":(1,5,3,1),"skip_reason":"invalid geometry"}])
        y=Coverage(2);y.restore(x.pack())
        self.assertEqual(x.status(),y.status())
        self.assertEqual(y.status()["repeated_unordered"],1)
        with self.assertRaises(RuntimeError):y.add([{"identity":(1,5,3,1)}])
    def test_fixed_dev(self):
        for name,n in (("megadepth",32),("texturepoor",24)):
            rows=read(OUT/(name+"_quarter_val.json"))
            self.assertEqual(len(rows),n);self.assertEqual(len(set(r["pair_id"] for r in rows)),n)
            if name=="texturepoor":self.assertEqual({r["scene"] for r in rows},{"1012","1026","1029"})
    def test_texture_loss_grad_and_empty(self):
        cfg=get_cfg_defaults();obj=BalancedLoss(cfg.LOSS);obj.coarse_percent=1
        for group in ("flat","edge"):
            x=torch.randn(1,16,16,requires_grad=True)
            image=torch.zeros(1,1,32,32)
            if group=="edge":image[:,:,:,16:]=torch.rand(1,1,32,16)
            mask=torch.ones(1,4,4,dtype=torch.bool)
            d=dict(sim_matrix=x,conf_matrix_gt=torch.eye(16)[None],image0=image,mask0=mask,mask1=mask,valid_mask0=torch.ones_like(image[:,0],dtype=torch.bool))
            loss=obj.compute_coarse_loss(d);loss.backward()
            self.assertTrue(torch.isfinite(loss) and x.grad.norm()>0)
            stat=d["balanced_stats"]
            self.assertEqual(stat["lt"]+stat["other"],16)
            if stat["lt"]:self.assertLessEqual(stat["lt_weight"]/stat["lt"],4/16+1e-8)
            d["conf_matrix_gt"].zero_();self.assertEqual(float(obj.compute_coarse_loss(d)),0.)
    def test_crop_border_and_grad(self):
        torch.manual_seed(66)
        for m in (False,True):
            h=CropHead(m)
            d={"b_idx_it":torch.zeros(3,dtype=torch.long)}
            pyramid=[]
            for i in (0,1):
                k=str(i);d["scale"+k]=torch.tensor([[2.,2.]])
                d["intermediate_coord_"+k]=torch.tensor([[4.,4.],[11.,7.],[0.,0.]])
                d["feat"+k+"_f"]=torch.randn(1,48,16,16,requires_grad=True)
                pyramid.append([None,torch.randn(1,96,4,4,requires_grad=True)])
            h(d,pyramid,SLiM._extract_feat_window_bilinear)
            self.assertEqual(tuple(d["all_offset_1"].shape),(1,3,2))
            torch.testing.assert_close(d["fine_coord_0"],d["intermediate_coord_0"])
            torch.testing.assert_close(d["fine_coord_1"]-d["intermediate_coord_1"],d["all_offset_1"][0]*6)
            d["fine_coord_1"].sum().backward()
            self.assertGreater(d["feat0_f"].grad.norm(),0)
            if m:self.assertGreater(pyramid[0][1].grad.norm(),0)
            else:self.assertIsNone(pyramid[0][1].grad)
        ramp=torch.arange(16,dtype=torch.float32).reshape(1,1,4,4)
        sub=SLiM._extract_feat_window_bilinear(ramp,torch.tensor([0,0]),torch.tensor([[1.5,1.5],[2.,1.5]]),1)
        torch.testing.assert_close(sub.flatten(),torch.tensor([5.,5.5]))

if __name__=="__main__":unittest.main()
