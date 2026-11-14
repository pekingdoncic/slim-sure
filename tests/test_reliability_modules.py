"""CPU contracts for production scoring/candidate code, not GPU acceptance."""
import subprocess
import types
import linecache
import sys
import unittest
import torch
from torch import nn
from src.slim import SLiM
from src.reliability import ReliabilityHead
from src.texture_budget import local_texture, allocate_candidates


def bare_model():
    m = SLiM.__new__(SLiM)
    nn.Module.__init__(m)
    m.capture_reliability = True
    m.reliability_enabled = False
    m.texture_budget_enabled = False
    m.texture_fraction = 0.2
    m.texture_threshold = 0.0
    m.max_coarse_matches = 6
    m.coarse_match_thres = 0.1
    m.coarse_temperature = nn.Parameter(torch.tensor(0.05))
    return m


def coarse_input(h=4, w=4):
    return {"hw0_c": (h,w), "hw1_c": (h,w), "coarse_scale": 8,
            "image0": torch.zeros(1,1,h*8,w*8), "image1": torch.zeros(1,1,h*8,w*8),
            "valid_mask0": torch.ones(1,h*8,w*8,dtype=torch.bool),
            "valid_mask1": torch.ones(1,h*8,w*8,dtype=torch.bool)}


class ReliabilityContracts(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2026)
        torch.set_num_threads(1)

    def test_head_empty_single_gradient_and_fp32(self):
        head = ReliabilityHead(192)
        self.assertEqual(sum(p.numel() for p in head.parameters()), 29185)
        for count in (0,1,7):
            d0,d1 = torch.randn(count,192), torch.randn(count,192)
            snapshot = d0.clone()
            with torch.autocast("cpu", dtype=torch.bfloat16):
                logits = head(d0,d1,torch.rand(count))
            self.assertEqual(logits.dtype, torch.float32)
            self.assertEqual(logits.shape,(count,))
            self.assertTrue(torch.equal(d0,snapshot))
            if count:
                head.zero_grad(set_to_none=True)
                nn.functional.binary_cross_entropy_with_logits(logits,torch.zeros_like(logits)).backward()
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters()))
                before=head.projection.weight.detach().clone()
                torch.optim.AdamW(head.parameters(),lr=1e-3).step()
                self.assertFalse(torch.equal(before,head.projection.weight))

    def test_actual_native_coarse_method_preserved(self):
        source=subprocess.check_output(["git","show","HEAD:src/slim.py"],text=True)
        baseline=types.ModuleType("src._engineering_baseline")
        baseline.__package__="src"
        linecache.cache["baseline/src/slim.py"]=(len(source),None,source.splitlines(True),"baseline/src/slim.py")
        sys.modules[baseline.__name__]=baseline
        exec(compile(source,"baseline/src/slim.py","exec"),baseline.__dict__)
        for shape,zero in (((4,4),False),((1,1),False),((4,4),True)):
            h,w=shape
            a,b=torch.randn(1,8,h,w),torch.randn(1,8,h,w)
            if zero: a.zero_(); b.zero_()
            native,modified=coarse_input(h,w),coarse_input(h,w)
            m=bare_model()
            baseline.SLiM._coarse_match_norm_clue(m,native,a,b)
            disabled=coarse_input(h,w)
            m.capture_reliability=False
            m._coarse_match_norm_clue(disabled,a,b)
            self.assertEqual(set(disabled),set(native))
            self.assertIsNone(getattr(m,"reliability_head",None))
            m.capture_reliability=True
            m._coarse_match_norm_clue(modified,a,b)
            for key in ("b_idx_c","i_idx_c","j_idx_c","coarse_coord_0","coarse_coord_1"):
                self.assertTrue(torch.equal(native[key],modified[key]),key)
            ids=modified["coarse_match_id"]
            self.assertEqual(ids.unique().numel(),ids.numel())
            self.assertTrue(torch.equal(modified["coarse_row_id"],torch.arange(ids.numel())))
            self.assertEqual(modified["coarse_descriptor0"].shape,(ids.numel(),8))
            self.assertEqual(modified["coarse_confidence"].shape,(ids.numel(),))
            # Baseline confidence is observed from the actual native function frame.
            from reliability_v1.engineering_checks import _capture_native_confidence
            observer=bare_model()
            observer._coarse_match_norm_clue=types.MethodType(baseline.SLiM._coarse_match_norm_clue,observer)
            _capture_native_confidence(observer)
            observed=coarse_input(h,w);observed.update(batch_size=1,feat0_c=a)
            observer._coarse_match_norm_clue(observed,a,b)
            self.assertTrue(torch.equal(observed["coarse_confidence"],modified["coarse_confidence"]))
            if ids.numel():
                self.assertTrue(torch.equal(modified["coarse_descriptor0"],a.flatten(2).transpose(1,2)[modified["b_idx_c"],modified["i_idx_c"]]))
            m.texture_budget_enabled=True; m.texture_fraction=0.0
            zero_texture=coarse_input(h,w)
            m._coarse_match_norm_clue(zero_texture,a,b)
            for key in ("b_idx_c","i_idx_c","j_idx_c","coarse_coord_0","coarse_coord_1","coarse_confidence"):
                self.assertTrue(torch.equal(modified[key],zero_texture[key]),key)
            for key in ("candidate_ids0","candidate_ids1"):
                self.assertTrue(torch.equal(modified[key][0],zero_texture[key][0]))
            m.texture_fraction=0.2
            positive_texture=coarse_input(h,w)
            m._coarse_match_norm_clue(positive_texture,a,b)
            self.assertEqual(positive_texture["candidate_counts"],modified["candidate_counts"])
            self.assertEqual(positive_texture["correlation_shapes"],modified["correlation_shapes"])

    def test_capture_rejects_overflow_instead_of_empty_matches(self):
        model=bare_model()
        huge=torch.full((1,2,1,1),1e19,dtype=torch.float32)
        self.assertTrue(torch.isfinite(huge).all())
        scaled=huge/(2**.5)
        self.assertTrue(torch.isfinite(torch.linalg.vector_norm(scaled,dim=1)).all())
        with self.assertRaisesRegex(FloatingPointError,"similarity before softmax"):
            model._coarse_match_norm_clue(coarse_input(1,1),huge,huge)
        # The disabled branch retains original behavior and output fields.
        model.capture_reliability=False
        unchanged=coarse_input(1,1)
        model._coarse_match_norm_clue(unchanged,huge,huge)
        self.assertEqual(unchanged["b_idx_c"].numel(),0)
        model.capture_reliability=True
        nonfinite=torch.full((1,2,1,1),float("nan"))
        with self.assertRaisesRegex(FloatingPointError,"descriptor norm"):
            model._coarse_match_norm_clue(coarse_input(1,1),nonfinite,nonfinite)

    def test_empty_and_single_parent_through_native_fine_and_rru(self):
        from src.recurrent_refinement import RecurrentRefinementUnit
        for count in (0,1):
            model=bare_model()
            model.fine_temperature=nn.Parameter(torch.tensor(0.005))
            model.fine_match_thres=.03
            model.max_fine_matches=4
            model.refine_iters=2
            model.refine_lookup_radius=3
            model.lookup_window_size=6
            model.recurrent_refine_unit=RecurrentRefinementUnit(input_dim=4,hidden_dim=4).eval()
            idx=torch.zeros(count,dtype=torch.long)
            data={"b_idx_c":idx,"i_idx_c":idx,"j_idx_c":idx,
                  "coarse_coord_0":torch.full((count,2),8.),"coarse_coord_1":torch.full((count,2),8.),
                  "coarse_scale":8,"fine_scale":1,"feat0_f":torch.randn(1,4,32,32),"feat1_f":torch.randn(1,4,32,32),
                  "coarse_match_id":torch.arange(count),"coarse_descriptor0":torch.randn(count,8),
                  "coarse_descriptor1":torch.randn(count,8),"coarse_confidence":torch.ones(count)}
            with torch.no_grad():
                model._fine_correlation(data)
                model._get_fine_coord_test(data)
                model._refinement(data)
                ReliabilityHead(8).score_data(data)
            self.assertTrue(torch.isfinite(data["fine_coord_0"]).all())
            self.assertTrue(torch.isfinite(data["fine_coord_1"]).all())
            self.assertEqual(data["fine_parent_id"].shape,data["m_idx_it"].shape)
            if count==0:
                self.assertEqual(data["fine_coord_0"].shape,(0,2))
                self.assertEqual(data["fine_parent_reliability_score"].numel(),0)
            else:
                self.assertGreater(data["fine_coord_0"].shape[0],0)
                self.assertTrue((data["fine_parent_id"]==0).all())

    def test_child_parent_identity(self):
        head=ReliabilityHead(8)
        data={"coarse_descriptor0":torch.randn(3,8),"coarse_descriptor1":torch.randn(3,8),
              "coarse_confidence":torch.rand(3),"coarse_match_id":torch.tensor([11,45,72]),
              "m_idx_it":torch.tensor([2,0,2,1]),"mconf":torch.tensor([.1,.2,.3,.4])}
        original=data["mconf"].clone()
        head.score_data(data)
        self.assertTrue(torch.equal(data["fine_parent_id"],torch.tensor([72,11,72,45])))
        self.assertTrue(torch.equal(data["fine_parent_reliability_score"],data["reliability_score"][data["m_idx_it"]]))
        self.assertTrue(torch.equal(data["mconf"],original))

    def test_texture_boundaries_padding_and_scale(self):
        mask=torch.zeros(1,32,32,dtype=torch.bool); mask[:,:22,:19]=True
        image=torch.zeros(1,1,32,32); image[:,0][mask]=.7
        texture,count=local_texture(image,mask,(4,4))
        self.assertTrue(torch.allclose(texture,torch.zeros_like(texture),atol=1e-9))
        self.assertEqual(count[0,0].item(),64)
        self.assertEqual(count[0,15].item(),18)
        ramp=torch.arange(32).float().view(1,1,1,32).expand(1,1,32,32)/31
        texture,count=local_texture(ramp,torch.ones(1,32,32,dtype=torch.bool),(4,4))
        self.assertAlmostEqual(texture[0,10].item(),1/31**2,places=7)
        self.assertEqual(count[0,10].item(),256)
        empty,count=local_texture(image,torch.zeros_like(mask),(4,4))
        self.assertTrue(torch.isnan(empty).all()); self.assertEqual(count.sum().item(),0)

    def test_candidate_budget_zero_ties_and_refill(self):
        norms=torch.arange(10,0,-1,dtype=torch.float32)
        native=torch.tensor([0,1,2,3,4]); legal=torch.ones(10,dtype=torch.bool)
        texture=torch.ones(10); texture[9]=0
        selected=allocate_candidates(native,norms,legal,texture,0.0,.2)
        self.assertTrue(torch.equal(selected,torch.tensor([0,1,2,3,9])))
        permuted=native.flip(0)
        self.assertTrue(torch.equal(allocate_candidates(permuted,norms,legal,texture,None,0.0),permuted))
        refill=allocate_candidates(native,norms,legal,texture,-1,.2)
        self.assertTrue(torch.equal(refill,native))
        tied=allocate_candidates(native,torch.ones(10),legal,texture,-1,.2)
        self.assertTrue(torch.equal(tied,native))
        self.assertEqual(allocate_candidates(native[:0],norms,legal,texture,None,.2).numel(),0)
        self.assertTrue(torch.equal(allocate_candidates(native[:1],norms,legal,texture,0,.2),native[:1]))


if __name__=="__main__":
    unittest.main()
