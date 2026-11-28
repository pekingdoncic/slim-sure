import json
import unittest

import numpy as np

from experiments.lowtex_e2e_v1.evaluation.texturepoor_sfm.dataset import EXP, TEST, VAL, preprocess
from experiments.lowtex_e2e_v1.evaluation.texturepoor_sfm.geometry import relative, transformed_intrinsics
from experiments.lowtex_e2e_v1.evaluation.texturepoor_sfm.metrics import auc,evaluate_pair


class GeometryTest(unittest.TestCase):
    def test_relative_and_intrinsics(self):
        p=np.eye(4);p[:3,3]=[0.5,0.1,0.0]
        q=np.eye(4);q[:3,:3]=[[0,-1,0],[1,0,0],[0,0,1]]
        q[:3,3]=[0.2,0,0]
        x=np.array([0.4,-0.2,3,1])
        np.testing.assert_allclose(relative(p,q)@(p@x),q@x)
        k=np.array([[151.,0.,23.],[0.,143.,19.],[0.,0.,1.]])
        gray=np.ones((75,100),np.uint8)
        for side in (64,96):
            image,mask,scale=preprocess(gray,side)
            self.assertEqual(image.shape,(side,side))
            self.assertEqual(mask.sum(),side*round(0.75*side))
            projected=k@x[:3];projected=projected[:2]/projected[2]
            new=transformed_intrinsics(k,*np.reciprocal(scale),3,8)@x[:3]
            np.testing.assert_allclose((new[:2]/new[2]-[3,8])*scale,projected,rtol=1e-6)

    def test_pose_failure(self):
        rng=np.random.default_rng(66); xyz=rng.uniform([-1,-1,3],[1,1,8],size=(60,3))
        k=np.array([[500.,0,220],[0,500,210],[0,0,1]])
        t=np.eye(4);t[0,3]=0.3
        def proj(x):
            h=x@k.T
            return h[:,:2]/h[:,2:]
        row=dict(pair_id="projection",scene="synthetic",K0=k,K1=k,T_0to1=t,wh0=[500,500],wh1=[500,500])
        r=evaluate_pair(row,proj(xyz),proj(xyz+t[:3,3]))
        self.assertEqual(r["epi_correct"],60)
        self.assertLess(r["pose_deg"],0.01)
        self.assertIsNone(evaluate_pair(row,[],[])["pose_deg"])
        self.assertEqual(auc([0,None])["auc@5"],50.)

    def test_split(self):
        for split,expected in [("val",VAL),("test",TEST)]:
            rows=json.loads((EXP/"manifests"/f"texturepoor_{split}_pairs.json").read_text())
            self.assertEqual({r["scene"] for r in rows},expected)
            pairs=[(r["scene"],tuple(sorted(r["pair_id"].split("/")[1:]))) for r in rows]
            self.assertEqual(len(pairs),len(set(pairs)))
            self.assertTrue(all(a!=b for _,(a,b) in pairs))
        self.assertFalse(VAL & TEST)


if __name__=="__main__":
    unittest.main()
