"""CPU contract tests for production evaluation helpers; never launch GPU or pilot."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

from reliability_v1.metrics import binary_metrics,ScoreAccumulator,paired_pose_summary,pose_auc
from reliability_v1.evaluation import (evaluate_reliability,freeze_pilot,coverage_metrics,
                                     canonical_protocol)
from reliability_v1.geometry_worker import prediction_identity


def pred(pid="p"):
    return {"pair_id":pid,"scene":"s","labels":np.array([1,0,1,-1]),
            "logits":np.array([2,-1,1,0]),"coarse_confidence":np.array([.8,.1,.5,.2]),
            "norm0":np.array([5,2,4,1]),"norm1":np.array([4,1,3,1]),
            "coarse_match_id":np.array([9,12,18,31]),
            "ignore_reasons":{"missing_depth":1},"low_texture_mask":np.array([1,1,0,0],bool)}


class MetricContracts(unittest.TestCase):
    def test_ties_and_constants(self):
        value=binary_metrics([0,1,0,1],[0,1,1,2])
        self.assertAlmostEqual(value["score_AUROC"],.875)
        self.assertAlmostEqual(value["score_AUPRC"],5/6)
        self.assertIsNone(binary_metrics([0,1],[.5,.5])["score_AUROC"])
        self.assertIsNone(binary_metrics([1,1],[0,1])["score_AUPRC"])
        self.assertIsNone(binary_metrics([],[])["score_AUROC"])

    def test_disk_matches_direct_and_pair_equal(self):
        with tempfile.TemporaryDirectory() as tmp:
            acc=ScoreAccumulator(Path(tmp)/"scores.sqlite")
            acc.add("a",pred("a"))
            b=pred("b")
            for k in ("labels","logits","coarse_confidence","norm0","norm1","coarse_match_id","low_texture_mask"):
                b[k]=b[k][:2]
            b["logits"]=np.array([-3,3])
            acc.add("b",b)
            s=acc.summary()
            reference=binary_metrics([1,0,1,1,0],[2,-1,1,-3,3])
            self.assertAlmostEqual(s["pooled"]["rankings"]["reliability"]["score_AUROC"],reference["score_AUROC"])
            self.assertAlmostEqual(s["pooled"]["rankings"]["reliability"]["score_AUPRC"],reference["score_AUPRC"])
            self.assertEqual(s["pair_count"],2)
            self.assertEqual(s["ignore_count"],1)
            self.assertNotAlmostEqual(s["pair_bce"],s["pooled"]["bce"])
            acc.close()

    def test_full_traversal_cache_and_identity(self):
        records=[{"pair_id":"p"+str(i),"scene":"s"+str(i%2)} for i in range(9)]
        calls=[]
        def predictor(record,mode):
            calls.append(record["pair_id"]);return pred(record["pair_id"])
        with tempfile.TemporaryDirectory() as tmp:
            s=evaluate_reliability(records,predictor,tmp,"checkpoint")
            self.assertEqual(len(calls),9)
            self.assertEqual(s["pair_count"],9)
            self.assertTrue(s["complete_pair_coverage"])
            evaluate_reliability(records,predictor,tmp,"checkpoint")
            self.assertEqual(len(calls),9)
            with self.assertRaises(ValueError):
                evaluate_reliability(records,predictor,tmp,"different")
            p=Path(tmp)/"score_metrics.json";p.write_text(p.read_text()+" ")
            with self.assertRaises(ValueError):
                evaluate_reliability(records,predictor,tmp,"checkpoint")

    def test_fixed_grid_coverage_not_match_count(self):
        p={"source_grid_valid":np.ones(4,bool),"source_grid_valid_gt":np.array([1,1,0,1],bool),
           "source_grid_low_texture":np.array([1,1,0,0],bool),"coarse_source_ids":np.array([0,0,1,3]),
           "labels":np.array([1,1,0,-1]),"final_source_input":np.array([[0,0],[1,1],[8,0]]),
           "source_grid_hw":(2,2),"coarse_stride_xy":(8,8)}
        m=coverage_metrics(p)
        self.assertEqual(m["all_region"]["correct_coarse_cells"],1)
        self.assertEqual(m["all_region"]["correct_coverage_all_valid_cells"],.25)
        self.assertEqual(m["local_low_texture"]["correct_coverage_all_valid_cells"],.5)
        self.assertEqual(m["all_region"]["final_covered_cells"],2)
        self.assertEqual(coverage_metrics({})["status"],"blocked")

    def test_pilot_scene_round_robin(self):
        records=[{"pair_id":p,"scene":s} for s,p in (("z","3"),("a","2"),("z","1"),("a","0"),("z","4"))]
        self.assertEqual([x["pair_id"] for x in freeze_pilot(records,4)],["0","1","2","3"])

    def test_pose_failure_denominator_and_bootstrap(self):
        self.assertLess(pose_auc([1,np.inf])["pose_AUC@5"],pose_auc([1])["pose_AUC@5"])
        a=np.array([[1,2,3,4,5],[np.inf]*5])
        result=paired_pose_summary(a,a,bootstrap=10)
        for item in result["paired_bootstrap"]["delta_A_texture_minus_A0"].values():
            self.assertEqual(item["mean"],0)
            self.assertEqual(item["ci95"],[0,0])
        with self.assertRaises(ValueError):paired_pose_summary(a,np.zeros((2,4)),bootstrap=1)

    def test_geometry_order_identity_and_canonical_sources(self):
        protocol=canonical_protocol()
        p={"points0_orig":np.array([[0.,0.],[1.,1.]]),"points1_orig":np.array([[0.,0.],[2.,2.]]),
           "K0":np.eye(3),"K1":np.eye(3),"T_0to1":np.eye(4)}
        q={k:v.copy() for k,v in p.items()};q["points0_orig"]=q["points0_orig"][::-1]
        self.assertNotEqual(prediction_identity(p,protocol),prediction_identity(q,protocol))


if __name__=="__main__":unittest.main()
