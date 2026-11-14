"""Coordinate/visibility contracts against the actual production label functions."""
import importlib.util
import json
from pathlib import Path
import sys
import unittest
import tempfile
import contextlib
import io
import gc
from unittest.mock import patch
import numpy as np
import torch
import cv2
import h5py

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import reliability_data as rd
from reliability_data import project_visible, label_parents, plan_geometry, ManifestDataset, verify_frozen_file, sha256_file, load_pair

def make_data():
    # Anisotropic scale explicitly exercises independent x/y stride and mapping.
    plan={"orig_hw":[40,60],"resized_hw":[20,20],"input_hw":[24,24],"pad_ltrb":[0,0,4,4],
          "input_to_orig":[[3.,0.,0.],[0.,2.,0.],[0.,0.,1.]],
          "orig_to_input":[[1/3,0.,0.],[0.,.5,0.],[0.,0.,1.]],
          "scale_input_to_orig_xy":[3.,2.]}
    return {"image0":torch.zeros(1,1,24,24),"image1":torch.zeros(1,1,24,24),
      "hw1_c":(3,3),"pair_id":"contract","direction":"0to1",
      "_gt":{"plan0":plan,"plan1":plan,"depth0":np.full((40,60),5.),
             "depth1":np.full((40,60),5.),"K0":np.eye(3),"K1":np.eye(3),
             "T_0to1":np.eye(4),"depth_absolute_tolerance":.01,"depth_relative_tolerance":.05}}

def parents(data,p0,p1):
    data["coarse_coord_0"]=torch.tensor(p0,dtype=torch.float32).reshape(-1,2)
    data["coarse_coord_1"]=torch.tensor(p1,dtype=torch.float32).reshape(-1,2)
    data["coarse_match_id"]=torch.arange(len(p0),dtype=torch.int64)+100
    return data

class Contracts(unittest.TestCase):
    def test_identity_projection_and_no_half_stride(self):
        d=make_data()
        out=project_visible([[0,0],[8,8],[16,16]],d)
        np.testing.assert_allclose(out["points1_gt_input"],[[0,0],[8,8],[16,16]])
        self.assertTrue(out["valid_gt"].all())
        lab=label_parents(parents(d,[[0,0],[8,8]],[[0,0],[8,8]]))
        self.assertEqual(lab["labels"].tolist(),[1,1])
    def test_inclusive_per_axis_half_stride(self):
        d=make_data()
        lab=label_parents(parents(d,[[8,8],[8,8],[8,8]],[[12,12],[12.001,8],[8,3.999]]))
        self.assertEqual(lab["labels"].tolist(),[1,0,0])
        np.testing.assert_array_equal(lab["stride_xy"],[8,8])
    def test_empty_single_and_parent_identity(self):
        d=make_data()
        self.assertEqual(len(label_parents(parents(d,[],[]))["labels"]),0)
        d=parents(d,[[8,8]],[[8,8]])
        self.assertEqual(label_parents(d)["raw_coarse_match_id"].tolist(),[100])
        d=parents(d,[[8,8],[8,8]],[[8,8],[8,8]])
        d["coarse_match_id"][:]=100
        with self.assertRaises(ValueError):label_parents(d)
    def test_ignore_not_negative(self):
        d=make_data()
        d["_gt"]["depth0"][16,24]=0
        lab=label_parents(parents(d,[[8,8],[24,0]],[[8,8],[0,0]]))
        self.assertEqual(lab["labels"].tolist(),[-1,-1])
        self.assertEqual(lab["reasons"].tolist(),[2,1])
        d=make_data();d["_gt"]["depth1"][16,24]=0
        self.assertEqual(label_parents(parents(d,[[8,8]],[[8,8]]))["reasons"].tolist(),[5])
        d=make_data();d["_gt"]["depth1"][16,24]=10
        self.assertEqual(label_parents(parents(d,[[8,8]],[[8,8]]))["reasons"].tolist(),[6])
        d=make_data()
        self.assertEqual(label_parents(parents(d,[[8,8]],[[20,8]]))["reasons"].tolist(),[7])
    def test_true_pose_direction_and_depth_shape_scale(self):
        d=make_data();d["_gt"]["T_0to1"][0,3]=15.
        p=project_visible([[8,8]],d)
        np.testing.assert_allclose(p["points1_gt_input"],[[9,8]])
        d=make_data()
        d["_gt"]["depth0"]=np.full((20,30),5.)
        d["_gt"]["depth1"]=np.full((20,30),5.)
        np.testing.assert_allclose(project_visible([[8,8]],d)["points1_gt_input"],[[8,8]])
    def test_original_canonical_production_agreement(self):
        sys.dont_write_bytecode=True
        sys.path.insert(0,"/home/bupt/projects")
        from hybrid_match.common.gt_coarse import project_visible as reference
        d=make_data();g=d["_gt"];points=np.array([[0,0],[8,8],[19,19],[20,0],[-1,0]],float)
        actual=project_visible(points,d)
        expected=reference(points,g["plan0"],g["plan1"],g["depth0"],g["depth1"],g["K0"],g["K1"],g["T_0to1"],g)
        for k in ("points1_gt_input","valid_gt","invalid_reason"):
            np.testing.assert_allclose(actual[k],expected[k],equal_nan=True)
    def test_frozen_file_rejects_mutation(self):
        with tempfile.TemporaryDirectory(prefix="slim_data_contract_") as temp:
            p=Path(temp)/"file"
            p.write_bytes(b"original")
            expected=sha256_file(p)
            verify_frozen_file(p,expected)
            verify_frozen_file(p,expected)
            p.write_bytes(b"changed-input")
            with self.assertRaises(RuntimeError):verify_frozen_file(p,expected)

    def test_loader_native_input_units_and_image_only_io(self):
        with tempfile.TemporaryDirectory(prefix="slim_loader_contract_") as temp:
            base=Path(temp)
            image=base/"image.png";depth=base/"depth.h5"
            cv2.imwrite(str(image),np.full((20,30),100,np.uint8))
            with h5py.File(depth,"w") as h:h["depth"]=np.full((20,30),5.)
            record={"pair_id":"fixture","scene":"fixture","file_sha256":{}}
            for side in (0,1):
                record[f"image{side}"]=str(image);record[f"depth{side}"]=str(depth)
                record[f"K{side}"]=np.eye(3).tolist();record[f"pose{side}"]=np.eye(4).tolist()
                record["file_sha256"][f"image{side}"]=sha256_file(image)
                record["file_sha256"][f"depth{side}"]=sha256_file(depth)
            loaded=load_pair(record,device="cpu")
            self.assertNotIn("scale0",loaded);self.assertNotIn("scale1",loaded)
            self.assertEqual(loaded["model_coordinate_space"],"input")
            self.assertIn("input_to_orig_scale0",loaded)
            self.assertEqual(tuple(loaded["image0"].shape),(1,1,832,832))
            self.assertEqual(tuple(loaded["valid_mask0"].shape),(1,832,832))
            self.assertEqual(tuple(loaded["mask0"].shape),(1,104,104))
            loaded["hw1_c"]=(104,104)
            out=label_parents(parents(loaded,[[8,8]],[[8,8]]))
            self.assertEqual(out["labels"].tolist(),[1])
            depth.unlink()
            inference=load_pair(record,device="cpu",include_gt=False)
            self.assertNotIn("_gt",inference)
            torch.testing.assert_close(inference["image0"],loaded["image0"])
            loaded["scale0"]=loaded["input_to_orig_scale0"]
            with self.assertRaises(ValueError):label_parents(loaded)

    def test_complete_manifest_and_index_resume(self):
        with tempfile.TemporaryDirectory(prefix="slim_manifest_contract_") as temp:
            root=Path(temp);base=root/"data/megadepth"
            native=base/"index/scene_info_0.1_0.7"
            mapped=base/"index/scene_info_0.1_0.7_no_sfm"
            lists=base/"index/trainvaltest_list"
            for p in (native,mapped,lists):p.mkdir(parents=True)
            stems=[]
            for scene in ("9001","9002","0015"):
                stem=scene+"_0.1_0.3";stems.append(stem)
                img=[];dep=[];original=[]
                for key in ("a","b"):
                    rel=Path("phoenix/S6/zl548/MegaDepth_v1")/scene/"dense0"
                    ip=base/"train"/rel/"imgs"/(key+".jpg")
                    dp=base/"train"/rel/"depths"/(key+".h5")
                    ip.parent.mkdir(parents=True,exist_ok=True);dp.parent.mkdir(parents=True,exist_ok=True)
                    cv2.imwrite(str(ip),np.full((20,30),100,np.uint8))
                    with h5py.File(dp,"w") as h:h["depth"]=np.full((20,30),5.)
                    img.append(str(ip.relative_to(base/"train")));dep.append(str(dp.relative_to(base/"train")))
                    original.append("Undistorted_SfM/"+scene+"/images/"+key+".png")
                pairs=np.empty((3,3),dtype=object)
                pairs[0]=[np.array([0,1]),.5,np.zeros(4)]
                pairs[1]=[np.array([1,0]),.5,np.zeros(4)]
                pairs[2]=[np.array([0,1]),.5,np.zeros(4)]
                spec=dict(image_paths=np.array(original,dtype=object),depth_paths=np.array(dep,dtype=object),
                          intrinsics=np.stack([np.eye(3)]*2),poses=np.stack([np.eye(4)]*2),
                          pair_infos=pairs)
                np.savez(native/(stem+".npz"),**spec)
                spec["image_paths"]=np.array(img,dtype=object);np.savez(mapped/(stem+".npz"),**spec)
            (lists/"train_list.txt").write_text("\n".join(stems))
            run=root/"runs/run";run.mkdir(parents=True)
            config={"repo_root":str(root)}
            original_hash=rd.sha256_file
            def interrupted(path):
                if Path(path).name=="9002_0.1_0.3.npz":raise RuntimeError("test interruption after index checkpoint")
                return original_hash(path)
            with patch.object(rd,"test_scene_exclusions",return_value=({"0015"},[])):
                with contextlib.redirect_stdout(io.StringIO()),patch.object(rd,"sha256_file",side_effect=interrupted):
                    with self.assertRaisesRegex(RuntimeError,"test interruption"):rd.audit_data(config,run)
                gc.collect()
                config["resume_audit"]=True
                with contextlib.redirect_stdout(io.StringIO()):
                    summary=rd.audit_data(config,run)
            self.assertEqual(summary["counts"]["candidate_pairs"],9)
            self.assertEqual(summary["counts"]["excluded_test_scene"],3)
            self.assertEqual(summary["counts"]["excluded_duplicate_directed_pair"],2)
            self.assertEqual(summary["train_count"],2);self.assertEqual(summary["val_count"],2)
            self.assertFalse(set(summary["train_scenes"])&set(summary["val_scenes"]))
            train=ManifestDataset(run/"train_manifest.jsonl")
            try:
                sample=train[0]
                self.assertEqual(len(sample["file_sha256"]),4)
                loaded=load_pair(sample,device="cpu")
                self.assertNotIn("scale0",loaded)
                self.assertEqual(train.scene_round_robin_indices(8),[0,1])
            finally:train.close()

    def test_production_parent_anchor_guard(self):
        d=make_data()
        d["hw0_c"]=(3,3);d["mask0"]=torch.ones(1,3,3,dtype=torch.bool)
        d["mask1"]=torch.ones(1,3,3,dtype=torch.bool)
        parents(d,[[8,8]],[[8,8]])
        d["i_idx_c"]=torch.tensor([4]);d["j_idx_c"]=torch.tensor([4])
        d["b_idx_c"]=torch.tensor([0]);d["coarse_match_id"]=torch.tensor([4*9+4])
        self.assertTrue(label_parents(d)["contract_checked"])
        d["coarse_coord_0"]+=4
        with self.assertRaisesRegex(ValueError,"input grid anchors"):label_parents(d)
        d["coarse_coord_0"]-=4
        d["mask1"][0,1,1]=False
        with self.assertRaisesRegex(ValueError,"padding"):label_parents(d)
        d["mask1"][0,1,1]=True
        d["coarse_match_id"]+=1
        with self.assertRaisesRegex(ValueError,"raw parent identity"):label_parents(d)

    def test_r832_geometry_rounding(self):
        p=plan_geometry((1001,1500))
        self.assertEqual(p["input_hw"],[832,832])
        self.assertEqual(p["resized_hw"],[555,832])
        self.assertEqual(p["scale_input_to_orig_xy"],[1500/832,1001/555])

if __name__=="__main__":
    unittest.main(verbosity=2)
