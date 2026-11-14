"""Real transactional fixture checks for the bounded audit implementation."""
import contextlib
import errno
import io
import json
import pickle
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch
import cv2
import h5py
import numpy as np
import reliability_data as rd


def fixture(root,raw_pickle=False):
    base=root/"data/megadepth";native=base/"index/scene_info_0.1_0.7"
    mapped=base/"index/scene_info_0.1_0.7_no_sfm";lists=base/"index/trainvaltest_list"
    for p in (native,mapped,lists):p.mkdir(parents=True)
    stems=[]
    for scene in ("9001","9002","0015"):
        stem=scene+"_0.1_0.3";stems.append(stem);images=[];depths=[];original=[]
        for key in ("a","b"):
            rel=Path("phoenix/S6/zl548/MegaDepth_v1")/scene/"dense0"
            image=base/"train"/rel/"imgs"/(key+".jpg");depth=base/"train"/rel/"depths"/(key+".h5")
            image.parent.mkdir(parents=True,exist_ok=True);depth.parent.mkdir(parents=True,exist_ok=True)
            cv2.imwrite(str(image),np.full((8,12),80,np.uint8))
            with h5py.File(depth,"w") as h:h["depth"]=np.ones((8,12),np.float32)
            images.append(str(image.relative_to(base/"train")));depths.append(str(depth.relative_to(base/"train")))
            original.append("Undistorted_SfM/"+scene+"/images/"+key+".png")
        pairs=np.empty((4,3),dtype=object)
        for row,(i,j,o) in enumerate(((0,1,.5),(1,0,.5),(0,1,.5),(0,1,0.))):
            pairs[row]=[np.array([i,j]),o,np.zeros(4)]
        fields=dict(image_paths=np.array(original,dtype=object),depth_paths=np.array(depths,dtype=object),
            intrinsics=np.stack([np.eye(3)]*2),poses=np.stack([np.eye(4)]*2),pair_infos=pairs,
            unused_points3D=np.zeros((50,3)))
        def save(path,entry):
            if raw_pickle:
                with path.open("wb") as handle:pickle.dump(entry,handle,protocol=4)
            else:np.savez(path,**entry)
        save(native/(stem+".npz"),fields)
        fields["image_paths"]=np.array(images,dtype=object);save(mapped/(stem+".npz"),fields)
    (lists/"train_list.txt").write_text("\n".join(stems))
    run=root/"runs/run";run.mkdir(parents=True)
    return run,{"repo_root":str(root),"audit_pair_batch_size":2}


class BoundedAudit(unittest.TestCase):
    def test_partial_batch_resume_and_one_index_pilot(self):
        with tempfile.TemporaryDirectory(prefix="slim_bounded_") as directory:
            root=Path(directory);run,cfg=fixture(root)
            original=rd.atomic_json;interrupted=[False]
            def checkpoint(path,value):
                original(path,value)
                if Path(path).name=="data_audit_progress.json" and value.get("index")=="9001_0.1_0.3" and value.get("cursor")==2 and not interrupted[0]:
                    interrupted[0]=True;raise RuntimeError("injected post-commit stop")
            with patch.object(rd,"test_scene_exclusions",return_value=({"0015"},[])),contextlib.redirect_stdout(io.StringIO()):
                with patch.object(rd,"atomic_json",side_effect=checkpoint):
                    with self.assertRaisesRegex(RuntimeError,"post-commit"):rd.audit_data(cfg,run)
                db=sqlite3.connect(run/"data_audit.incomplete.sqlite")
                self.assertEqual(db.execute("SELECT COUNT(*) FROM audit_chunks").fetchone()[0],0)
                self.assertEqual(db.execute("SELECT cursor FROM audit_work").fetchone()[0],2)
                self.assertEqual(db.execute("SELECT COUNT(*) FROM pairs").fetchone()[0],2);db.close()
                cfg.update(resume_audit=True,max_new_indices=1)
                first=rd.audit_data(cfg,run)
                self.assertEqual(first["status"],"incomplete");self.assertEqual(first["committed_indices"],1)
                self.assertFalse((run/"train_manifest.jsonl").exists())
                original_chunk=(run/"candidate_audit_chunks/9001_0.1_0.3.jsonl.gz").read_bytes()
                second=rd.audit_data(cfg,run)
                self.assertEqual(second["committed_indices"],2)
                self.assertEqual(original_chunk,(run/"candidate_audit_chunks/9001_0.1_0.3.jsonl.gz").read_bytes())
                final=rd.audit_data(cfg,run)
            self.assertTrue(final["audit_complete"]);self.assertEqual(final["expected_indices"],3)
            self.assertEqual(final["committed_indices"],3);self.assertEqual(final["counts"]["candidate_pairs"],12)
            self.assertEqual(final["counts"]["excluded_native_overlap_filter"],2)
            self.assertEqual(final["counts"]["excluded_duplicate_directed_pair"],2)
            self.assertEqual(final["counts"]["excluded_test_scene"],4)
            self.assertEqual(final["train_count"],2);self.assertEqual(final["val_count"],2)
            self.assertFalse((run/"data_audit.sqlite-wal").exists())
            bounds=json.loads((run/"audit_memory_bounds.json").read_text())
            self.assertEqual(bounds["decoder_address_space_limit_bytes"],1024**3)
            for log in run.glob("audit_resources_*.jsonl"):
                for line in log.read_text().splitlines():
                    state=json.loads(line)
                    self.assertLessEqual(state["cache_entries"],256);self.assertLessEqual(state["frame_cache_entries"],256)

    def test_native_protocol4_dictionary_loads_once_within_cap(self):
        with tempfile.TemporaryDirectory(prefix="slim_raw_pickle_") as directory:
            root=Path(directory);run,cfg=fixture(root,raw_pickle=True)
            source=root/"data/megadepth/index/scene_info_0.1_0.7/9001_0.1_0.3.npz"
            self.assertEqual(source.read_bytes()[:2],b"\x80\x04")
            cfg["max_new_indices"]=1
            with patch.object(rd,"test_scene_exclusions",return_value=({"0015"},[])),contextlib.redirect_stdout(io.StringIO()):
                result=rd.audit_data(cfg,run)
            self.assertEqual(result["committed_indices"],1)
            self.assertEqual(result["counts"]["candidate_pairs"],4)
            self.assertEqual(result["counts"]["legal_available_pairs"],2)
            self.assertEqual(result["counts"]["excluded_native_overlap_filter"],1)
            receipts=[json.loads(p.read_text()) for p in (run/"bounded_index_cache").glob("*.json") if ".measure." not in p.name and p.name.count(".")==1]
            self.assertEqual(len(receipts),1)
            receipt=receipts[0]
            self.assertEqual(receipt["source_format"],"raw_pickle_dict_whole_decode_within_cap")
            self.assertEqual(receipt["mapping_format"],receipt["source_format"])
            self.assertEqual(receipt["source_loads"],1);self.assertEqual(receipt["mapping_loads"],1)
            self.assertLess(receipt["peak_rss_bytes"],receipt["address_space_limit_bytes"])
            self.assertFalse((run/"data_audit_summary.json").exists())

    def test_storage_error_blocks_without_false_completion(self):
        with tempfile.TemporaryDirectory(prefix="slim_io_block_") as directory:
            run,cfg=fixture(Path(directory));sha=rd.sha256_file
            def failed(path):
                if Path(path).suffix==".jpg":raise OSError(errno.EIO,"fixture storage I/O failure")
                return sha(path)
            with patch.object(rd,"test_scene_exclusions",return_value=({"0015"},[])),patch.object(rd,"sha256_file",side_effect=failed),contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(OSError):rd.audit_data(cfg,run)
            state=json.loads((run/"bounded_audit_failure.json").read_text())
            self.assertEqual(state["status"],"blocked")
            self.assertEqual(state["counts"].get("candidate_pairs",0),0)
            db=sqlite3.connect(run/"data_audit.incomplete.sqlite")
            self.assertEqual(db.execute("SELECT COUNT(*) FROM audit_chunks").fetchone()[0],0)
            self.assertEqual(db.execute("SELECT cursor FROM audit_work").fetchone()[0],0);db.close()
            self.assertFalse((run/"data_audit_summary.json").exists())

    def test_decoder_failure_blocks_without_truncating(self):
        with tempfile.TemporaryDirectory(prefix="slim_decode_block_") as directory:
            run,cfg=fixture(Path(directory))
            with patch.object(rd,"test_scene_exclusions",return_value=({"0015"},[])),patch("subprocess.run",return_value=type("Failed",(),{"returncode":1})()),contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError,"Bounded index decode blocked"):rd.audit_data(cfg,run)
            self.assertFalse((run/"train_manifest.jsonl").exists())
            self.assertTrue(list((run/"bounded_index_cache").glob("*.blocked.json")))
            self.assertFalse(list((run/"bounded_index_cache").glob("*.sqlite")))


if __name__=="__main__":unittest.main(verbosity=2)
