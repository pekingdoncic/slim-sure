import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from reliability_v1.smoke_manifest import (
    build_smoke_manifest, SmokeDataset, digest, SCHEMA
)
from reliability_data import canonical_json, sha256_file


class SmokeManifestContracts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.data = self.root/"data"/"megadepth"/"train"
        self.data.mkdir(parents=True)
        self.source = self.root/"runs"/"original"
        self.source.mkdir(parents=True)
        self.out = self.root/"runs"/"recovery"
        self.cfg = {"repo_root":str(self.root),"data_root":str(self.root/"data"),"seed":2026}
        self.db = sqlite3.connect(self.source/"data_audit.incomplete.sqlite")
        self.db.executescript("""
            CREATE TABLE audit_identity(key TEXT PRIMARY KEY,value TEXT);
            CREATE TABLE audit_chunks(index_id TEXT PRIMARY KEY,path TEXT,sha256 TEXT,counts TEXT);
            CREATE TABLE files(path TEXT PRIMARY KEY,resolved TEXT,sha256 TEXT,size INTEGER,status TEXT,reason TEXT,shape TEXT,detail TEXT);
            CREATE TABLE frames(id TEXT PRIMARY KEY,scene TEXT,image TEXT,depth TEXT,K TEXT,pose TEXT,status TEXT,reason TEXT);
            CREATE TABLE indices(id TEXT PRIMARY KEY,path TEXT,sha256 TEXT,mapping_path TEXT,mapping_sha256 TEXT,candidate_count INTEGER,status TEXT);
            CREATE TABLE pairs(id INTEGER PRIMARY KEY,pair_id TEXT UNIQUE,scene TEXT,frame0 TEXT,frame1 TEXT,index_id TEXT,ordinal INTEGER,overlap REAL,split TEXT);
            CREATE INDEX pairs_scene ON pairs(scene);
        """)
        exclusions={"scenes":["0015","0022"],"sources":[]}
        (self.source/"test_scene_exclusions.json").write_text(json.dumps(exclusions))
        (self.source/"prior_test_subset_verification.json").write_text(json.dumps({"status":"pass","records":[{"scene_ids":["0015","0022"]}]}))
        self.source_identity={"schema":"test","excluded_scenes":["0015","0022"],"train_list_sha256":"0"*64,"stems":[f"{i:04d}" for i in range(100,115)]}
        self.db.execute("INSERT INTO audit_identity VALUES('source',?)",(canonical_json(self.source_identity),))
        self.pair_ids={}
        for scene in ["0015","0022"]+[f"{i:04d}" for i in range(100,114)]:
            self.add_scene(scene,committed=True)
        self.add_scene("0114",committed=False)
        self.db.commit()

    def tearDown(self):
        self.db.close()
        self.tmp.cleanup()

    def add_scene(self,scene,committed=True):
        indexid=scene+"_0.1_0.3"
        meta=self.source/(indexid+".npz");meta.write_bytes(("original-"+scene).encode())
        mapped=self.source/(indexid+"_mapped.npz");mapped.write_bytes(("mapping-"+scene).encode())
        self.db.execute("INSERT INTO indices VALUES(?,?,?,?,?,?,?)",
                        (indexid,str(meta),sha256_file(meta),str(mapped),sha256_file(mapped),3,"audited"))
        if committed:
            chunk=self.source/(indexid+".jsonl.gz");chunk.write_bytes(("commit-"+scene).encode())
            self.db.execute("INSERT INTO audit_chunks VALUES(?,?,?,?)",(indexid,str(chunk),sha256_file(chunk),"{}"))
        frames=[]
        K=[[100.,0.,10.],[0.,100.,10.],[0.,0.,1.]]
        pose=[[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,1.]]
        for number in range(4):
            folder=self.data/scene;folder.mkdir(exist_ok=True)
            image=folder/(str(number)+".jpg");image.write_bytes(("image-"+scene+"-"+str(number)).encode())
            depth=folder/(str(number)+".h5");depth.write_bytes(("depth-"+scene+"-"+str(number)).encode())
            for file in (image,depth):
                self.db.execute("INSERT INTO files VALUES(?,?,?,?,?,?,?,?)",
                    (str(file),str(file.resolve()),sha256_file(file),file.stat().st_size,"valid",None,"[20,20]","{}"))
            identity=dict(scene=scene,image=str(image),depth=str(depth),K=K,pose=pose)
            fid=hashlib.sha256(canonical_json(identity).encode()).hexdigest()
            self.db.execute("INSERT INTO frames VALUES(?,?,?,?,?,?,?,?)",(fid,scene,str(image),str(depth),canonical_json(K),canonical_json(pose),"valid",None))
            frames.append(fid)
        ids=[]
        for i in range(3):
            pid=hashlib.sha256((frames[i]+"|"+frames[i+1]+"|0to1").encode()).hexdigest()[:32]
            self.db.execute("INSERT INTO pairs(pair_id,scene,frame0,frame1,index_id,ordinal,overlap,split) VALUES(?,?,?,?,?,?,?,?)",
                    (pid,scene,frames[i],frames[i+1],indexid,i,0.2,None))
            ids.append(pid)
        self.pair_ids[scene]=ids

    def test_deterministic_disjoint_committed_only_and_adapter(self):
        first=build_smoke_manifest(self.cfg,self.source,self.out)
        second=build_smoke_manifest(self.cfg,self.source,self.out)
        self.assertEqual(first,second)
        self.assertEqual((len(first["train"]),len(first["val"])),(8,4))
        self.assertFalse(set(first["train_scenes"]) & set(first["val_scenes"]))
        selected=first["train"]+first["val"]
        self.assertTrue(all(r["scene"] not in {"0015","0022","0114"} for r in selected))
        self.assertTrue(first["verification"]["all_selected_data_hashes_verified"])
        self.assertFalse(first["formal_training_eligible"])
        self.assertFalse((self.out/"train_manifest.jsonl").exists())
        self.assertFalse((self.out/"val_manifest.jsonl").exists())
        train=SmokeDataset(self.out/"engineering_smoke_manifest.json","train")
        self.assertEqual(len(train),8)
        self.assertEqual(train.scene_round_robin_indices(100),list(range(8)))
        self.assertEqual(train[0],first["train"][0])
        self.assertFalse(train.formal_training_eligible)
        self.assertEqual(train.identity_sha256,first["content_sha256"])

    def test_committed_audit_growth_does_not_reselect_frozen_smoke(self):
        original=build_smoke_manifest(self.cfg,self.source,self.out)
        indexid="0114_0.1_0.3"
        chunk=self.source/(indexid+".jsonl.gz");chunk.write_bytes(b"new complete commit")
        self.db.execute("INSERT INTO audit_chunks VALUES(?,?,?,?)",(indexid,str(chunk),sha256_file(chunk),"{}"))
        self.db.commit()
        reused=build_smoke_manifest(self.cfg,self.source,self.out)
        self.assertEqual(original,reused)

    def test_selected_data_mutation_fails_without_manifest_replacement(self):
        original=build_smoke_manifest(self.cfg,self.source,self.out)
        target=self.out/"engineering_smoke_manifest.json";before=target.read_bytes()
        data=Path(original["train"][0]["image0"]);data.write_bytes(b"changed")
        with self.assertRaisesRegex(RuntimeError,"checksum mismatch"):
            build_smoke_manifest(self.cfg,self.source,self.out)
        self.assertEqual(target.read_bytes(),before)

    def test_selected_chunk_mutation_fails(self):
        original=build_smoke_manifest(self.cfg,self.source,self.out)
        Path(original["train"][0]["committed_audit_chunk"]["path"]).write_bytes(b"half-written")
        with self.assertRaisesRegex(RuntimeError,"checksum mismatch"):
            build_smoke_manifest(self.cfg,self.source,self.out)

    def test_manifest_tampering_and_incomplete_commit_fail(self):
        original=build_smoke_manifest(self.cfg,self.source,self.out)
        target=self.out/"engineering_smoke_manifest.json"
        changed=dict(original);changed["formal_training_eligible"]=True
        target.write_text(json.dumps(changed))
        with self.assertRaisesRegex(RuntimeError,"checksum mismatch"):
            build_smoke_manifest(self.cfg,self.source,self.out)
        with self.assertRaisesRegex(RuntimeError,"Invalid engineering"):
            SmokeDataset(target,"train")

    def test_budget_is_enforced_without_resampling_or_false_commit(self):
        with self.assertRaisesRegex(RuntimeError,"bounded budget"):
            build_smoke_manifest(self.cfg,self.source,self.out,max_read_bytes=1)
        self.assertFalse((self.out/"engineering_smoke_manifest.json").exists())

    def test_no_quality_or_overlap_ranking_and_explicit_test_ids(self):
        first=build_smoke_manifest(self.cfg,self.source,self.out)
        selected_ids=[r["pair_id"] for r in first["train"]+first["val"]]
        # Existing valid overlap is eligibility, never the selection rank.
        self.db.execute("UPDATE pairs SET overlap=0.6")
        self.db.commit()
        other=self.root/"runs"/"other"
        later=build_smoke_manifest(self.cfg,self.source,other)
        self.assertEqual(selected_ids,[r["pair_id"] for r in later["train"]+later["val"]])
        exclude=json.loads((self.source/"test_scene_exclusions.json").read_text())
        exclude["test_pair_ids"]=[selected_ids[0]]
        (self.source/"test_scene_exclusions.json").write_text(json.dumps(exclude))
        third=build_smoke_manifest(self.cfg,self.source,self.root/"runs"/"third")
        self.assertNotIn(selected_ids[0],[r["pair_id"] for r in third["train"]+third["val"]])

if __name__=="__main__":
    unittest.main()
