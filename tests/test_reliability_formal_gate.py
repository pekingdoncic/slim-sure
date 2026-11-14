"""CPU-only adversarial receipts for the formal-training gate."""
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from reliability_v1.formal_gate import FormalPrerequisiteError, verify_formal_prerequisites


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True) + "\n")


class FormalGateTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.run = Path(self.temp.name)
        self.list = self.run / "train_list.txt"
        self.list.write_text("0100_0.1_0.3\n0200_0.1_0.3\n")
        self.profile = self.run / "profile.json"
        self.profile.write_text("{}\n")
        self.test_list = self.run / "test_list.txt"
        self.test_list.write_text("0015\n0022\n")
        self.excluded = ["0015", "0022"]
        write(self.run / "test_scene_exclusions.json",
              dict(scenes=self.excluded, sources=[dict(path=str(self.test_list),
                    sha256=sha(self.test_list), scenes=self.excluded)]))
        self.dbpath = self.run / "data_audit.sqlite"
        with sqlite3.connect(self.dbpath) as db:
            db.executescript("""
                CREATE TABLE indices(id TEXT PRIMARY KEY,status TEXT);
                CREATE TABLE audit_chunks(index_id TEXT PRIMARY KEY);
                CREATE TABLE audit_work(index_id TEXT);
                CREATE TABLE audit_identity(key TEXT PRIMARY KEY,value TEXT);
                CREATE TABLE pairs(id INTEGER PRIMARY KEY,pair_id TEXT,scene TEXT,frame0 TEXT,frame1 TEXT,index_id TEXT,ordinal INTEGER,split TEXT,overlap REAL);
                CREATE TABLE manifests(split TEXT,position INTEGER,pair_row INTEGER);
                CREATE TABLE frames(id TEXT PRIMARY KEY,image TEXT,depth TEXT,status TEXT);
                CREATE TABLE files(path TEXT PRIMARY KEY,status TEXT,sha256 TEXT);
            """)
            stems = self.list.read_text().splitlines()
            db.execute("INSERT INTO audit_identity VALUES('source',?)", (json.dumps(dict(
                stems=stems, train_list_sha256=sha(self.list), excluded_scenes=self.excluded)),))
            for i, (split, scene) in enumerate((("train", "0100"), ("val", "0200")), 1):
                stem = stems[i-1]
                db.execute("INSERT INTO indices VALUES(?, 'audited')", (stem,))
                db.execute("INSERT INTO audit_chunks VALUES(?)", (stem,))
                f0, f1 = scene + "_0", scene + "_1"
                db.execute("INSERT INTO pairs VALUES(?,?,?,?,?,?,?,?,?)",
                           (i, "pair" + str(i), scene, f0, f1, stem, 0, split, 0.2))
                db.execute("INSERT INTO manifests VALUES(?,0,?)", (split, i))
                for fid in (f0, f1):
                    image, depth = fid + ".jpg", fid + ".h5"
                    db.execute("INSERT INTO frames VALUES(?,?,?,'valid')", (fid, image, depth))
                    for name in (image, depth):
                        db.execute("INSERT INTO files VALUES(?,'valid',?)", (name, "a"*64))
                write(self.run / (split + "_manifest.jsonl"), dict(
                    pair_id="pair"+str(i), scene=scene, frame0=f0, frame1=f1,
                    index_id=stem, ordinal=0, direction="0to1"))
        self.summary = dict(status="committed", scope="full_available", audit_complete=True,
            expected_indices=2, committed_indices=2, train_count=1, val_count=1,
            train_scenes=["0100"], val_scenes=["0200"], excluded_test_scenes=self.excluded,
            train_list_sha256=sha(self.list), data_audit_sha256=sha(self.dbpath),
            manifest_sha256={s:sha(self.run/(s+"_manifest.jsonl")) for s in ("train","val")})
        self.config = dict(repo_root=str(self.run), train_index=str(self.list), epochs=10,
            profile=str(self.profile), batch_size=1, num_workers=0, max_parent_samples=1024)
        self.config["training_protocol"] = dict(status="confirmed",
            name="full_manifest_once_per_epoch", overlap=dict(comparison=">",threshold=0.0),
            manifest_sha256=self.summary["manifest_sha256"],
            data_audit_sha256=self.summary["data_audit_sha256"], epochs=10,
            train_pair_visits_per_epoch=1, validation_pair_visits_per_pass=1,
            replacement=False, drop_last=False)
        self.identity = dict(profile_sha=sha(self.profile), config_sha="fixture",
            data={k:self.summary[k] for k in ("manifest_sha256","data_audit_sha256","train_list_sha256",
                  "train_count","val_count","train_scenes","val_scenes")})
        self.accepted = dict(status="pass", scope="full_manifest_engineering",
                             formal_training_eligible=True, identity=self.identity)
        self.budget = dict(status="estimated", identity=self.identity, train_pairs=1,
            validation_pairs=1, epochs=10, expected_training_pair_visits=10,
            expected_validation_pair_visits=11, training_seconds=20.0, validation_seconds=11.0,
            cost_config=dict(train_pair_seconds=2.0, validation_pair_seconds=1.0,
                batch_size=1, num_workers=0, max_parent_samples=1024, profile_sha256=sha(self.profile)),
            storage=dict(conservative_peak_bytes=1024,required_free_margin_bytes=2*1024**3))
        self.save()

    def save(self):
        write(self.run/"data_audit_summary.json", self.summary)
        write(self.run/"engineering_acceptance.json", self.accepted)
        write(self.run/"formal_budget.json", self.budget)

    def rehash_db(self):
        checksum = sha(self.dbpath)
        self.summary["data_audit_sha256"] = checksum
        self.identity["data"]["data_audit_sha256"] = checksum
        self.config["training_protocol"]["data_audit_sha256"] = checksum
        self.save()

    def gate(self):
        self.save()
        return verify_formal_prerequisites(self.config, self.run, self.identity)

    def test_complete_toy_fixture(self):
        result = self.gate()
        self.assertEqual(result["expected_indices"], 2)
        self.assertEqual(result["expected_training_pair_visits"], 10)

    def test_core_smoke_receipt_cannot_authorize(self):
        self.accepted["scope"] = "core_smoke"
        with self.assertRaisesRegex(FormalPrerequisiteError, "acceptance scope"):
            self.gate()

    def test_same_scope_still_requires_eligible_identity(self):
        self.accepted["formal_training_eligible"] = False
        with self.assertRaisesRegex(FormalPrerequisiteError, "smoke-only"):
            self.gate()
        self.accepted["formal_training_eligible"] = True
        self.accepted["identity"] = {"data": "other run"}
        with self.assertRaisesRegex(FormalPrerequisiteError, "engineering identity"):
            self.gate()

    def test_summary_pass_does_not_hide_missing_index(self):
        with sqlite3.connect(self.dbpath) as db:
            db.execute("DELETE FROM indices WHERE id='0200_0.1_0.3'")
        self.rehash_db()
        with self.assertRaisesRegex(FormalPrerequisiteError, "index coverage"):
            self.gate()

    def test_summary_pass_does_not_hide_uncommitted_chunk(self):
        with sqlite3.connect(self.dbpath) as db:
            db.execute("DELETE FROM audit_chunks WHERE index_id='0200_0.1_0.3'")
        self.rehash_db()
        with self.assertRaisesRegex(FormalPrerequisiteError, "chunk coverage"):
            self.gate()


    def test_explicit_missing_index_cannot_shrink_formal_scope(self):
        with sqlite3.connect(self.dbpath) as db:
            db.execute("UPDATE indices SET status='missing_index' WHERE id='0200_0.1_0.3'")
            db.execute("DELETE FROM audit_chunks WHERE index_id='0200_0.1_0.3'")
        self.rehash_db()
        with self.assertRaisesRegex(FormalPrerequisiteError, "scope reduction"):
            self.gate()

    def test_recorded_missing_index_count_blocks_even_if_rows_claim_audited(self):
        self.summary["counts"]={"missing_indices":1}
        with self.assertRaisesRegex(FormalPrerequisiteError, "scope reduction"):
            self.gate()

    def test_incomplete_transaction_rejected(self):
        with sqlite3.connect(self.dbpath) as db:
            db.execute("INSERT INTO audit_work VALUES('0200_0.1_0.3')")
        self.rehash_db()
        with self.assertRaisesRegex(FormalPrerequisiteError, "incomplete audit transactions"):
            self.gate()

    def test_rehashed_manifest_must_match_database(self):
        p = self.run/"val_manifest.jsonl"
        obj = json.loads(p.read_text())
        obj["frame1"] = "tampered"
        write(p,obj)
        self.summary["manifest_sha256"]["val"] = sha(p)
        with self.assertRaisesRegex(FormalPrerequisiteError, "manifest row"):
            self.gate()

    def test_database_hash_not_trusted(self):
        with sqlite3.connect(self.dbpath) as db:
            db.execute("UPDATE pairs SET overlap=0")
        with self.assertRaisesRegex(FormalPrerequisiteError, "database checksum"):
            self.gate()

    def test_zero_overlap_rejected_even_after_rehash(self):
        with sqlite3.connect(self.dbpath) as db:
            db.execute("UPDATE pairs SET overlap=0")
        self.rehash_db()
        with self.assertRaisesRegex(FormalPrerequisiteError, "overlap > 0.0"):
            self.gate()

    def test_duplicate_manifest_pair_rejected(self):
        with sqlite3.connect(self.dbpath) as db:
            db.execute("UPDATE manifests SET pair_row=1 WHERE split='val'")
        self.rehash_db()
        with self.assertRaisesRegex(FormalPrerequisiteError, "duplicate manifest pair"):
            self.gate()

    def test_actual_scene_overlap_rejected(self):
        with sqlite3.connect(self.dbpath) as db:
            db.execute("UPDATE pairs SET scene='0100' WHERE split='val'")
        p=self.run/"val_manifest.jsonl"
        obj=json.loads(p.read_text());obj["scene"]="0100";write(p,obj)
        self.summary["manifest_sha256"]["val"]=sha(p)
        self.summary["val_scenes"][:]=["0100"]
        self.rehash_db()
        with self.assertRaisesRegex(FormalPrerequisiteError, "scenes overlap"):
            self.gate()

    def test_test_scene_rejected(self):
        with sqlite3.connect(self.dbpath) as db:
            db.execute("UPDATE pairs SET scene='0015' WHERE split='val'")
        p=self.run/"val_manifest.jsonl"
        obj=json.loads(p.read_text());obj["scene"]="0015";write(p,obj)
        self.summary["manifest_sha256"]["val"]=sha(p)
        self.summary["val_scenes"][:]=["0015"]
        self.rehash_db()
        with self.assertRaisesRegex(FormalPrerequisiteError, "test scene"):
            self.gate()

    def test_file_validity_receipt_required(self):
        with sqlite3.connect(self.dbpath) as db:
            db.execute("UPDATE files SET status='missing' WHERE path='0100_0.jpg'")
        self.rehash_db()
        with self.assertRaisesRegex(FormalPrerequisiteError, "invalid/unaudited"):
            self.gate()

    def test_protocol_pending_rejected(self):
        self.config["training_protocol"]["status"]="pending"
        with self.assertRaisesRegex(FormalPrerequisiteError, "explicit training protocol"):
            self.gate()

    def test_native_sampler_cannot_masquerade_as_full_manifest(self):
        self.config["training_protocol"]["replacement"]=True
        with self.assertRaisesRegex(FormalPrerequisiteError, "visit every"):
            self.gate()

    def test_non_native_overlap_condition_rejected(self):
        self.config["training_protocol"]["overlap"]["comparison"]=">="
        with self.assertRaisesRegex(FormalPrerequisiteError, "native overlap"):
            self.gate()

    def test_finite_complete_cost_required(self):
        self.budget["training_seconds"]=float("nan")
        with self.assertRaisesRegex(FormalPrerequisiteError, "non-finite"):
            self.gate()
        self.budget["training_seconds"]=2.0
        with self.assertRaisesRegex(FormalPrerequisiteError, "complete visits"):
            self.gate()

    def test_cost_must_include_initial_validation(self):
        self.budget["expected_validation_pair_visits"]=10
        with self.assertRaisesRegex(FormalPrerequisiteError, "validation_pair_visits"):
            self.gate()

    def test_changed_test_evidence_rejected(self):
        self.test_list.write_text("0015\n0022\n0100\n")
        with self.assertRaisesRegex(FormalPrerequisiteError, "test exclusion source checksum"):
            self.gate()


if __name__ == "__main__":
    unittest.main()
