"""Read-only prerequisites for formal training; temporary smoke is never sufficient.

This gate intentionally fails closed. It hashes committed artifacts and streams
manifest/SQLite rows with bounded memory. It never audits raw data, runs a model,
or treats missing evidence as permission to start training.
"""
from contextlib import closing
import hashlib
import json
import math
from pathlib import Path
import sqlite3


class FormalPrerequisiteError(RuntimeError):
    """A required formal-training receipt is missing, incomplete, or stale."""


def _require(condition, message):
    if not condition:
        raise FormalPrerequisiteError("Formal training blocked: " + message)


def _read(path):
    _require(path.is_file(), "missing " + str(path))
    with path.open(encoding="utf-8") as handle:
        result = json.load(handle)
    _require(isinstance(result, dict), "expected JSON object: " + str(path))
    return result


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _integer(value, name, minimum=1):
    _require(type(value) is int and value >= minimum, "invalid " + name)
    return value


def _positive(value, name, minimum=0):
    _require(type(value) in (int, float) and math.isfinite(value) and value > minimum,
             "non-finite or non-positive " + name)
    return value


def _same(value, expected, name):
    _require(value == expected, name + " missing or stale")


def _verify_database(db, stems, summary, run, excluded):
    rows = dict(db.execute("SELECT id,status FROM indices"))
    _same(set(rows), set(stems), "actual audited index coverage")
    _require(all(s in ("audited", "missing_index") for s in rows.values()),
             "uncommitted index status")
    _require(all(s == "audited" for s in rows.values()),
             "missing configured index; formal scope reduction is not authorized")
    _require(summary.get("counts", {}).get("missing_indices", 0) == 0,
             "audit recorded missing indices; formal scope reduction is not authorized")
    chunks = {r[0] for r in db.execute("SELECT index_id FROM audit_chunks")}
    _same(chunks, {k for k, v in rows.items() if v == "audited"},
          "committed audit chunk coverage")
    _same(db.execute("SELECT COUNT(*) FROM audit_work").fetchone()[0], 0,
          "incomplete audit transactions")
    source = db.execute("SELECT value FROM audit_identity WHERE key='source'").fetchone()
    _require(source is not None, "missing audit source identity")
    source = json.loads(source[0])
    _same(source.get("stems"), stems, "audit source index order")
    _same(source.get("train_list_sha256"), summary["train_list_sha256"],
          "audit source list hash")
    _same(set(source.get("excluded_scenes", [])), excluded, "audit test exclusions")
    _require(db.execute("SELECT 1 FROM pairs WHERE split NOT IN ('train','val') OR split IS NULL LIMIT 1").fetchone() is None,
             "unassigned eligible pair")
    _require(db.execute("SELECT 1 FROM manifests WHERE split NOT IN ('train','val') LIMIT 1").fetchone() is None,
             "unexpected manifest split")
    _require(db.execute("SELECT pair_row FROM manifests GROUP BY pair_row HAVING COUNT(*) != 1 LIMIT 1").fetchone() is None,
             "duplicate manifest pair")
    _same(db.execute("SELECT COUNT(*) FROM pairs").fetchone()[0],
          summary["train_count"] + summary["val_count"], "full eligible pair coverage")
    seen_scenes = {}
    for split in ("train", "val"):
        expected = summary[split + "_count"]
        count, low, high = db.execute(
            "SELECT COUNT(*),MIN(position),MAX(position) FROM manifests WHERE split=?", (split,)).fetchone()
        _same((count, low, high), (expected, 0, expected - 1), split + " positions/count")
        cursor = db.execute(
            """SELECT m.position,p.pair_id,p.scene,p.frame0,p.frame1,p.index_id,p.ordinal,p.split,p.overlap
               FROM manifests m LEFT JOIN pairs p ON p.id=m.pair_row
               WHERE m.split=? ORDER BY m.position""", (split,))
        n = 0
        scenes = set()
        with (run / (split + "_manifest.jsonl")).open(encoding="utf-8") as handle:
            for row in cursor:
                position, pair_id, scene, f0, f1, index_id, ordinal, assigned, overlap = row
                _require(assigned == split and index_id in rows and rows[index_id] == "audited",
                         "manifest references an unaudited/wrong-split pair")
                _require(type(overlap) in (float, int) and math.isfinite(overlap) and overlap > 0,
                         "pair violates native finite overlap > 0.0 contract")
                record = json.loads(handle.readline())
                _same(record, dict(pair_id=pair_id, scene=scene, frame0=f0, frame1=f1,
                                   index_id=index_id, ordinal=ordinal, direction="0to1"),
                      split + " manifest row " + str(position))
                scenes.add(scene)
                n += 1
            _require(not handle.read(1), "extra rows in " + split + " manifest")
        _same(n, expected, split + " streamed count")
        _same(scenes, set(summary[split + "_scenes"]), split + " actual scenes")
        _require(bool(scenes) and not scenes.intersection(excluded), split + " contains test scene")
        seen_scenes[split] = scenes
    _require(not seen_scenes["train"].intersection(seen_scenes["val"]),
             "training and validation scenes overlap")
    # Check the persisted validity chain, without opening or decoding raw files.
    invalid = db.execute(
        """SELECT 1 FROM pairs p
           LEFT JOIN frames a ON a.id=p.frame0 LEFT JOIN frames b ON b.id=p.frame1
           LEFT JOIN files ai ON ai.path=a.image LEFT JOIN files ad ON ad.path=a.depth
           LEFT JOIN files bi ON bi.path=b.image LEFT JOIN files bd ON bd.path=b.depth
           WHERE a.status IS NOT 'valid' OR b.status IS NOT 'valid'
             OR ai.status IS NOT 'valid' OR ad.status IS NOT 'valid'
             OR bi.status IS NOT 'valid' OR bd.status IS NOT 'valid'
             OR ai.sha256 IS NULL OR ad.sha256 IS NULL OR bi.sha256 IS NULL OR bd.sha256 IS NULL
           LIMIT 1""").fetchone()
    _require(invalid is None, "manifest pair references invalid/unaudited frame or file")


def verify_formal_prerequisites(config, run, run_identity):
    """Return verified evidence, or raise FormalPrerequisiteError before training.

    Required receipts: final data_audit_summary.json + final DB/manifests,
    engineering_acceptance.json with full_manifest_engineering scope,
    config.training_protocol confirmed for the implemented exhaustive protocol,
    and formal_budget.json with matching complete counts and numeric costs.
    """
    try:
        return _verify(config, Path(run).resolve(), run_identity)
    except FormalPrerequisiteError:
        raise
    except (OSError, ValueError, KeyError, TypeError, sqlite3.Error) as error:
        raise FormalPrerequisiteError("Formal training blocked: malformed/missing/stale prerequisite: " + str(error)) from error


def _verify(config, run, identity):
    summary = _read(run / "data_audit_summary.json")
    _same(summary.get("status"), "committed", "full audit status")
    _same(summary.get("scope"), "full_available", "full audit scope")
    _require(summary.get("audit_complete") is True, "full audit incomplete")
    root = Path(config["repo_root"])
    data_root = Path(config.get("data_root", root / "data"))
    train_list = Path(config.get("train_index", data_root / "megadepth/index/trainvaltest_list/train_list.txt"))
    stems = [line.strip().removesuffix(".npz") for line in train_list.read_text().splitlines() if line.strip()]
    _require(bool(stems) and len(stems) == len(set(stems)), "empty/duplicate configured indices")
    for key in ("expected_indices", "committed_indices"):
        _same(_integer(summary.get(key), key), len(stems), key)
    for alias in ("expected_index_count", "completed_index_count"):
        if alias in summary:
            _same(summary[alias], len(stems), alias)
    _same(_sha(train_list), summary.get("train_list_sha256"), "training list checksum")
    for split in ("train", "val"):
        _integer(summary.get(split + "_count"), split + " count")
        _same(_sha(run / (split + "_manifest.jsonl")), summary["manifest_sha256"].get(split),
              split + " manifest checksum")
    database = run / "data_audit.sqlite"
    _require(not any(Path(str(database) + suffix).exists() for suffix in ("-wal", "-shm", "-journal")),
             "final audit database has a transaction sidecar")
    _same(_sha(database), summary.get("data_audit_sha256"), "final audit database checksum")
    exclusions = _read(run / "test_scene_exclusions.json")
    excluded = set(exclusions["scenes"])
    _require({"0015", "0022"}.issubset(excluded), "known MegaDepth1500 test exclusions missing")
    _same(excluded, set(summary["excluded_test_scenes"]), "summary test exclusions")
    _require(bool(exclusions.get("sources")), "test exclusion source evidence missing")
    source_scenes = {"0015", "0022"}
    for source in exclusions["sources"]:
        _same(_sha(source["path"]), source["sha256"], "test exclusion source checksum")
        source_scenes.update(source["scenes"])
    _same(source_scenes, excluded, "test exclusion evidence coverage")
    with closing(sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)) as db:
        db.execute("PRAGMA query_only=ON")
        db.execute("PRAGMA cache_size=-8192")
        db.execute("PRAGMA temp_store=FILE")
        db.execute("PRAGMA mmap_size=0")
        _verify_database(db, stems, summary, run, excluded)
    # The runtime identity must bind the exact full artifacts, not smoke hashes.
    for key in ("manifest_sha256", "data_audit_sha256", "train_list_sha256",
                "train_count", "val_count", "train_scenes", "val_scenes"):
        _same(identity["data"].get(key), summary[key], "runtime identity " + key)
    accepted = _read(run / "engineering_acceptance.json")
    _same(accepted.get("status"), "pass", "engineering acceptance status")
    _same(accepted.get("scope"), "full_manifest_engineering", "engineering acceptance scope")
    _require(accepted.get("formal_training_eligible") is True, "engineering receipt is smoke-only")
    _same(accepted.get("identity"), identity, "engineering identity")
    protocol = config.get("training_protocol", {})
    _same(protocol.get("status"), "confirmed", "explicit training protocol")
    _same(protocol.get("name"), "full_manifest_once_per_epoch", "implemented training protocol")
    _same(protocol.get("overlap"), {"comparison": ">", "threshold": 0.0}, "native overlap condition")
    _require(protocol.get("replacement") is False and protocol.get("drop_last") is False,
             "formal protocol must visit every full-manifest pair once per epoch")
    epochs = _integer(config["epochs"], "epochs")
    _same(protocol.get("epochs"), epochs, "protocol epochs")
    _same(protocol.get("manifest_sha256"), summary["manifest_sha256"], "protocol manifests")
    _same(protocol.get("data_audit_sha256"), summary["data_audit_sha256"], "protocol database")
    _same(protocol.get("train_pair_visits_per_epoch"), summary["train_count"], "protocol train coverage")
    _same(protocol.get("validation_pair_visits_per_pass"), summary["val_count"], "protocol validation coverage")
    budget = _read(run / "formal_budget.json")
    _same(budget.get("status"), "estimated", "formal budget status")
    _same(budget.get("identity"), identity, "formal budget identity")
    expected = dict(train_pairs=summary["train_count"], validation_pairs=summary["val_count"],
                    epochs=epochs, expected_training_pair_visits=summary["train_count"] * epochs,
                    expected_validation_pair_visits=summary["val_count"] * (epochs + 1))
    for key, value in expected.items():
        _same(budget.get(key), value, "formal budget " + key)
    cost = budget["cost_config"]
    for key in ("batch_size", "num_workers", "max_parent_samples"):
        _same(cost.get(key), config[key], "cost config " + key)
    _same(cost.get("profile_sha256"), _sha(config["profile"]), "cost profile checksum")
    _same(identity.get("profile_sha"), cost["profile_sha256"], "runtime profile identity")
    train_seconds = _positive(cost.get("train_pair_seconds"), "training pair cost")
    val_seconds = _positive(cost.get("validation_pair_seconds"), "validation pair cost")
    for name, computed in (("training_seconds", train_seconds * expected["expected_training_pair_visits"]),
                           ("validation_seconds", val_seconds * expected["expected_validation_pair_visits"])):
        estimate = _positive(budget.get(name), name)
        _require(math.isclose(estimate, computed, rel_tol=1e-6, abs_tol=1e-6), name + " does not cover complete visits")
    _positive(budget["storage"].get("conservative_peak_bytes"), "storage peak")
    margin = budget["storage"].get("required_free_margin_bytes")
    _integer(margin, "storage margin", 2 * 1024**3)
    return dict(status="verified", scope="formal_prerequisites", expected_indices=len(stems),
                train_count=summary["train_count"], val_count=summary["val_count"],
                manifest_sha256=summary["manifest_sha256"], data_audit_sha256=summary["data_audit_sha256"],
                training_protocol=protocol["name"], expected_training_pair_visits=expected["expected_training_pair_visits"])
