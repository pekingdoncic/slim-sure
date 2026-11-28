"""Files shared by the isolated server pipeline."""
import contextlib
import datetime as dt
import fcntl
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
PKG = "experiments.lowtex_pro5000_hr_aug"


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def sha(path):
    with Path(path).open("rb") as fp:
        h = hashlib.sha256()
        for chunk in iter(lambda: fp.read(1 << 20), b""):
            h.update(chunk)
        return h.hexdigest()


@contextlib.contextmanager
def atomic_path(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix="." + path.name)
    os.close(fd)
    tmp = Path(name)
    try:
        yield tmp
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def write(path, value):
    with atomic_path(path) as tmp:
        tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False, default=str) + "\n")


def append(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fp:
        fp.write(json.dumps(value, allow_nan=False, default=str) + "\n")
        fp.flush()


@contextlib.contextmanager
def lock(path, blocking=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        fcntl.flock(f, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        yield


def config(model):
    import yaml
    directory = EXP / "configs"
    out = {}
    for p in (directory/"base.yaml", directory/"pro5000_shared.yaml", next(directory.glob(model.lower()+"_*.yaml"))):
        out.update(yaml.safe_load(p.read_text()))
    if out["lhg"] and not out["bbr"] or out["bbr"] and not out["ccfi"]:
        raise ValueError("LHG requires BBR and BBR requires CCFI")
    return out


def locations(mode, run_id, model="P0"):
    if mode not in {"smoke","short","medium","full"} or model not in {"P0","P1","P2","P3","P4"}:
        raise ValueError("invalid mode or experiment")
    if not re.fullmatch(r"[A-Za-z0-9_-][A-Za-z0-9_.-]{0,79}", run_id):
        raise ValueError("invalid run ID")
    return EXP/"outputs"/mode/model/run_id, EXP/"reports"/mode/run_id
