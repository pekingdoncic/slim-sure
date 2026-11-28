"""Enumerate the frozen sampler on the actual official indices, without training or decoding images."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from experiments.lowtex_e2e_v1.training.sampling import Pool


def main():
    exp = Path(__file__).resolve().parents[1]
    base = exp.parents[1] / "data/megadepth"
    budget = json.loads((exp / "manifests/full_budget.json").read_text())
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=budget["seeds"], default=budget["seeds"][0])
    pool = Pool(exp, base, parser.parse_args().seed)
    ids = json.loads((pool.cache / "unordered_identity.json").read_text())
    assert ids["total"] == budget["unordered_unique"]
    row_visits = [np.zeros(b["legal_rows"], dtype=np.uint32) for b in pool.bins]
    physical = np.zeros(ids["total"], dtype=np.bool_)
    tables = []
    for b in pool.bins:
        path = pool.cache / (b["bin"] + ".unordered.npy")
        assert hashlib.sha256(path.read_bytes()).hexdigest() == ids["hashes"][path.name]
        tables.append(np.load(path, mmap_mode="r"))
    visits = 0
    started = time.monotonic()
    for epoch, stop in enumerate(pool.ends[1:], start=1):
        while visits < stop:
            bi, row, _, first = pool.at(visits)
            assert first == (row_visits[bi][row] == 0), (visits, bi, row)
            row_visits[bi][row] += 1
            physical[tables[bi][row]] = True
            visits += 1
        print(f"epoch={epoch} sampler_visits={visits} elapsed_s={time.monotonic()-started:.2f}", flush=True)
    assert visits == budget["micro_batches"]
    assert visits == budget["optimizer_updates"] * budget["accumulation"]
    ordered = sum(int(np.count_nonzero(v)) for v in row_visits)
    assert ordered == budget["ordered_rows"]
    for b, counts in zip(pool.bins, row_visits):
        assert int(counts.sum()) == budget["per_bin_quota"][b["bin"]]
        assert counts.min() > 0
    assert physical.all()
    print(json.dumps(dict(status="PASS", mode="sampler_enumeration_only", seed=pool.seed,
                         visits=visits, ordered=ordered, unordered=int(physical.sum()),
                         bins=len(row_visits), elapsed_s=time.monotonic()-started)), flush=True)


if __name__ == "__main__":
    main()
