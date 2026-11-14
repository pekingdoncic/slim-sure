# SLiM reliability V1

Original SLiM weights and BN stay frozen. Only the 29,185-parameter shared
projection and reliability head train; both new model switches default to off.

## Core engineering while the full audit is incomplete

After a bounded environment probe, use a distinct recovery directory:

```bash
cd /home/bupt/projects/SLiM-master
source /home/bupt/miniconda3/etc/profile.d/conda.sh
conda activate slim
PYTHONDONTWRITEBYTECODE=1 python reliability_engineering_recovery.py \
  --config configs/reliability_v1.json \
  --source-run runs/reliability_v1_20261114T172319 \
  --run-dir runs/reliability_v1_20261114T172319/recovery_20261114T201118
```

This entry only dispatches core engineering. Its fixed-seed 8+4 manifest selects
committed, hash-verified pairs, excludes test scenes/pairs, and separates training
and validation scenes. It is explicitly ineligible for formal training.
Three actual pairs exercise the corrected-coordinate baseline regression; the
real training, validation and checkpoint/RNG/optimizer/cursor recovery chain then
runs at most 50 physical optimizer updates including recovery branches and retries.
A supervisor monitors the current boot's storage log, host backing-volume space,
available memory and swap, and records failures without modifying WSL configuration.

The current recovery passed core engineering with 34 physical updates, a real
checkpoint, and exact next-step recovery equality. The full audit remains
incomplete. Scope and evidence are recorded in
`runs/reliability_v1_20261114T172319/recovery_20261114T201118/`.
Historical storage failure evidence is preserved in the original run.

## Bounded audit

`reliability_data.py --resume-audit --max-new-indices 1` supports a single-index
boundary and durable candidate-batch cursors. The original committed index chunks
are reused after identity verification. File/frame caches hold at most 256
entries; SQLite handles deduplication and ordering on disk. The raw-pickle
index decoder runs in an isolated process with at most 1 GiB address space.
An oversized or failed index blocks progress; it is not silently skipped.
The recovery report records measured pilot limits and unfinished integrity checks.

## Formal training gates

The legacy `reliability_stage.py --run-stage engineering` remains the path for
acceptance against a completed final manifest. The separate core receipt cannot
substitute for its full-manifest receipt. Both the CLI and direct `full_train`
function require:

- Complete configured-index audit with no missing indices or unfinished transactions.
- Final immutable train/validation manifests, scene isolation and test exclusions.
- Full-manifest engineering acceptance matching code/config/data/profile identity.
- An explicitly confirmed training protocol matching the implemented sampler.
- Complete numeric time/storage costs and a matching cost configuration.

The native dataset uses overlap strictly greater than 0.0 and its sampler draws
100 times **with replacement per NPZ subset** (36,800 nominal visits for all 368
configured subsets before final holdout). The new trainer instead visits every
final training pair once per epoch. This protocol difference is unresolved.
No formal 10-epoch training, `evaluate`, or `all` was started in this recovery.

Legacy `formal_budget.json` output is insufficient for the stricter gate: the
future cost-generation path needs matching identity, numeric validation costs,
the final pair counts and confirmed protocol fields. Merely invoking `all` does
not resolve these prerequisites. Warm smoke timings are planning evidence only.

Checkpoints atomically commit content-addressed payloads with optimizer, RNG,
sampler cursor and experiment identity. CPU fixtures cannot replace real-data
training and recovery acceptance.

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -p 'test_reliability*.py' -v
```
