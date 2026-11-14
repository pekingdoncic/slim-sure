# SLiM reliability V1

The default is engineering acceptance only. Original SLiM weights and BN stay frozen;
only the 29,185-parameter shared projection and reliability head train.
The original model has both new switches disabled by default.

```bash
cd /home/bupt/projects/SLiM-master
source /home/bupt/miniconda3/etc/profile.d/conda.sh
conda activate slim
python -u reliability_stage.py --config configs/reliability_v1.json --run-stage engineering
```

Stage 1 audits every configured index and freezes a scene-disjoint full_available
split, then uses 8 real training pairs, at most 50 physical optimizer updates
across retries, and 4 independent validation pairs. A matching acceptance receipt
is mandatory for formal training. CPU tests and inference preflight are not a
substitute for real optimizer/checkpoint recovery acceptance.

```bash
python -u reliability_stage.py --config configs/reliability_v1.json --run-stage full_train
python -u reliability_stage.py --config configs/reliability_v1.json --run-stage full_train --resume
python -u reliability_stage.py --config configs/reliability_v1.json --run-stage evaluate
```

Formal training traverses every frozen pair once per epoch for 10 epochs; there is
no pair/step/whole-task time cap. Every epoch validates the complete held-out
manifest. `--run-stage all` explicitly runs engineering, initializes a fresh head
and optimizer, then trains and evaluates. It is never the default.

A stalled pair watchdog is separate from total training duration. Checkpoints
atomically commit content-addressed payloads and include optimizer, RNG, order,
cursor and experiment identity. Data is reverified against frozen hashes on first
read and when filesystem metadata changes. Independent pilot exports use one
unique pair per GPU worker; timing workers exclude GT/depth diagnostics.

The current run is blocked by WSL virtual swap-device I/O failure. No final split
or real-data smoke checkpoint has been published. See
`runs/reliability_v1_20261114T172319/Engineering_Reliability_V1_Report.md`.
After the environment is stable, resume its audit using that run's
`resume_data_audit.sh`, then rerun engineering. Formal modes remain gated.

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -p 'test_reliability*.py' -v
```
