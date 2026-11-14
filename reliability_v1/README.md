# SLiM reliability V1

Original SLiM weights and BN remain frozen. The shared projection and reliability
head have 29,185 trainable parameters. R832, batch size 1, workers 0, AMP and the
locked numerical profile remain unchanged.

## Phase 1 preparation and evidence

The independent run is `runs/phase1_20261115T004137`. Historical run
`runs/reliability_v1_20261114T172319` is read-only. Its identity-matched
3-pair baseline regression, 8+4 real smoke, 34 physical optimizer updates and
exact next-step recovery are inherited only for the unchanged core paths.

This Phase 1 does not update the formal model or execute real final100 EPE.
It completes the 368-index audit, freezes the physical-scene split and sampled
plans, computes train-only statistics, checks real labels, and exercises the
new functions in isolated component tests. The authoritative readiness result
and limitations are in `Phase1_Readiness_Report.md` and `phase1_ready.json`
inside the independent run.

The bounded audit preserves 107 historical commits and adds 261. SQLite performs
deduplication, ordering and manifest export; Python file/frame caches hold at
most 256 entries. Each source decoder has a hard address-space cap (default
1 GiB; this measured run needs 2 GiB for its largest index). An initial 1 GiB
MemoryError remains recorded. No index is silently omitted. Historical storage
faults remain preserved; healthy current-boot probes do not prove their cause fixed.

## Confirmed second-task protocol

The full legal pool is split by physical scene, excluding test scenes. Native
overlap remains finite and strictly greater than zero. The confirmed sampler
draws at most 100 pairs without replacement per training scene per epoch for
10 epochs. Seed2026, epoch and stable scene identity determine each plan.
The fixed validation set has at most50 pairs per independent validation scene.
A stable round-robin100-pair validation subset is reserved for final EPE.

Each visit uses one shared deterministic subset of at most1024 valid parents for
the head BCE and scalar LR records. Records retain float32 features/weights and
lossless raw IDs. End-of-epoch LR uses all committed visits through that epoch,
pair-equal streamed L-BFGS and L2=1e-6 on the three standardized coefficients.
The intercept is unregularized. Failed LR fits are never reused as successful
phase commits. The head optimizer contains only the head parameters.

Comparison uses pair AUROC/AP (AP is the AUPRC implementation), scene-equal
aggregation and paired scene bootstrap2000, seed2026. Empty or single-class
pairs have undefined AUROC and AP. Validation pair-equal BCE selects the main
checkpoint; exact ties retain the earlier epoch. The report also compares the
AUROC-best epoch. Final EPE samples depth at the actual final source point and
aggregates valid children by median, giving each raw parent one vote.

The old exhaustive trainer and native replacement-per-NPZ sampler are separate
protocols. Neither is selected by this experiment; the legacy gate stays intact.

## Next task only: launch and resume

After checking that Phase1 reports ready, the following detached launcher runs
`full_train -> final_epe -> report`. It is **not launched by Phase1**:

```bash
cd /home/bupt/projects/SLiM-master
bash launch_reliability_phase2.sh --config configs/reliability_phase2_20261115.json
```

Resume the failed stage and continue the pipeline:

```bash
bash launch_reliability_phase2.sh --config configs/reliability_phase2_20261115.json --resume
```

Resume only final EPE after completed training:

```bash
bash launch_reliability_phase2.sh --config configs/reliability_phase2_20261115.json --stage final_epe --resume
/home/bupt/miniconda3/envs/slim/bin/python reliability_phase2.py report --config configs/reliability_phase2_20261115.json
```

For syntax/path review only, add `--dry-run`; it starts no workers and writes
no output. The launcher records PID, logs and heartbeat, stops on stage failure,
and records the exact affected-stage resume command. Storage faults stop the
private worker process group without shutting down WSL.

Formal launch requires all eight full Phase1 checks, complete368-index audit,
disjoint final scenes, exact code/config/weight/manifest/statistics/plan identity,
resolved label reviews, and explicit protocol and storage cost. An engineering
smoke receipt cannot bypass these checks. Fixed-validation file changes pause
validation. Invalid training visits retain their planned positions without
replacement; more than20 skips among the first100 planned visits pauses training.

```bash
/home/bupt/miniconda3/envs/slim/bin/python -m unittest discover -s tests -p 'test_reliability*.py' -v
```
