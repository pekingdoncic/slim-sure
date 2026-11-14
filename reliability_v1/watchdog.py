"""Timeout provenance for the existing canonical geometry worker."""
import csv
import json
import math
from pathlib import Path
from .runtime import sha,atomic_json

def geometry_timeout(run):
    source=Path("/home/bupt/projects/hybrid_match/outputs/m2b2_pilot_diagnosis_20261112T213600/geometry_controller.json")
    costs=Path("/home/bupt/projects/hybrid_match/outputs/m2b2_pilot_diagnosis_20261112T213600/geometry_cost_records.csv")
    expected="7b54623ef2892eaeb965822cea23d12d5036d992ecb950c9d464df9a5668063c"
    if sha(source)!=expected:raise RuntimeError("Historical geometry watchdog source changed")
    record=dict(seconds=242,source=str(source),source_sha256=expected,committed_megadepth_workers=60,
        measured_worker_p95_seconds=21.1311633919,max_worker_seconds=31.18335408,
        formula="max(120, ceil(10 * historical_worker_p95 + 30 cold-import seconds))",
        cost_source=str(costs),cost_source_sha256=sha(costs),
        scope="Historical R1024/R1200 watchdog provenance only; not new R832 performance evidence")
    atomic_json(Path(run)/"geometry_watchdog.json",record)
    return record
