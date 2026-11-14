#!/bin/bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PY=/home/bupt/miniconda3/envs/slim/bin/python
CONFIG=""
DRY=0
RESUME=0
STAGE=pipeline
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    --resume) RESUME=1; shift ;;
    --stage) STAGE="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$CONFIG" ]] || { echo "--config is required" >&2; exit 2; }
CONFIG="$("$PY" -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve())' "$CONFIG")"
ARGS=("$STAGE" --config "$CONFIG")
[[ "$RESUME" -eq 0 ]] || ARGS+=(--resume)
if [[ "$DRY" -eq 1 ]]; then
  exec "$PY" "$ROOT/reliability_phase2.py" "${ARGS[@]}" --dry-run
fi
RUN="$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["run_dir"])' "$CONFIG")"
[[ -f "$RUN/phase1_ready.json" ]] || { echo "Phase1 receipt missing" >&2; exit 3; }
# The supervisor performs the exact identity and readiness check before training.
nohup "$PY" -u "$ROOT/reliability_phase2.py" "${ARGS[@]}" >"$RUN/launch.log" 2>&1 < /dev/null &
PID=$!
printf '%s\n' "$PID" >"$RUN/launch.pid"
printf 'PID=%s LOG=%s\n' "$PID" "$RUN/launch.log"
