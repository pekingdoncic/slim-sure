#!/usr/bin/env bash
set -euo pipefail
exp="$(cd "$(dirname "$0")/.." && pwd)"
cd "$exp/../.."
export PYTHONDONTWRITEBYTECODE=1
if [[ -f "$exp/reports/fast_screening/legacy_r832/transition.json" ]]; then
  exec "$exp/scripts/run_fast_screening.sh" "$@"
fi
py="${SLIM_PYTHON:-${CONDA_PREFIX:-$HOME/miniconda3/envs/slim}/bin/python}"
case "${1:-}" in
  --status) exec "$py" -B "$exp/scripts/workflow.py" --status ;;
  --phase02-worker) exec "$py" -B "$exp/scripts/workflow.py" --phase02-only ;;
  "")
    nohup "$py" -B "$exp/scripts/archive_development.py" >> "$exp/logs/workflow/archive_development.log" 2>&1 < /dev/null &
    exec "$py" -B "$exp/scripts/stage_supervisor.py"
    ;;
  *) echo "Usage: run_workflow.sh [--status]" >&2; exit 2 ;;
esac
