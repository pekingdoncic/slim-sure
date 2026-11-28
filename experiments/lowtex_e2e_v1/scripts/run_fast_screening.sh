#!/usr/bin/env bash
set -euo pipefail
exp="$(cd "$(dirname "$0")/.." && pwd)"
case "${1:-}" in
  --status) cat "$exp/reports/fast_screening/state.json";;
  --help|-h) echo "run_workflow.sh [--status | train --exp MODEL --mode smoke|screening|extended [--cycles 3|4|5]]";;
  train) shift; exec "$exp/scripts/screen_python.sh" -m experiments.lowtex_e2e_v1.screening.train "$@";;
  "") exec "$exp/scripts/screen_python.sh" -m experiments.lowtex_e2e_v1.screening.workflow ;;
  *) echo "Usage: run_workflow.sh [--help | --status | train ...]" >&2; exit 2 ;;
esac
