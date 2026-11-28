#!/usr/bin/env bash
set -euo pipefail
exp="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$exp/logs/fast_screening"
exec bash "$exp/scripts/run_workflow.sh" >> "$exp/logs/fast_screening/owner.log" 2>&1
