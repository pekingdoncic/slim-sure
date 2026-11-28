#!/usr/bin/env bash
set -euo pipefail
exp="$(cd "$(dirname "$0")/.." && pwd)"
cd "$exp/../.."
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
export TRITON_CACHE_DIR="$exp/cache/triton" CUDA_CACHE_PATH="$exp/cache/cuda" MPLCONFIGDIR="$exp/cache/mpl" XDG_CACHE_HOME="$exp/cache" TORCH_HOME="$exp/cache/torch"
if [[ -z "${PYTORCH_CUDA_ALLOC_CONF:-}${PYTORCH_ALLOC_CONF:-}" ]]; then export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; fi
py="${SLIM_PYTHON:-${CONDA_PREFIX:-$HOME/miniconda3/envs/slim}/bin/python}"
if [[ "${1:-}" == "-m" && "${2:-}" == "experiments.lowtex_e2e_v1.screening.train" ]]; then
  args=("$@"); mode="screening"; experiment=""
  for ((i=2;i<${#args[@]};i++)); do
    case "${args[i]}" in
      --mode) mode="${args[i+1]}";;
      --exp) experiment="${args[i+1]}";;
      --help|-h) mode="help";;
    esac
  done
  if [[ "$mode" == screening && -n "$experiment" ]]; then
    "$py" -B -m experiments.lowtex_e2e_v1.screening.admission --exp "$experiment"
  fi
fi
exec "$py" -B "$@"
