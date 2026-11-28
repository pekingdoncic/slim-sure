#!/usr/bin/env bash
set -euo pipefail
exp="$(cd "$(dirname "$0")/.." && pwd)"
cd "$exp/../.."
if [[ ! -v PYTORCH_CUDA_ALLOC_CONF && ! -v PYTORCH_ALLOC_CONF ]]; then
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
fi
args=("$@")
for ((i=0;i<${#args[@]};i++)); do
  if [[ "${args[i]}" == "--gpu" ]]; then export CUDA_VISIBLE_DEVICES="${args[i+1]}"; fi
  if [[ "${args[i]}" == --gpu=* ]]; then export CUDA_VISIBLE_DEVICES="${args[i]#*=}"; fi
done
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
export TRITON_CACHE_DIR="$exp/cache/triton" CUDA_CACHE_PATH="$exp/cache/cuda" MPLCONFIGDIR="$exp/cache/mpl" XDG_CACHE_HOME="$exp/cache" TORCH_HOME="$exp/cache/torch"
py="${SLIM_PYTHON:-${CONDA_PREFIX:-$HOME/miniconda3/envs/slim}/bin/python}"
mkdir -p "$exp/logs/phase02"
log="$(mktemp "$exp/logs/phase02/entry.XXXXXXXX.log")"
set +e
"$py" -B -m experiments.lowtex_e2e_v1.training.train "$@" 2>&1 | tee "$log"
rc=${PIPESTATUS[0]}
set -e
printf '%s\n' "$rc" > "$log.exitcode"
exit "$rc"
