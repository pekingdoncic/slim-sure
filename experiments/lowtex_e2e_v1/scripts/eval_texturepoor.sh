#!/usr/bin/env bash
set -eu -o pipefail
exp="$(cd "$(dirname "$0")/.." && pwd)"
cd "$exp/../.."
export PYTHONDONTWRITEBYTECODE=1
if [[ ! -v PYTORCH_CUDA_ALLOC_CONF && ! -v PYTORCH_ALLOC_CONF ]]; then
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
fi
exec "${SLIM_PYTHON:-/home/bupt/miniconda3/envs/slim/bin/python}" -B -m experiments.lowtex_e2e_v1.evaluation.texturepoor_sfm.evaluate "$@"
