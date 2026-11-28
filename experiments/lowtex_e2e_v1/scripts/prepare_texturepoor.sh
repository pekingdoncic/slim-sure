#!/usr/bin/env bash
set -eu -o pipefail
exp="$(cd "$(dirname "$0")/.." && pwd)"
cd "$exp/../.."
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1
exec "${SLIM_PYTHON:-/home/bupt/miniconda3/envs/slim/bin/python}" -B -m experiments.lowtex_e2e_v1.evaluation.texturepoor_sfm.build_pairs
