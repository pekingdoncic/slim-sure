#!/usr/bin/env bash
set -euo pipefail
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd -- "$here/../.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-$here/outputs/shared/cache/pyc}"
export PYTHONDONTWRITEBYTECODE=1
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$here/outputs/shared/cache/triton}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-$here/outputs/shared/cache/cuda}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$here/outputs/shared/cache/matplotlib}"
py="${PYTHON:-python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
while (($#)); do
    case "$1" in
        --python) py="${2:?}"; shift 2 ;;
        --weights) export CCS_WEIGHTS="${2:?}"; shift 2 ;;
        --gpu) export CUDA_VISIBLE_DEVICES="${2:?}"; shift 2 ;;
        --vram-cap-gib) export CCS_VRAM_CAP_GIB="${2:?}"; shift 2 ;;
        *) echo "unknown test argument $1" >&2; exit 2 ;;
    esac
done
"$py" -m compileall -q experiments/lowtex_pro5000_hr_aug
for file in "$here"/scripts/*.sh; do bash -n "$file"; done
mkdir -p -- "$here/outputs/shared/cache/pytest"
test_tmp="$(mktemp -d "$here/outputs/shared/cache/pytest/session-XXXXXXXX")"
test_opts=("$here/tests" -o "cache_dir=$here/outputs/shared/cache/pytest")
CUDA_VISIBLE_DEVICES="" "$py" -m pytest "${test_opts[@]}" --basetemp "$test_tmp/cpu" -m "not gpu"
"$py" -m pytest "${test_opts[@]}" --basetemp "$test_tmp/gpu" -m gpu
