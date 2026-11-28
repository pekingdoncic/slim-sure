#!/usr/bin/env bash
set -euo pipefail
dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd -- "$dir/../.."
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
py="${PYTHON:-python}"
args=("$@")
while (($#)); do
    case "$1" in
        --python) py="${2:?}"; shift 2 ;;
        --gpu) export CUDA_VISIBLE_DEVICES="${2:?}"; shift 2 ;;
        *) shift ;;
    esac
done
exec "$py" -u -m experiments.lowtex_pro5000_hr_aug.training.preflight "${args[@]}"
