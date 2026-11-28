#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED=1
if [[ -z "${PYTORCH_CUDA_ALLOC_CONF:-}" && -z "${PYTORCH_ALLOC_CONF:-}" ]]; then
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
fi
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd -- "$here/../.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-$here/outputs/shared/cache/pyc}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$here/outputs/shared/cache/triton}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-$here/outputs/shared/cache/cuda}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$here/outputs/shared/cache/inductor}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$here/outputs/shared/cache/matplotlib}"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
py="${PYTHON:-python}"
mode=smoke
run_id=pro5000
readonly argv=("$@")
monitor=0
help=0
while (($#)); do
    case "$1" in
        --python) py="${2:?--python needs a value}"; shift 2 ;;
        --python=*) py="${1#*=}"; shift ;;
        --gpu) export CUDA_VISIBLE_DEVICES="${2:?--gpu needs a value}"; shift 2 ;;
        --gpu=*) export CUDA_VISIBLE_DEVICES="${1#*=}"; shift ;;
        --mode) mode="${2:?--mode needs a value}"; shift 2 ;;
        --mode=*) mode="${1#*=}"; shift ;;
        --run-id) run_id="${2:?--run-id needs a value}"; shift 2 ;;
        --run-id=*) run_id="${1#*=}"; shift ;;
        --monitor) monitor=1; shift ;;
        --help|-h) help=1; shift ;;
        *) shift ;;
    esac
done
module=experiments.lowtex_pro5000_hr_aug.training.pipeline
if ((monitor || help)); then
    exec "$py" -u -m "$module" "${argv[@]}"
fi
[[ "$mode" =~ ^(smoke|short|medium|full)$ && "$run_id" =~ ^[A-Za-z0-9_-][A-Za-z0-9_.-]{0,79}$ ]] || {
    echo "Invalid mode or run ID" >&2
    exit 2
}
logdir="$here/logs/$mode/$run_id"
mkdir -p -- "$logdir"
cmd=("$py" -u -m "$module" "${argv[@]}")
if command -v stdbuf >/dev/null 2>&1; then cmd=(stdbuf -oL -eL "${cmd[@]}"); fi
"${cmd[@]}" 2>&1 | tee -a "$logdir/terminal-$(date -u +%Y%m%dT%H%M%SZ)-$$.log"
