#!/usr/bin/env bash
set -eu -o pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$here/logs/audit"
log="$(mktemp "$here/logs/audit/pool.XXXXXXXX.log")"
set +e
"${SLIM_PYTHON:-/home/bupt/miniconda3/envs/slim/bin/python}" -B "$here/scripts/audit_pool.py" >"$log" 2>&1
rc=$?
set -e
echo "$rc" > "$log.exitcode"
tail -n 2 "$log"
exit "$rc"
