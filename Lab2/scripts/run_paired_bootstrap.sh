#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
tag="${1:-paired_hash0}"
source "$ROOT/scripts/env.sh"
cd "$ROOT"
set +e
"$ROOT/venv/bin/python" "$ROOT/scripts/paired_bootstrap.py" "$tag" \
    2>&1 | tee "$ROOT/logs/paired_bootstrap_${tag}.log"
status="${PIPESTATUS[0]}"
printf '%s\n' "$status" > "$ROOT/results/regbench_paired_bootstrap_${tag}_EXIT_CODE"
exit "$status"
