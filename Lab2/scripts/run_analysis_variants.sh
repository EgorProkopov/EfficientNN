#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
tag="${1:-}"
suffix="${tag:+_$tag}"
source "$ROOT/scripts/env.sh"
cd "$ROOT"
set +e
"$ROOT/venv/bin/python" "$ROOT/scripts/analyze_regbench_variants.py" ${tag:+"$tag"} \
    2>&1 | tee "$ROOT/logs/analyze_regbench_1000_eight_methods${suffix}.log"
status="${PIPESTATUS[0]}"
printf '%s\n' "$status" > "$ROOT/results/regbench_1000_eight_methods${suffix}_ANALYSIS_EXIT_CODE"
exit "$status"
