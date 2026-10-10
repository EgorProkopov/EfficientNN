#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
examples="${1:?training example count required}"
tag="${2:-}"
suffix="${tag:+_$tag}"
source "$ROOT/scripts/env.sh"
mkdir -p "$ROOT/results"
cd "$ROOT"
set +e
"$ROOT/venv/bin/python" "$ROOT/scripts/analyze_regbench.py" "$examples" ${tag:+"$tag"} \
    2>&1 | tee "$ROOT/logs/analyze_regbench_${examples}${suffix}.log"
status="${PIPESTATUS[0]}"
printf '%s\n' "$status" > "$ROOT/results/regbench_${examples}${suffix}_ANALYSIS_EXIT_CODE"
exit "$status"
