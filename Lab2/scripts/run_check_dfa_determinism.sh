#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/env.sh"
cd "$ROOT"
set +e
for label in first second; do
    PYTHONHASHSEED=0 "$ROOT/venv/bin/python" "$ROOT/check_dfa_determinism.py" "$label" \
        2>&1 | tee "$ROOT/logs/check_dfa_determinism_${label}.log"
    status="${PIPESTATUS[0]}"
    if [[ "$status" -ne 0 ]]; then
        printf '%s\n' "$status" > "$ROOT/results/dfa_determinism_EXIT_CODE"
        exit "$status"
    fi
done
printf '0\n' > "$ROOT/results/dfa_determinism_EXIT_CODE"
