#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
gpu="${1:?GPU index required}"
run="${2:?run name required}"
if [[ -e "$ROOT/runs/regbench/$run/EVAL_EXIT_CODE" ]]; then
    printf 'Evaluation already completed: %s\n' "$run" >&2
    exit 1
fi
env_file="$ROOT/runs/regbench/$run/env.txt"
if [[ -f "$env_file" ]]; then
    recorded_hash_seed="$(sed -n 's/^PYTHONHASHSEED=//p' "$env_file")"
    if [[ "$recorded_hash_seed" =~ ^[0-9]+$ ]]; then
        export PYTHONHASHSEED="$recorded_hash_seed"
    elif [[ "$recorded_hash_seed" == "unset" ]]; then
        unset PYTHONHASHSEED
    elif [[ "$recorded_hash_seed" != "unset" ]]; then
        printf 'Invalid PYTHONHASHSEED in %s\n' "$env_file" >&2
        exit 1
    fi
fi
printf 'PYTHONHASHSEED=%s\n' "${PYTHONHASHSEED:-unset}" > "$ROOT/runs/regbench/$run/eval_env.txt"

CUDA_VISIBLE_DEVICES="$gpu" setsid nohup bash -c '
    root="$1"
    run="$2"
    cd "$root"
    source "$root/env.sh"
    set +e
    "$root/venv/bin/python" "$root/evaluate_regbench.py" "$run" 2>&1 | tee "$root/logs/eval_regbench_$run.log"
    status="${PIPESTATUS[0]}"
    printf "%s\n" "$status" > "$root/runs/regbench/$run/EVAL_EXIT_CODE"
    exit "$status"
' _ "$ROOT" "$run" > "$ROOT/logs/launcher_eval_$run.log" 2>&1 < /dev/null &
pid=$!
disown "$pid"
printf '%s\n' "$pid" > "$ROOT/logs/pid_eval_$run.txt"
printf 'Launched evaluation %s on GPU %s, PID %s\n' "$run" "$gpu" "$pid"
