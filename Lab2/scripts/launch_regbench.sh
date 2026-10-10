#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
gpu="${1:?GPU index required}"
method="${2:?attention method required}"
examples="${3:?training example count required}"
seed="${4:?seed required}"
epochs="${5:-200}"
test="${6:-true}"
tag="${7:-}"
name="${method}_${examples}_seed${seed}_e${epochs}${tag:+_$tag}"

cd "$ROOT"
CUDA_VISIBLE_DEVICES="$gpu" setsid nohup bash "$ROOT/scripts/run_regbench.sh" \
    "$method" "$examples" "$seed" "$epochs" "$test" "$tag" \
    > "$ROOT/logs/launcher_${name}.log" 2>&1 < /dev/null &
pid=$!
disown "$pid"
printf '%s\n' "$pid" > "$ROOT/logs/pid_${name}.txt"
printf 'Launched %s on GPU %s, PID %s\n' "$name" "$gpu" "$pid"
