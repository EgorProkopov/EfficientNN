#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
method="${1:?attention method required}"
tag="${2:-}"
name="${method}${tag:+_$tag}"
out_dir="$ROOT/runs/lm_smoke${tag:+_$tag}"
source "$ROOT/env.sh"
mkdir -p "$out_dir"
cd "$ROOT/upstream/Library"
set +e
"$ROOT/venv/bin/python" train_memory_mosaics.py \
    --method "$method" \
    --out_dir "$out_dir" \
    --compile false \
    --batch_size 1 \
    --n_layer 1 \
    --max_iters 1 \
    --eval_interval 1 \
    --eval_iters 1 \
    --log_interval 1 \
    --save_checkpoint_interval 1000 \
    2>&1 | tee "$ROOT/logs/lm_smoke_${name}.log"
status="${PIPESTATUS[0]}"
printf '%s\n' "$status" > "$out_dir/${method}_EXIT_CODE"
exit "$status"
