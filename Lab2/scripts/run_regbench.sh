#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/env.sh"

method="${1:?attention method required}"
examples="${2:?training example count required}"
seed="${3:?seed required}"
epochs="${4:-200}"
test="${5:-true}"
run_name="${method}_${examples}_seed${seed}_e${epochs}"
if [[ -n "${6:-}" ]]; then
    run_name="${run_name}_${6}"
fi
run_dir="$ROOT/runs/regbench/$run_name"
checkpoint_dir="$ROOT/checkpoints/regbench/$run_name"
log="$ROOT/logs/regbench_${run_name}.log"

if [[ -e "$run_dir" || -e "$log" ]]; then
    printf 'Run already exists: %s\n' "$run_name" >&2
    exit 1
fi
mkdir -p "$run_dir" "$checkpoint_dir"
printf 'PYTHONHASHSEED=%s\n' "${PYTHONHASHSEED:-unset}" > "$run_dir/env.txt"
cd "$ROOT/seq_icl"
set +e
"$ROOT/venv/bin/python" train.py \
    experiment=dfa/mm \
    "dataset.num_examples=$examples" \
    dataset.num_test_examples=1000 \
    model.n_layer=2 \
    model.n_embd=128 \
    model.n_head=8 \
    "+model.method=$method" \
    optimizer.lr=1e-4 \
    optimizer.weight_decay=0.01 \
    "train.seed=$seed" \
    "trainer.max_epochs=$epochs" \
    "train.test=$test" \
    "hydra.run.dir=$run_dir" \
    "callbacks.model_checkpoint.dirpath=$checkpoint_dir" \
    '~wandb' 2>&1 | tee "$log"
status="${PIPESTATUS[0]}"
printf '%s\n' "$status" > "$run_dir/EXIT_CODE"
exit "$status"
