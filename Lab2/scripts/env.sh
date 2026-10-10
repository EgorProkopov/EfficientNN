#!/usr/bin/env bash
set -euo pipefail

source /etc/profile.d/wm.sh
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export HOME="$ROOT"
export XDG_CACHE_HOME="$ROOT/cache"
export XDG_CONFIG_HOME="$ROOT/cache/config"
export XDG_DATA_HOME="$ROOT/cache/data"
export TMPDIR="$ROOT/tmp"
export PIP_CACHE_DIR="$ROOT/cache/pip"
export PIP_CONFIG_FILE=/dev/null
export HF_HOME="$ROOT/cache/huggingface"
export TORCH_HOME="$ROOT/cache/torch"
export TORCH_EXTENSIONS_DIR="$ROOT/cache/torch_extensions"
export TORCHINDUCTOR_CACHE_DIR="$ROOT/cache/torchinductor"
export TRITON_CACHE_DIR="$ROOT/cache/triton"
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
export CUDA_CACHE_PATH="$ROOT/cache/cuda"
export C_INCLUDE_PATH="/usr/local/cuda/include${C_INCLUDE_PATH:+:$C_INCLUDE_PATH}"
export MPLCONFIGDIR="$ROOT/cache/matplotlib"
export WANDB_DIR="$ROOT/cache/wandb"
export WANDB_CACHE_DIR="$ROOT/cache/wandb"
export WANDB_CONFIG_DIR="$ROOT/cache/wandb"
export PYTHONPYCACHEPREFIX="$ROOT/cache/pycache"
export PYTHONPATH="$ROOT/upstream/Library:$ROOT/seq_icl"
export HYDRA_FULL_ERROR=1
mkdir -p "$ROOT/cache" "$ROOT/tmp" "$ROOT/logs" "$ROOT/checkpoints" "$ROOT/data" "$ROOT/figures"
