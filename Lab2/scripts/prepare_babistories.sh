#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/env.sh"
cd "$ROOT/data/BabiStories/extracted"
set +e
"$ROOT/venv/bin/python" "$ROOT/upstream/Library/memory_mosaics/data/BabiStories/prepare.py" \
    2>&1 | tee "$ROOT/logs/prepare_babistories.log"
status="${PIPESTATUS[0]}"
printf '%s\n' "$status" > "$ROOT/data/BabiStories/PREPARE_EXIT_CODE"
exit "$status"
