#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
revision=191158129c8dab3637769988e336009a9bc5d4f2
base="https://raw.githubusercontent.com/facebookresearch/MemoryMosaics/$revision/BabiStories/data"
dest="$ROOT/data/BabiStories/archive"
export HOME="$ROOT" XDG_CACHE_HOME="$ROOT/cache" TMPDIR="$ROOT/tmp"
mkdir -p "$dest" "$ROOT/tmp"

for part in 001 002 003 004 005; do
    file="$dest/babistories-dataset.7z.$part"
    case "$part" in
        001) expected=ce46de124da5392bd170370a405d645b4ff93603 ;;
        002) expected=cecd7247593412cf351780365aca9dd07096e501 ;;
        003) expected=e4ca7ce0d9656c666dfb2475e00c14653ce0c94a ;;
        004) expected=fbe941304c7e1dac1d9da5af88577a5e398d0d10 ;;
        005) expected=cfdbcb3cf27778c374dddebb7ab55b4d98f2b65e ;;
    esac
    if [[ ! -f "$file" || "$(git hash-object "$file")" != "$expected" ]]; then
        curl --http1.1 --fail --location --silent --show-error --retry 5 --continue-at - \
            --output "$file" "$base/babistories-dataset.7z.$part"
    fi
    actual="$(git hash-object "$file")"
    [[ "$actual" == "$expected" ]] || { printf 'SHA mismatch for %s\n' "$file" >&2; exit 1; }
    printf '%s %s bytes SHA %s\n' "$part" "$(wc -c < "$file")" "$actual"
done
