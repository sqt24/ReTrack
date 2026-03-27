#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${STORAGE_ROOT:-}" ]]; then
  ROOT="$STORAGE_ROOT"
elif [[ -n "${NAS:-}" ]]; then
  ROOT="$NAS/retrack"
else
  ROOT="."
fi

python scripts/check_assets.py --storage-root "$ROOT"
