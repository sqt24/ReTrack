#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${STORAGE_ROOT:-}" ]]; then
  ROOT="$STORAGE_ROOT"
else
  ROOT="."
fi

python scripts/check_assets.py --storage-root "$ROOT"
