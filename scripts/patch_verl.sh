#!/bin/bash
# Apply local patches to the installed verl package.
#
# Idempotent: re-running when already applied is a no-op with exit 0.
# Run after `uv sync` or any reinstall that would clobber the patch.
#
# Patches applied:
#   patches/verl_5609_dp_fix.patch — DP>1 hang fix for dense-model rollout (PR #5609).
#     Remove once `verl>=0.7.2` is on PyPI with the fix included.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

if ! python -c "import verl" 2>/dev/null; then
  echo "[patch_verl] verl is not importable in the current Python env." >&2
  echo "[patch_verl] Activate the venv first (source .venv/bin/activate)." >&2
  exit 1
fi

VERL_DIR="$(python -c 'import verl, os; print(os.path.dirname(verl.__file__))')"
SITE_PACKAGES="$(dirname "$VERL_DIR")"
echo "[patch_verl] verl install:      $VERL_DIR"
echo "[patch_verl] verl version:      $(python -c 'import verl; print(verl.__version__)' 2>/dev/null || echo unknown)"

apply_patch() {
  local patch_file="$1"
  local name="$(basename "$patch_file")"
  # Idempotency check: if the forward patch is already a no-op, we're done.
  if patch --dry-run -p1 -d "$SITE_PACKAGES" -R < "$patch_file" >/dev/null 2>&1; then
    echo "[patch_verl] $name: already applied, skipping."
    return 0
  fi
  if ! patch --dry-run -p1 -d "$SITE_PACKAGES" < "$patch_file" >/dev/null 2>&1; then
    echo "[patch_verl] $name: dry-run failed — verl source has diverged from what the patch expects." >&2
    echo "[patch_verl] $name: run manually to see context:" >&2
    echo "    patch -p1 -d $SITE_PACKAGES < $patch_file" >&2
    return 1
  fi
  patch -p1 -d "$SITE_PACKAGES" < "$patch_file"
  echo "[patch_verl] $name: applied."
}

apply_patch "$PROJECT_ROOT/patches/verl_5609_dp_fix.patch"

echo "[patch_verl] done."
