#!/bin/bash
# Apply local patches to the installed verl + vllm packages.
#
# Idempotent: re-running when already applied is a no-op with exit 0.
# Run after `uv sync` or any reinstall that would clobber the patches.
#
# Patches applied:
#   patches/verl_5609_dp_fix.patch — verl: DP args weren't propagated to vllm
#     for dense (EP=1) models. Fixed in upstream PR #5609, missed 0.7.1 tag.
#     Remove once `verl>=0.7.2` is on PyPI with the fix included.
#   patches/vllm_dp_external_executor_fix.patch — vllm: skip DP rank-adjust
#     when the distributed_executor_backend is a class (not a known string).
#     Fixes the "DP adjusted local rank N is out of bounds" assert with
#     verl's ExternalZeroMQDistributedExecutor. See verl issue #4926 + vLLM
#     PR #32816. Remove once a vllm release ships that allowlist fix.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

if ! python -c "import verl" 2>/dev/null; then
  echo "[patch_verl] verl is not importable in the current Python env." >&2
  echo "[patch_verl] Activate the venv first (source .venv/bin/activate)." >&2
  exit 1
fi
if ! python -c "import vllm" 2>/dev/null; then
  echo "[patch_verl] vllm is not importable in the current Python env." >&2
  echo "[patch_verl] Activate the venv first (source .venv/bin/activate)." >&2
  exit 1
fi

VERL_DIR="$(python -c 'import verl, os; print(os.path.dirname(verl.__file__))')"
SITE_PACKAGES="$(dirname "$VERL_DIR")"
echo "[patch_verl] verl install:      $VERL_DIR"
echo "[patch_verl] verl version:      $(python -c 'import verl; print(verl.__version__)' 2>/dev/null || echo unknown)"
echo "[patch_verl] vllm version:      $(python -c 'from importlib.metadata import version; print(version("vllm"))' 2>/dev/null || echo unknown)"

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
apply_patch "$PROJECT_ROOT/patches/vllm_dp_external_executor_fix.patch"

echo "[patch_verl] done."
