#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MEGATRON_ROOT="${MEGATRON_ROOT:-${REPO_ROOT}/third_party/Megatron-LM}"
PATCH_FILE="${MEGATRON_FULL_VALIDATION_PATCH:-${REPO_ROOT}/training/patches/megatron_full_validation_scalar_eval_iters.patch}"
TARGET_FILE="${MEGATRON_ROOT}/megatron/training/training.py"

buggy_line="eval_iters = torch.tensor(args.eval_iters, dtype=torch.long, device='cuda')"
fixed_line="eval_iters = torch.tensor(eval_iters, dtype=torch.long, device='cuda')"

[[ -f "${TARGET_FILE}" ]] || { echo "Missing Megatron training file: ${TARGET_FILE}" >&2; exit 1; }
[[ -f "${PATCH_FILE}" ]] || { echo "Missing Megatron patch: ${PATCH_FILE}" >&2; exit 1; }

if rg -F -- "${fixed_line}" "${TARGET_FILE}" >/dev/null; then
  echo "Megatron full-validation scalar eval_iters fix already applied."
  exit 0
fi

if ! rg -F -- "${buggy_line}" "${TARGET_FILE}" >/dev/null; then
  echo "Megatron full-validation code does not match the pinned supported version" >&2
  exit 1
fi

git -C "${MEGATRON_ROOT}" apply --check "${PATCH_FILE}"
git -C "${MEGATRON_ROOT}" apply "${PATCH_FILE}"
rg -F -- "${fixed_line}" "${TARGET_FILE}" >/dev/null || {
  echo "Megatron full-validation patch verification failed" >&2
  exit 1
}
echo "Applied Megatron full-validation scalar eval_iters fix."
