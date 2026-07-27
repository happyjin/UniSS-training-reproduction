#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
"${ROOT}/e0_baselines/run_2gpu.sh" --smoke
"${ROOT}/e1_continued_sft/run_2gpu.sh" --smoke
"${ROOT}/e2_grpo_g4/run_2gpu.sh" --smoke
"${ROOT}/e3_grpo_g8/run_2gpu.sh" --smoke
