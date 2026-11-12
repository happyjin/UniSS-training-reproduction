#!/usr/bin/env bash
# PMC (thinker_pmc_prereg_v1 §1.5): VE1's DART manifest command with --lw-mode pmc (CPU only, ~5 s); the plan is VE1's
# (plan seed 20261109); the mode only enters the in-lattice statistic (h is v2p's), so the draws and rows equal VE1's.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
export PYTHONPATH=.:third_party/Megatron-LM CUDA_VISIBLE_DEVICES=
PY=/opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python
C=eval_outputs/mist_w3po_v1/crf
L=logs/mist_w3po_v1/pmc
$PY -m experiments.mist_w3po_v1.crf.dart_aug --manifest --lw-mode pmc --sched $C/sched_t5_merge3_w10_v1 \
    --out $C/pmc_dart_merge3_w10_v1 --workers 32 > $L/manifest_pmc_merge3_w10.log 2>&1
echo "PMC_MANIFEST_EXIT rc=$?" >> $L/manifest_pmc_merge3_w10.log
