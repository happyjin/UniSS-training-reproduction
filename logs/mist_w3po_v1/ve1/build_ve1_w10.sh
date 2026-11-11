#!/usr/bin/env bash
# VE1 data (thinker_v2p_evidence_ve1_prereg_v1 §1): T5's evidence-time labels at the registered merge3 granularity on
# the 1.0 s grid + the DART manifest (same draws as every T run; lattice mode v2p only for its in-lattice statistic). CPU.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
export PYTHONPATH=.:third_party/Megatron-LM CUDA_VISIBLE_DEVICES=
PY=/opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python
C=eval_outputs/mist_w3po_v1/crf
L=logs/mist_w3po_v1/ve1
$PY -W ignore -m experiments.mist_w3po_v1.crf.t5_evidence_labels --build --granularity merge3 --grid 1000 --workers 40 \
    --align-dir $C/t5_align_v1/train --out $C/sched_t5_merge3_w10_v1 > $L/build_merge3_w10.log 2>&1 \
  && $PY -m experiments.mist_w3po_v1.crf.dart_aug --manifest --lw-mode v2p --sched $C/sched_t5_merge3_w10_v1 \
    --out $C/ve1_dart_merge3_w10_v1 --workers 32 > $L/manifest_merge3_w10.log 2>&1
echo "VE1_BUILD_EXIT rc=$?" >> $L/build_merge3_w10.log
