#!/usr/bin/env bash
# v4a regression runs (CPU only), one log per test; noclobber logs
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
export PYTHONPATH=.:third_party/Megatron-LM CUDA_VISIBLE_DEVICES= PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
PY=/opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python
D=logs/mist_w3po_v1/v4a_regress
for t in test_j1v2 test_j1 test_joint_speech_crf test_train_crf_sft test_latent_write_crf test_dart_aug test_d1 test_pmc test_pmc_noc test_merge_read_segments; do
  [ -e $D/$t.log ] && { echo "exists $D/$t.log"; continue; }
  $PY -m experiments.mist_w3po_v1.crf.$t > $D/$t.log 2>&1
  echo "$t rc=$?"
done
echo REGRESS_DONE
