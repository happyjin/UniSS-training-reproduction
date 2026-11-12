#!/usr/bin/env bash
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
export PYTHONPATH=.:third_party/Megatron-LM CUDA_VISIBLE_DEVICES= PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
/opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python -m experiments.mist_w3po_v1.crf.j1v2_precheck --out eval_outputs/mist_w3po_v1/crf/j1v2_precheck_v1.json --procs 64 --steps4 200 --pc1-logs logs/mist_w3po_v1/v4a_regress/test_j1v2.log logs/mist_w3po_v1/v4a_regress/test_j1.log logs/mist_w3po_v1/v4a_regress/test_joint_speech_crf.log logs/mist_w3po_v1/v4a_regress/test_train_crf_sft_patched.log logs/mist_w3po_v1/v4a_regress/test_latent_write_crf.log logs/mist_w3po_v1/v4a_regress/test_dart_aug.log
echo J1V2_PRECHECK_EXIT rc=$?
