#!/usr/bin/env bash
# helper 1 again at 4 cards x 20 GiB (17 GiB offloaded to CPU); output hyp_r3h1.jsonl
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
export MIST_V7_KIND=decode MIST_V7_JOB=mist8m_ali_lag2_r3h1b
CUDA_VISIBLE_DEVICES=0,1,3,4 PYTHONPATH=.:third_party/Megatron-LM /opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python -m experiments.uniss_qwen3_omni_v1.evaluation.gstar_si_stream_online_v6s --thinker eval_outputs/uniss_qwen3_omni_v1/thinker_gstar_s2dl_zh2en_3k_20261030/thinker_merged --windows eval_outputs/uniss_qwen3_omni_v1/gstar_v8_20261103/runs/on_ali_S2dllag2/helper_r3h1.windows.jsonl --out eval_outputs/uniss_qwen3_omni_v1/gstar_v8_20261103/runs/on_ali_S2dllag2/hyp_r3h1.jsonl --policy lag --lag-ms 2000 --window 1 --post-key lowlat --dir zh2en --batch 16 --max-gib 20 > eval_outputs/uniss_qwen3_omni_v1/gstar_v8_20261103/runs/on_ali_S2dllag2/infer_r3h1.log 2>&1; echo "EXIT r3h1b $?"
