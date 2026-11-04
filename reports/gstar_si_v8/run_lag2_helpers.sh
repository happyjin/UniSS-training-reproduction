#!/usr/bin/env bash
# reverse helpers for the AliMeeting Lag2 resume shards (amendment 3 exploratory run); outputs hyp_r3h<k>
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
export MIST_V7_KIND=decode
( export MIST_V7_JOB=mist8m_ali_lag2_r3h0; CUDA_VISIBLE_DEVICES=5,6,7 PYTHONPATH=.:third_party/Megatron-LM /opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python -m experiments.uniss_qwen3_omni_v1.evaluation.gstar_si_stream_online_v6s --thinker eval_outputs/uniss_qwen3_omni_v1/thinker_gstar_s2dl_zh2en_3k_20261030/thinker_merged --windows eval_outputs/uniss_qwen3_omni_v1/gstar_v8_20261103/runs/on_ali_S2dllag2/helper_r3h0.windows.jsonl --out eval_outputs/uniss_qwen3_omni_v1/gstar_v8_20261103/runs/on_ali_S2dllag2/hyp_r3h0.jsonl --policy lag --lag-ms 2000 --window 1 --post-key lowlat --dir zh2en --batch 16 --max-gib 23 > eval_outputs/uniss_qwen3_omni_v1/gstar_v8_20261103/runs/on_ali_S2dllag2/infer_r3h0.log 2>&1; echo "EXIT r3h0 $?" ) &
sleep 60
( export MIST_V7_JOB=mist8m_ali_lag2_r3h1; CUDA_VISIBLE_DEVICES=0,1,3,4 PYTHONPATH=.:third_party/Megatron-LM /opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python -m experiments.uniss_qwen3_omni_v1.evaluation.gstar_si_stream_online_v6s --thinker eval_outputs/uniss_qwen3_omni_v1/thinker_gstar_s2dl_zh2en_3k_20261030/thinker_merged --windows eval_outputs/uniss_qwen3_omni_v1/gstar_v8_20261103/runs/on_ali_S2dllag2/helper_r3h1.windows.jsonl --out eval_outputs/uniss_qwen3_omni_v1/gstar_v8_20261103/runs/on_ali_S2dllag2/hyp_r3h1.jsonl --policy lag --lag-ms 2000 --window 1 --post-key lowlat --dir zh2en --batch 16 --max-gib 17 > eval_outputs/uniss_qwen3_omni_v1/gstar_v8_20261103/runs/on_ali_S2dllag2/infer_r3h1.log 2>&1; echo "EXIT r3h1 $?" ) &
sleep 60
wait
