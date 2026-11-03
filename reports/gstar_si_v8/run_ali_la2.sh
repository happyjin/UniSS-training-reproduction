#!/usr/bin/env bash
# E-pol AliMeeting: LA2 on the prefix-trained Zh->En model over 20 whole meetings, spread over all 8 cards (~11 GB each,
# v7 p8 layout) outside the queue, whose declared-memory accounting has no room left beside the C1 training.
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
export MIST_V7_JOB=mist8r_ali_la2 MIST_V7_KIND=decode
D=eval_outputs/uniss_qwen3_omni_v1/gstar_v8_20261103/runs/on_ali_S2dlLA2
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTHONPATH=.:third_party/Megatron-LM /opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python -m experiments.uniss_qwen3_omni_v1.evaluation.gstar_si_stream_online_v6s --thinker eval_outputs/uniss_qwen3_omni_v1/thinker_gstar_s2dl_zh2en_3k_20261030/thinker_merged --windows data/online_v6_20261031/windows_test_online_ali.jsonl --out $D/hyp_s8.jsonl --shard 0 --num-shards 1 --policy la2 --window 1 --post-key lowlat --dir zh2en --batch 12 --max-gib 9 > $D/infer_s8.log 2>&1 && echo ALI_LA2_DONE
