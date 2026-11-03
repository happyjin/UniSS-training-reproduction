#!/usr/bin/env bash
# One decoder spread over all 8 cards (--max-gib 11: 88 GiB budget for the 66 GB Thinker, no CPU offload), outside the
# queue.   bash run_spread.sh <job tag> <thinker> <windows> <out hyp> <log> <policy> <dir>
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
export MIST_V7_JOB=mist8r_$1 MIST_V7_KIND=decode
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTHONPATH=.:third_party/Megatron-LM /opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python -m experiments.uniss_qwen3_omni_v1.evaluation.gstar_si_stream_online_v6s --thinker $2 --windows $3 --out $4 --shard 0 --num-shards 1 --policy $6 --window 1 --post-key lowlat --dir $7 --batch 12 --max-gib 11 > $5 2>&1 && echo "SPREAD_DONE $1"
