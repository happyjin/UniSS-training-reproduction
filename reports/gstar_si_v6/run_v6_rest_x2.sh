#!/usr/bin/env bash
# Paper v6: when the dec3x2 job (MIST-Simul Zh->En on whole meetings: Zh->En AMI and AliMeeting subsets) has ended,
# decode the windows of its shards that the memory watchdog stopped (make_rest.py: batch 4, 2 workers) via a gpuq7 queue.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
exec >> logs/multispk/v6/rest.log 2>&1
V6=eval_outputs/uniss_qwen3_omni_v1/gstar_v6_20261031; ON=data/online_v6_20261031
until ! ps -eo args | grep -q "[g]star_multi_v6 --thinker .*jobs/dec3x2"; do sleep 60; done
echo "[$(date -u +%F' '%T) UTC] dec3x2 ended"
jobs=()
for st in zh2en ali; do
  out=$(python3 reports/gstar_si_v6/make_rest.py $V6/on_${st}_S2dl $ON/windows_test_online_${st}_sub.jsonl S2DLZ \
        gstar_si_stream_online_v6 '["--policy", "wait", "--window", "1", "--post-key", "lowlat", "--dir", "zh2en"]' --batch 4 --workers 2)
  echo "$st: $out"
  rf=$(echo "$out" | awk '{print $1}')
  [ -f "$rf" ] && jobs+=("{\"id\": \"rest_on_${st}_S2dl\", \"gpus\": 1, \"vllm\": false, \"deps\": [], \"mem\": 86000, \"cmd\": \"CUDA_VISIBLE_DEVICES=\$GPU0 PYTHONPATH=.:third_party/Megatron-LM /opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python -m experiments.uniss_qwen3_omni_v1.evaluation.gstar_multi_v6 --thinker eval_outputs/uniss_qwen3_omni_v1/thinker_gstar_s2dl_zh2en_3k_20261030/thinker_merged $rf\"}")
done
[ ${#jobs[@]} -eq 0 ] && { echo "nothing left"; exit 0; }
J=reports/gstar_si_v6/jobs/rest_x2.json
( IFS=,; echo "[${jobs[*]}]" ) > $J.part && python3 -c "import json,sys; json.load(open(sys.argv[1]))" $J.part && command mv -n $J.part $J
CARDS=2,3,4,5,6,7 bash reports/gstar_si_v6/run_v6_queue.sh $J logs/multispk/v6/q_rest_x2
