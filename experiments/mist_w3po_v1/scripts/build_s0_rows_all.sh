#!/usr/bin/env bash
# The six stage-0 row sets (S0-all and its five leave-one-out ablations), built in parallel (CPU).
#   bash build_s0_rows_all.sh   (refuses existing outputs through the builder)
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
J=/opt/dlami/nvme/jasonleeeli
export TMPDIR=$J/work/mist_w3po_v1/tmp PYTHONPATH=.:third_party/Megatron-LM
PY=$J/conda_envs/qwen3omni/bin/python
D=eval_outputs/mist_w3po_v1/data
DEC=eval_outputs/mist_w3po_v1/decode
DAG="--dagger $(ls $DEC/dagger_v8_online14/hyp_s*.jsonl $DEC/dagger_v8_ref1500/hyp_s*.jsonl | tr '\n' ' ') --dagger-windows $D/windows_zh2en_v1/windows_train_online.jsonl $D/windows_zh2en_v1/windows_train_diar.jsonl $D/dagger_v1/windows_ref1500.jsonl"
declare -A V
V[all]="--jitter $DAG --diar --context --winmap"
V[no1]="$DAG --diar --context --winmap"
V[no2]="--jitter --diar --context --winmap"
V[no3]="--jitter $DAG --context --winmap"
V[no4]="--jitter $DAG --diar --winmap"
V[no5]="--jitter $DAG --diar --context"
for k in "${!V[@]}"; do
  $PY -m experiments.mist_w3po_v1.data.build_s0_rows_zh2en ${V[$k]} --out $D/rows_s0_$k > logs/mist_w3po_v1/rows_s0_$k.log 2>&1 &
done
wait
grep -h -A3 '"read_rows"' logs/mist_w3po_v1/rows_s0_*.log
echo ROWS_DONE
