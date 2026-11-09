#!/usr/bin/env bash
# TurnCRF CPU stage (reports/mist_w3po_v1/turn_crf_cpu_stage.zh-CN.md): train the semi-CRF (52 worker processes) and
# the linear-chain CRF (36) concurrently, detached (setsid nohup), one pre-registered setting each. CPU only.
# Refuses an existing output dir. Each run writes <out>/<kind>/{params.pt,train_log.jsonl,config.json,DONE|FAILED},
# its log <out>/<kind>.log and exit code <out>/<kind>.rc.
set -euo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
OUT=${OUT:-eval_outputs/mist_w3po_v1/crf/turn_v1}
if [ -e "$OUT" ]; then echo "拒绝覆盖已有输出: $OUT" >&2; exit 1; fi
mkdir -p "$OUT"
PY=/opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python
export PYTHONPATH=. TMPDIR=/opt/dlami/nvme/jasonleeeli/work/mist_w3po_v1/tmp OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=""
for spec in "semi 52" "chain 36"; do
  read -r kind nw <<< "$spec"
  setsid nohup bash -c "$PY -m experiments.mist_w3po_v1.crf.train_turn_crf --kind $kind --out $OUT/$kind --workers $nw; echo \$? > $OUT/$kind.rc" \
    > "$OUT/$kind.log" 2>&1 < /dev/null &
  disown
done
echo "launched semi (52 workers) and chain (36 workers) into $OUT"
