#!/usr/bin/env bash
# TS-PA causal replay (reports/mist_w3po_v1/tspa_causal_replay_prereg_v1.zh-CN.md). CPU only, detached, parallel.
# DRY_RUN=1 prints the commands only.
set -u
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
PY=/opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python
SP=eval_outputs/mist_w3po_v1/speech
D=eval_outputs/mist_w3po_v1/data
O=eval_outputs/mist_w3po_v1/tspa_precheck
L=logs/mist_v11/tspa
P=${PROCS:-60}
mkdir -p $O $L
run() {
  local n=$1; shift
  if [ -e "$O/$n.json" ]; then echo "skip $n (exists)"; return; fi
  if [ "${DRY_RUN:-0}" = 1 ]; then echo "DRY $n: $*"; return; fi
  PYTHONPATH=. setsid nohup "$@" --out "$O/$n.json" > "$L/$n.log" 2>&1 < /dev/null &
  echo "$n pid $!"
}
M=experiments.mist_w3po_v1.talker.tspa_causal_replay
run replay_zh2en_train16_to_dev5 $PY -m $M --procs $P \
  --fit-cands "$SP/train16_crf_v2p_s3000_finalpost_c2g06/talker_cands/cands_*.jsonl" \
  --fit-win $D/windows_zh2en_c2g06_train16_v1/windows_train_online.jsonl \
  --eval-cands "$SP/plcrf_dev5_cands/talker/cands_*.jsonl" --eval-win $D/windows_zh2en_c2g06_v1_dev5/windows_dev_online.jsonl
for fo in even odd; do
  run replay_en2zh_train16_fold_$fo $PY -m $M --procs $P --lang cmn --units char --fold $fo \
    --fit-cands "$SP/train16_en2zh_v2p_20261112T204539Z_s02000/talker_cands/cands_*.jsonl" \
    --fit-win $D/windows_en2zh_c2g06_train16_v1/windows_train_online.jsonl
done
