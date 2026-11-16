#!/usr/bin/env bash
# TS-PA CPU prechecks P1/P2 (reports/mist_w3po_v1/tspa_precheck_prereg_v1.zh-CN.md). CPU only, detached, parallel.
# DRY_RUN=1 prints the commands only.
set -u
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
PY=/opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python
SP=eval_outputs/mist_w3po_v1/speech
D=eval_outputs/mist_w3po_v1/data
O=eval_outputs/mist_w3po_v1/tspa_precheck
L=logs/mist_v11/tspa
mkdir -p $O $L
run() {  # name, command...
  local n=$1; shift
  if [ -e "$O/$n.json" ]; then echo "skip $n (exists)"; return; fi
  if [ "${DRY_RUN:-0}" = 1 ]; then echo "DRY $n: $*"; return; fi
  PYTHONPATH=. setsid nohup "$@" --out "$O/$n.json" > "$L/$n.log" 2>&1 < /dev/null &
  echo "$n pid $!"
}
run p1a_zh2en_dev5 $PY -m experiments.mist_w3po_v1.talker.tspa_precheck_p1 --procs 24 \
  --cands "$SP/plcrf_dev5_cands/talker/cands_*.jsonl" --win $D/windows_zh2en_c2g06_v1_dev5/windows_dev_online.jsonl
run p1b_en2zh_train16 $PY -m experiments.mist_w3po_v1.talker.tspa_precheck_p1 --procs 56 --lang cmn --units char \
  --cands "$SP/train16_en2zh_v2p_20261112T204539Z_s02000/talker_cands/cands_*.jsonl" \
  --win $D/windows_en2zh_c2g06_train16_v1/windows_train_online.jsonl
run p1c_zh2en_train16 $PY -m experiments.mist_w3po_v1.talker.tspa_precheck_p1 --procs 56 \
  --cands "$SP/train16_crf_v2p_s3000_finalpost_c2g06/talker_cands/cands_*.jsonl" \
  --win $D/windows_zh2en_c2g06_train16_v1/windows_train_online.jsonl
for s in "" _tts2; do
  run p2_zh2en_dev$s $PY -m experiments.mist_w3po_v1.talker.tspa_precheck_p2 --procs 10 \
    --run $SP/jdev_bz_joint_v8h6k_20261113T073658Z_s06000__rlplem$s --win $D/windows_zh2en_c2g06_v1/windows_dev_online.jsonl
  run p2_en2zh_dev$s $PY -m experiments.mist_w3po_v1.talker.tspa_precheck_p2 --procs 10 --lang cmn --units char \
    --run $SP/jdev_be_joint_v8h6k_20261113T073658Z_s06000__rlplem$s --win $D/windows_en2zh_c2g06_v1/windows_dev_online.jsonl
done
