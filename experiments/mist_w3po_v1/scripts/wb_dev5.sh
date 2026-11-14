#!/usr/bin/env bash
# Plan A dev5 calibration (user 2026-11-14 ~10:20 Prague: "先A后B，8卡并行叠加，dev5速验叠加在8卡"): the method
# (joint v2' stage 2 step 3500 on the joint v8 s6000 merged base) decoded on dev5 with the inference-time write bias
# (joint_orch_wb.py -> crf/decode_crf_ckpt_waitbias --wait-bias b) for b in BIASES, plus the strict baseline (identity
# checkpoint of the same base, joint_orch.py, no bias) on the same dev5 windows, both directions: text only
# (--talker none), joint_proxy.sh's dev5 flags (finalpost, wait policy, 1.0 s reads, 4 window shards, --cap135), all 8
# cards (no --exclude). Orchestrators start 60 s apart; after each finishes, the chunk-protocol text LAAL (as
# joint_proxy.sh). Summary line WB_DEV5_DONE; results in logs/mist_w3po_v1/joint/wb/summary.tsv.
#   [BIASES="-1 -2 -3 -5"] [DRY_RUN=1] bash experiments/mist_w3po_v1/scripts/wb_dev5.sh
# Checked with bash -n and DRY_RUN=1 only.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
. experiments/mist_w3po_v1/scripts/joint_lib.sh
BIASES=${BIASES:-"-1 -2 -3 -5"}
BASE=$JO/joint_v8h6k_20261113T073658Z/merged_s06000/thinker_merged
IDN=$JO/joint_v8h6k_20261113T073658Z/merged_s06000/identity
CK=$CRF/train/joint_v2p_20261113T120415Z/ckpt/step_03500
WB=$JL/wb
WORCH="$PY $SC/joint_orch_wb.py"
COMMON="--talker none --shards 4 --shard-by window --window 1 --cap135"
tag() { echo "$1" | sed 's/^-/m/; s/\./p/'; }
rows=()   # "<dir z|e> <name> <orchestrator command>"
for x in z e; do
  [ $x = z ] && d=zh2en || d=en2zh
  n=wb${x}_dev5_base_s6000
  rows+=("$x $n $JORCH --dir $d --set dev5 --base $BASE --run $WB/$n --name $n --ckpt $IDN $COMMON")
  for b in $BIASES; do
    n=wb${x}_dev5_m3500_b$(tag $b)
    rows+=("$x $n $WORCH --wait-bias $b --dir $d --set dev5 --base $BASE --run $WB/$n --name $n --ckpt $CK $COMMON")
  done
done
for f in $BASE/sd_proj.pt $IDN $CK/DONE; do [ -e $f ] || { log "missing $f"; exit 1; }; done
if [ -n "${DRY_RUN:-}" ]; then
  for r in "${rows[@]}"; do set -- $r; x=$1 n=$2; shift 2; echo "$n: $*"; "$@" --dry 2>&1 | tail -1; done
  exit 0
fi
mkdir -p $WB
for r in "${rows[@]}"; do
  set -- $r; x=$1 n=$2; shift 2
  [ -e $WB/$n ] && { log "$WB/$n exists: not relaunched"; continue; }
  setsid nohup "$@" < /dev/null > $WB/$n.launch.log 2>&1 &
  log "$n orchestrator pid $!"
  sleep 60
done
printf 'name\ttext_all\ttext_ov\ttext_1spk\ttext_laal_chunk\n' > $WB/summary.tsv.tmp
for r in "${rows[@]}"; do
  set -- $r; x=$1 n=$2
  wait_done $WB/$n
  [ -z "$(orch_failed $WB/$n)" ] || log "$n: orchestrator ended with failures ($(tail -1 $WB/$n/orch.log | cut -c1-200))"
  mkdir -p $TE/$n
  if [ $x = z ]; then
    CUDA_VISIBLE_DEVICES= $PY -m experiments.mist_w3po_v1.evaluation.speech_latency_lit --run $DEC/$n --text-only --win $WZ_DEV5C \
      --out $TE/$n/textlat_chunk_$n.json > $TE/$n/textlat_chunk.log 2>&1 || log "chunk text LAAL failed: $n"
  else
    CUDA_VISIBLE_DEVICES= $PY -m experiments.mist_w3po_v1.evaluation.en2zh_lat --ref-win $RE_DEV --work $TE/$n/textlat_chunk_$n.json.work \
      --run $DEC/$n --text-only --win $WE_DEV5C --out $TE/$n/textlat_chunk_$n.json > $TE/$n/textlat_chunk.log 2>&1 || log "chunk text LAAL failed: $n"
  fi
  $PY - $DEC/$n/score_text.json $TE/$n/textlat_chunk_$n.json $n >> $WB/summary.tsv.tmp <<'EOF'
import json, sys
def j(p):
    try:
        return json.load(open(p))
    except Exception:
        return {}
t, c = j(sys.argv[1]), j(sys.argv[2])
l = ((c.get("unit") or {}).get("text_laal_online") or {}).get("all", {}).get("mean")
print("\t".join(str(v) for v in (sys.argv[3], t.get("tcpBLEU_all"), t.get("tcpBLEU_ov"), t.get("tcpBLEU_1spk"), l)))
EOF
  log "$n: $(tail -1 $WB/summary.tsv.tmp)"
done
command mv -f $WB/summary.tsv.tmp $WB/summary.tsv
log "WB_DEV5_DONE"
