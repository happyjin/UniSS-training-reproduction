#!/usr/bin/env bash
# v11 P0-3 (user 2026-11-15 ~08:10 Prague: "可以做但要看效果"; mock ACL review M2/P0): the strict deadline baseline B
# (identity checkpoint of the joint v8 s6000 merged base) decoded on dev5 with the same inference-time write bias as
# the method (joint_orch_wb.py -> decode_crf_ckpt_waitbias --wait-bias b), b in BIASES, both directions, text only.
# Together with the existing method rows (wb_dev5.sh: b = 0, -1, -2, -3, -5) this gives the BLEU vs text-LAAL curve of
# B and of write alignment on the same dev5 windows (matched-latency comparison). Dev only: nothing on test.
# Copy of wb_dev5.sh with the rows swapped; everything else unchanged. Summary WB_BASE_DEV5_DONE in $WB/summary_base.tsv.
#   [BIASES="1 0.5 -0.5 -1 -2"] [DRY_RUN=1] bash experiments/mist_w3po_v1/scripts/wb_dev5_base.sh
# Checked with bash -n and DRY_RUN=1 only.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
. experiments/mist_w3po_v1/scripts/joint_lib.sh
BIASES=${BIASES:-"1 0.5 -0.5 -1 -2"}
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
  for b in $BIASES; do
    n=wb${x}_dev5_base_b$(tag $b)
    rows+=("$x $n $WORCH --wait-bias $b --dir $d --set dev5 --base $BASE --run $WB/$n --name $n --ckpt $IDN $COMMON")
  done
done
for f in $BASE/sd_proj.pt $IDN; do [ -e $f ] || { log "missing $f"; exit 1; }; done
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
printf 'name\ttext_all\ttext_ov\ttext_1spk\ttext_laal_chunk\n' > $WB/summary_base.tsv.tmp
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
  $PY - $DEC/$n/score_text.json $TE/$n/textlat_chunk_$n.json $n >> $WB/summary_base.tsv.tmp <<'EOF'
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
  log "$n: $(tail -1 $WB/summary_base.tsv.tmp)"
done
command mv -f $WB/summary_base.tsv.tmp $WB/summary_base.tsv
log "WB_BASE_DEV5_DONE"
