#!/usr/bin/env bash
# v11 P0-2 post-processing (reports/mist_w3po_v1/v11_p02_alieval_prereg.zh-CN.md §3), the post-review v10 analysis
# inputs for the AliMeeting Eval rows, read-only on finished speech runs (v11_p01_postproc.sh with the Eval set):
#   lat     lat_units_dump (LATU_MODE perword / blockstart) with exactly alieval_run.sh lat()'s arguments
#           -> $A/lat/<mode>/aev_<row>_s<k>.json(.units.jsonl)                      (CPU)
#   pauses  talker_block_pauses over every row's two talker dirs -> $A/pauses/block_pauses_aev.jsonl   (CPU)
#   quality talker_quality (UTMOS, WavLM-SV) on TTS seed 1 -> $A/quality/<run>/quality.json  (GPU ~2 GiB, card with the
#           most free memory once used + 4 GiB <= 135 GiB; re-read nvidia-smi before each start)
# Rows: b = <B>__t1c, e2 = <B>__rlplem, m = <M>__rlplem, m4 = <M4000>__rlplem. Existing outputs are kept.
#   [DRY_RUN=1] bash experiments/mist_w3po_v1/scripts/alieval_postproc.sh
# Checked with bash -n and DRY_RUN=1 only.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
. experiments/mist_w3po_v1/scripts/joint_lib.sh
A=$JO/alieval_analysis
[ -n "${DRY_RUN:-}" ] && A=/tmp/claude-0/-opt-dlami-nvme-jasonleeeli-projects-UniSS/25116ace-4ea7-4668-a301-0c0d9f82e950/scratchpad/aev_dry
CW=$DD/windows_alieval_c2g06_chunk_v1/windows_dev_online.jsonl
REF=data/diarist_alimeeting_refs_v1/windows_eval_ref_human.jsonl
NB=jaev_b_joint_v8h6k_20261113T073658Z_s06000
NM=jaev_m_joint_v2p_20261113T120415Z_s03500_wbzm1
NM4=jaev_m4_joint_v2p_20261113T120415Z_s04000_wbzm1
ROWS="b:${NB}__t1c e2:${NB}__rlplem m:${NM}__rlplem m4:${NM4}__rlplem"
run() { if [ -n "${DRY_RUN:-}" ]; then echo "DRY: $*" | cut -c1-400; else "$@"; fi; }
latu() {   # <row> <run> <k> <mode>
  local row=$1 n=$2 k=$3 mode=$4 f
  f=$A/lat/$mode/aev_${row}_s$k.json
  [ -e $f ] && return 0
  mkdir -p $A/lat/$mode
  run env LATU_MODE=$mode CUDA_VISIBLE_DEVICES= $PY -m experiments.mist_w3po_v1.evaluation.lat_units_dump \
    experiments.mist_w3po_v1.evaluation.on_refset --ref-win $REF --dr ali \
    experiments.mist_w3po_v1.evaluation.speech_latency_lit --run $DEC/$n --talker $SPD/$n/talker --win $CW --out $f \
    > $f.log 2>&1 || log "lat_units_dump failed: $f"
}
pick_card() {
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | \
    awk -F', ' -v cap=$CAP_MIB '$2 + 4096 <= cap {print $2, $1}' | sort -n | head -1 | awk '{print $2}'
}
qual() {   # <run>
  local n=$1 o c
  o=$A/quality/$n/quality.json
  [ -e $o ] && return 0
  if [ -n "${DRY_RUN:-}" ]; then echo "DRY: talker_quality --talker $SPD/$n/talker --items $SPD/$n/talker_in/items.jsonl --out $o"; return 0; fi
  until c=$(pick_card); [ -n "$c" ]; do sleep 60; done
  mkdir -p $A/quality/$n
  TORCH_HOME=/opt/dlami/nvme/jasonleeeli/cache/torch CUDA_VISIBLE_DEVICES=$c $PY -m experiments.mist_w3po_v1.talker.talker_quality --talker $SPD/$n/talker \
    --items $SPD/$n/talker_in/items.jsonl --out $o > $A/quality/$n/quality.$(date -u +%s).log 2>&1 || log "quality failed: $n (card $c)"
}
dirs=()
for rs in $ROWS; do
  row=${rs%%:*}; n=${rs#*:}
  for k in 1 2; do
    [ $k = 1 ] && r=$n || r=${n}_tts2
    [ -s $SPD/$r/talker/results.jsonl ] || { log "missing talker results: $r"; continue; }
    dirs+=($SPD/$r/talker)
    for mode in perword blockstart; do latu $row $r $k $mode; done
  done
  qual $n
done
P=$A/pauses/block_pauses_aev.jsonl
if [ ! -e $P ] && [ ${#dirs[@]} -gt 0 ]; then
  mkdir -p $A/pauses
  run $PY -m experiments.mist_w3po_v1.evaluation.talker_block_pauses $(printf -- '--talker %s ' "${dirs[@]}") > $P.tmp 2> $P.log \
    && [ -z "${DRY_RUN:-}" ] && command mv -n $P.tmp $P
fi
log "ALIEVAL_POSTPROC_DONE"
