#!/usr/bin/env bash
# Plan A full-dev confirmation (user 2026-11-14 ~10:20 Prague: "先A后B"): the method row of joint_dev.sh (joint v2'
# stage 2 step 3500 on the joint v8 s6000 merged base -> text decode -> read-level merge -> PL-EM speech, two seeds ->
# chunk-protocol text / speech LAAL) with the inference-time write bias of the dev5 calibration (wb_dev5.sh): the text
# decode runs through joint_orch_wb.py --wait-bias <BZ> (zh2en) / <BE> (en2zh); every other step is joint_dev.sh's (its
# text_dec / speech / chunk_text / chunk_speech / row functions, copied below with text_dec's orchestrator for the
# method row swapped). Run dir logs/mist_w3po_v1/joint/wb/dev, names jdev_mz_<rt> / jdev_me_<rt> with rt =
# joint_v2p_20261113T120415Z_s03500_wb<tag>. All 8 cards (no --exclude), --cap135. Ends with WB_DEV_DONE and writes the
# marker logs/mist_w3po_v1/joint/wb/A_DONE (plan B's driver starts after it).
#   BZ=<bias zh2en> BE=<bias en2zh> [PORT0=19600] [DRY_RUN=1] bash experiments/mist_w3po_v1/scripts/wb_dev.sh
# Checked with bash -n and DRY_RUN=1 only.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
. experiments/mist_w3po_v1/scripts/joint_lib.sh
BZ=${BZ:?}; BE=${BE:?}
PORT0=${PORT0:-19600}
BASE_TALKER=plem; BSFX=__rlplem
DV=$JL/wb/dev
X=eval_outputs/mist_w3po_v1/talker_crf/talker_align_20261111T001542Z/export   # DRY only: an existing export
WORCH="$PY $SC/joint_orch_wb.py"
tag() { echo "$1" | sed 's/^-/m/; s/\./p/'; }
flag() { ( set -o noclobber; echo "$(date -u +%FT%TZ) $*" > $DV/FAILED_$(date -u +%s%N) ); log "WB_STOP dev: $*"; }
text_dec() {   # <tag> <dir z|e> <name> <base|-> <ckpt> <exclude...>: one detached text decode orchestrator
  local tag=$1 d=$2 n=$3 b=$4 ck=$5; shift 5
  local ex=""; [ $# -gt 0 ] && ex="--exclude $*"
  [ -e $DV/$tag ] && { log "$tag exists: adopted (no relaunch)"; return 0; }
  if [ "$b" = - ]; then
    [ $d = z ] && C="$TORCHZ --run $DV/$tag --name $n --ckpt $ck --talker none --window 1 --cap135 $ex" \
               || C="$EORCH --set dev --run $DV/$tag --name $n --ckpt $ck --talker none --window 1 --cap135 $ex"
  else
    [ $d = z ] && C="$WORCH --wait-bias $BZ --dir zh2en --set dev --base $b --run $DV/$tag --name $n --ckpt $ck --talker none --window 1 --cap135 $ex" \
               || C="$WORCH --wait-bias $BE --dir en2zh --set dev --base $b --run $DV/$tag --name $n --ckpt $ck --talker none --window 1 --cap135 $ex"
  fi
  [ -n "${DRY_RUN:-}" ] && { echo "$tag: $C"; $C --dry | tail -1; return 0; }
  setsid nohup $C < /dev/null > $DV/$tag.launch.log 2>&1 &
  log "$tag ($n) orchestrator pid $!"
}
speech() {   # <tag> <dir z|e> <rl name> <base> <port0> <exclude...>: PL-EM speech chain, two seeds, on <rl name>
  local tag=$1 d=$2 rl=$3 b=$4 p=$5; shift 5
  local ex=""; [ $# -gt 0 ] && ex="--exclude $*"
  local n=${rl%_rl}__rlplem pe=""
  if [ $d = e ]; then
    local tx; tx=$(en2zh_talker); [ -n "${DRY_RUN:-}" ] && tx=${tx:-$X}
    [ -n "$tx" ] && [ -e $tx/uniss/model.safetensors ] || { flag "$tag: no En->Zh PL-EM export ('$tx')"; return 1; }
    pe="--plem-export $tx"
  fi
  local dd; [ $d = z ] && dd=zh2en || dd=en2zh
  C="$JORCH --dir $dd --set dev --base $b --run $DV/$tag --name $n --ckpt - --speech-from $rl --talker plem $pe --window 1 --cap135 --scorer-port0 $p $ex"
  [ -n "${DRY_RUN:-}" ] && { echo "$tag: $C"; $C --dry | tail -1; return 0; }
  [ -e $DV/$tag ] && { log "$tag exists: adopted"; return 0; }
  setsid nohup $C < /dev/null > $DV/$tag.launch.log 2>&1 &
  log "$tag ($n) orchestrator pid $!"
}
chunk_text() {   # <dir z|e> <name>
  local O=$TE/$2; mkdir -p $O
  if [ $1 = z ]; then
    CUDA_VISIBLE_DEVICES= $PY -m experiments.mist_w3po_v1.evaluation.speech_latency_lit --run $DEC/$2 --text-only --win $WZ_DEVC \
      --out $O/textlat_chunk_$2.json > $O/textlat_chunk.log 2>&1 || log "chunk text LAAL failed: $2"
  else
    CUDA_VISIBLE_DEVICES= $PY -m experiments.mist_w3po_v1.evaluation.en2zh_lat --ref-win $RE_DEV --work $O/textlat_chunk_$2.json.work \
      --run $DEC/$2 --text-only --win $WE_DEVC --out $O/textlat_chunk_$2.json > $O/textlat_chunk.log 2>&1 || log "chunk text LAAL failed: $2"
  fi
}
chunk_speech() {   # <dir z|e> <speech name (seed 1)>: both seeds
  local d=$1 b=$2 nm O=$TE/$2
  mkdir -p $O
  for nm in $b ${b}_tts2; do
    if [ $d = z ]; then
      CUDA_VISIBLE_DEVICES= $PY -m experiments.mist_w3po_v1.evaluation.speech_latency_lit --run $DEC/$nm --talker $SPD/$nm/talker \
        --win $WZ_DEVC --out $O/lat_chunk_$nm.json > $O/lat_chunk_$nm.log 2>&1 || log "chunk LAAL failed: $nm"
    else
      CUDA_VISIBLE_DEVICES= $PY -m experiments.mist_w3po_v1.evaluation.en2zh_lat --ref-win $RE_DEV --work $O/lat_chunk_$nm.json.work \
        --run $DEC/$nm --talker $SPD/$nm/talker --win $WE_DEVC --out $O/lat_chunk_$nm.json > $O/lat_chunk_$nm.log 2>&1 || log "chunk LAAL failed: $nm"
    fi
  done
}
row() {   # <row b|m> <run tag> <base> <ckpt> <port0> <exclude...>: zh2en first, then en2zh
  [ $1 = b ] && [ $BASE_TALKER = t1c ] && { shift; row_t1c "$@"; return; }
  local r=$1 rt=$2 b=$3 ck=$4 p=$5; shift 5
  local NZ=jdev_${r}z_$rt NE=jdev_${r}e_$rt
  text_dec ${r}z z $NZ $b $ck "$@"
  [ -z "${DRY_RUN:-}" ] && sleep 90
  text_dec ${r}e e $NE $b $ck "$@"
  if [ -n "${DRY_RUN:-}" ]; then speech ${r}z_sp z ${NZ}_rl $b $p "$@"; speech ${r}e_sp e ${NE}_rl $b $((p + 10)) "$@"; return 0; fi
  wait_done $DV/${r}z; [ -z "$(orch_failed $DV/${r}z)" ] || { flag "$NZ decode failed ($(tail -1 $DV/${r}z/orch.log | cut -c1-200))"; return 1; }
  chunk_text z $NZ
  read_merge $NZ ' ' $WZ_DEV $DV/merge_$NZ.log || { flag "merge $NZ ($(tail -1 $DV/merge_$NZ.log | cut -c1-200))"; return 1; }
  speech ${r}z_sp z ${NZ}_rl $b $p "$@" || return 1
  wait_done $DV/${r}e; [ -z "$(orch_failed $DV/${r}e)" ] || { flag "$NE decode failed ($(tail -1 $DV/${r}e/orch.log | cut -c1-200))"; return 1; }
  chunk_text e $NE
  read_merge $NE '' $WE_DEV $DV/merge_$NE.log || { flag "merge $NE ($(tail -1 $DV/merge_$NE.log | cut -c1-200))"; return 1; }
  log "row $r: waiting for (a)'s PL-EM talker"
  until [ -n "$(en2zh_talker)" ]; do
    [ -n "$(en2zh_failed)" ] && { flag "(a) stopped before its PL-EM talker ($(en2zh_failed | cut -c1-200))"; return 1; }
    sleep 60
  done
  speech ${r}e_sp e ${NE}_rl $b $((p + 10)) "$@" || return 1
  wait_done $DV/${r}z_sp; [ -z "$(orch_failed $DV/${r}z_sp)" ] || { flag "${NZ}__rlplem speech failed"; return 1; }
  chunk_speech z ${NZ}__rlplem
  wait_done $DV/${r}e_sp; [ -z "$(orch_failed $DV/${r}e_sp)" ] || { flag "${NE}__rlplem speech failed"; return 1; }
  chunk_speech e ${NE}__rlplem
  log "row $r done: $NZ $NE (+ __rlplem, two seeds)"
}
B2=$JO/joint_v8h6k_20261113T073658Z/merged_s06000/thinker_merged
CK2=$CRF/train/joint_v2p_20261113T120415Z/ckpt/step_03500
RT=joint_v2p_20261113T120415Z_s03500_wbz$(tag $BZ)e$(tag $BE)
if [ -n "${DRY_RUN:-}" ]; then row m $RT $B2 $CK2 $PORT0; exit 0; fi
mkdir -p $DV
[ -e $JL/wb/A_DONE ] && { log "$JL/wb/A_DONE exists"; exit 1; }
log "plan A full dev: bias zh2en $BZ, en2zh $BE, run tag $RT"
row m $RT $B2 $CK2 $PORT0 || { log "WB_STOP dev: row failed"; exit 1; }
( set -o noclobber; echo "$(date -u +%FT%TZ) $RT jdev_mz_$RT jdev_me_$RT" > $JL/wb/A_DONE )
log "WB_DEV_DONE $RT"
