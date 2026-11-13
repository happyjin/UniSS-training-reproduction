#!/usr/bin/env bash
# Joint bidirectional run (b), full DEV evaluation (reports/mist_w3po_v1/joint_bidir_plan_prereg_v1.zh-CN.md §7), with
# the pipelines of the separate systems (decode window 1.0, 8 shards by meeting, --cap135; read-level merge; PL-EM talker
# c1b1s2 flags, two TTS seeds; current-protocol latencies from the orchestrators, chunk-protocol ones here).
#   baseline row (as soon as logs/mist_w3po_v1/joint/JOINT_S1_SELECTED exists, in parallel with stage 2, off the
#     training cards 4-7): the joint v8 (stage-1 merged + identity checkpoint) and the separate v8 models (identity
#     checkpoints) -> text tcpBLEU all / ov / 1spk + text LAAL both protocols; the joint v8's speech (read-level merge,
#     zh2en --sep ' ' / en2zh --sep '', PL-EM, two seeds) recorded only
#   method row (after logs/mist_w3po_v1/joint/JOINT_S2_SELECTED): stage-2 checkpoint on the stage-1 merged base ->
#     text, read-level merge, PL-EM speech (two seeds), latencies both protocols
#   zh2en before en2zh in each row: the en2zh speech waits for (a)'s PL-EM talker (EN2ZH_TALKER_SELECTED in
#   logs/mist_w3po_v1/en2zh/after_select.log; zh2en uses the existing one, talker_plem_20261111T001759Z)
#   gates (gate_2seed, point estimate + per-meeting paired bootstrap; text deltas in the summary): method vs joint
#     baseline (both directions), method vs the separate A'rl + PL-EM (zh2en dev_crf_v2p_s3000_finalpost_s8_c2g06__rlplem;
#     en2zh (a)'s dev_<run>_s<step>__rlplem after EN2ZH_DEV_DONE)
#   -> eval_outputs/mist_w3po_v1/joint/dev_summary_v1.json, logs/mist_w3po_v1/joint/JOINT_DEV_DONE, line JOINT_DEV_DONE
# A merge with pair_lost > 2 goes on only under the pre-authorised pattern (joint_lib.sh read_merge); otherwise, and on
# an orchestrator failure, JOINT_STOP dev <reason>.
# BASE_TALKER=t1c (prereg 修正 3, user 2026-11-13 ~08:00 Prague; default plem = everything above, unchanged): the
#   baseline row's speech is the original single-speaker SI talker T1c@1500, frozen (thinker_t_eval_orch --talker t1c =
#   crf_pair_dev_orch.speech verbatim: --speed-policy fixed --history-ban 8, no c1 / b1 / s2, no CTC scorers), on the
#   plain text decode (no read-level merge, no PL-EM, no wait for (a)'s talker) -> jdev_b{z,e}_<rt>__t1c[_tts2]; the
#   separate v8 rows get the same T1c speech chain (jdev_v8sep_{z,e}__t1c[_tts2]) as soon as their text is done; both
#   seeds, chunk-protocol speech LAAL; the "vs_joint_baseline" gate reads the baseline row's T1c speech; descriptive
#   gate_2seed joint v8 T1c vs separate v8 T1c (vs_separate_v8_t1c, recorded only). The method row is unchanged.
#   [PORT0=19300] [BASE_TALKER=t1c] [DRY_RUN=1] bash experiments/mist_w3po_v1/scripts/joint_dev.sh
# Run detached: setsid nohup bash ... > logs/mist_w3po_v1/joint/dev.log 2>&1 < /dev/null &
# Checked with bash -n and DRY_RUN=1 only.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
. experiments/mist_w3po_v1/scripts/joint_lib.sh
PORT0=${PORT0:-19300}
BASE_TALKER=${BASE_TALKER:-plem}
case $BASE_TALKER in plem) BSFX=__rlplem;; t1c) BSFX=__t1c;; *) log "BASE_TALKER must be plem|t1c"; exit 2;; esac
M1=$JL/JOINT_S1_SELECTED; M2=$JL/JOINT_S2_SELECTED
DONE_M=$JL/JOINT_DEV_DONE
SUM=$JO/dev_summary_v1.json
DV=$JL/dev
APRL=dev_crf_v2p_s3000_finalpost_s8_c2g06__rlplem
IZ=$JO/refs/ident_v8sep_z; IE=$JO/refs/ident_v8sep_e
X=eval_outputs/mist_w3po_v1/talker_crf/talker_align_20261111T001542Z/export   # DRY only: an existing export
flag() { ( set -o noclobber; echo "$(date -u +%FT%TZ) $*" > $DV/FAILED_$(date -u +%s%N) ); log "JOINT_STOP dev: $*"; }
text_dec() {   # <tag> <dir z|e> <name> <base|-> <ckpt> <exclude...>: one detached text decode orchestrator
  local tag=$1 d=$2 n=$3 b=$4 ck=$5; shift 5
  local ex=""; [ $# -gt 0 ] && ex="--exclude $*"
  [ -e $DV/$tag ] && { log "$tag exists: adopted (no relaunch)"; return 0; }
  if [ "$b" = - ]; then
    [ $d = z ] && C="$TORCHZ --run $DV/$tag --name $n --ckpt $ck --talker none --window 1 --cap135 $ex" \
               || C="$EORCH --set dev --run $DV/$tag --name $n --ckpt $ck --talker none --window 1 --cap135 $ex"
  else
    [ $d = z ] && C="$JORCH --dir zh2en --set dev --base $b --run $DV/$tag --name $n --ckpt $ck --talker none --window 1 --cap135 $ex" \
               || C="$JORCH --dir en2zh --set dev --base $b --run $DV/$tag --name $n --ckpt $ck --talker none --window 1 --cap135 $ex"
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
t1c_speech() {   # <tag> <dir z|e> <src decode name> <base|-> <exclude...>: frozen T1c chain, two seeds -> <src>__t1c
  local tag=$1 d=$2 src=$3 b=$4; shift 4
  local ex=""; [ $# -gt 0 ] && ex="--exclude $*"
  local A="--run $DV/$tag --name ${src}__t1c --ckpt - --speech-from $src --talker t1c --window 1 --cap135 $ex"
  if [ "$b" = - ]; then
    [ $d = z ] && C="$TORCHZ $A" || C="$EORCH --set dev $A"
  else
    [ $d = z ] && C="$JORCH --dir zh2en --set dev --base $b $A" || C="$JORCH --dir en2zh --set dev --base $b $A"
  fi
  [ -n "${DRY_RUN:-}" ] && { echo "$tag: $C"; $C --dry | tail -1; return 0; }
  [ -e $DV/$tag ] && { log "$tag exists: adopted"; return 0; }
  setsid nohup $C < /dev/null > $DV/$tag.launch.log 2>&1 &
  log "$tag (${src}__t1c) orchestrator pid $!"
}
row_t1c() {   # <run tag> <base> <ckpt> <port0, unused> <exclude...>: baseline row with the frozen T1c (修正 3)
  local rt=$1 b=$2 ck=$3; shift 4
  local NZ=jdev_bz_$rt NE=jdev_be_$rt
  text_dec bz z $NZ $b $ck "$@"
  [ -z "${DRY_RUN:-}" ] && sleep 90
  text_dec be e $NE $b $ck "$@"
  if [ -n "${DRY_RUN:-}" ]; then t1c_speech bz_t1c z $NZ $b "$@"; t1c_speech be_t1c e $NE $b "$@"; return 0; fi
  wait_done $DV/bz; [ -z "$(orch_failed $DV/bz)" ] || { flag "$NZ decode failed ($(tail -1 $DV/bz/orch.log | cut -c1-200))"; return 1; }
  chunk_text z $NZ
  t1c_speech bz_t1c z $NZ $b "$@" || return 1
  wait_done $DV/be; [ -z "$(orch_failed $DV/be)" ] || { flag "$NE decode failed ($(tail -1 $DV/be/orch.log | cut -c1-200))"; return 1; }
  chunk_text e $NE
  t1c_speech be_t1c e $NE $b "$@" || return 1
  wait_done $DV/bz_t1c; [ -z "$(orch_failed $DV/bz_t1c)" ] || { flag "${NZ}__t1c speech failed"; return 1; }
  chunk_speech z ${NZ}__t1c
  wait_done $DV/be_t1c; [ -z "$(orch_failed $DV/be_t1c)" ] || { flag "${NE}__t1c speech failed"; return 1; }
  chunk_speech e ${NE}__t1c
  log "row b done: $NZ $NE (+ __t1c, two seeds)"
}
v8_t1c() {   # the separate v8 rows (text launched by the caller): chunk text LAAL, then the T1c speech, both seeds
  local t d n
  for t in v8z v8e; do
    [ $t = v8z ] && { d=z; n=jdev_v8sep_z; } || { d=e; n=jdev_v8sep_e; }
    if [ -n "${DRY_RUN:-}" ]; then t1c_speech ${t}_t1c $d $n - 4 5 6 7; continue; fi
    wait_done $DV/$t; [ -z "$(orch_failed $DV/$t)" ] || { flag "$t decode failed"; return 1; }
    chunk_text $d $n
    t1c_speech ${t}_t1c $d $n - 4 5 6 7 || return 1
  done
  [ -n "${DRY_RUN:-}" ] && return 0
  for t in v8z v8e; do
    [ $t = v8z ] && { d=z; n=jdev_v8sep_z; } || { d=e; n=jdev_v8sep_e; }
    wait_done $DV/${t}_t1c; [ -z "$(orch_failed $DV/${t}_t1c)" ] || { flag "${n}__t1c speech failed"; return 1; }
    chunk_speech $d ${n}__t1c
  done
  log "separate v8 rows done: jdev_v8sep_z / _e (+ __t1c, two seeds)"
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
if [ -n "${DRY_RUN:-}" ]; then
  B=eval_outputs/uniss_qwen3_omni_v1/thinker_gstar_o2_20261029b/thinker_merged
  echo "(DRY: O2 + /nonexistent checkpoints stand in for the selections; markers now: $(ls $M1 $M2 2>/dev/null))"
  row b DRY $B /nonexistent/identity $PORT0 4 5 6 7
  text_dec v8z z jdev_v8sep_z - $IZ 4 5 6 7
  text_dec v8e e jdev_v8sep_e - $IE 4 5 6 7
  [ $BASE_TALKER = t1c ] && v8_t1c
  row m DRY $B /nonexistent/ckpt/step_03000 $((PORT0 + 20))
  echo "(a)'s talker now: '$(en2zh_talker)'; (a) stop markers: '$(en2zh_failed)'"
  exit 0
fi
for f in $DONE_M $SUM; do [ -e $f ] && { log "拒绝覆盖已有输出: $f"; exit 1; }; done
mkdir -p $DV
log "dev: waiting for $M1"
until [ -e $M1 ]; do
  grep -q "JOINT_STOP s1" $JL/stage1.log 2>/dev/null && { log "JOINT_STOP dev: stage 1 stopped"; exit 1; }
  sleep 60
done
read -r _ S1STEP _ S1RUN _ MERGED _ IDENT _ <<< "$(cat $M1)"
RT1=${S1RUN}_s$(printf %05d $S1STEP)
log "baseline row: $RT1 ($MERGED + $IDENT), separate v8 identity rows"
( row b $RT1 $MERGED $IDENT $PORT0 4 5 6 7 ) &
PB=$!
sleep 120
text_dec v8z z jdev_v8sep_z - $IZ 4 5 6 7
sleep 90
text_dec v8e e jdev_v8sep_e - $IE 4 5 6 7
PV=""; [ $BASE_TALKER = t1c ] && { ( v8_t1c ) & PV=$!; log "separate v8 rows: T1c speech chain pid $PV"; }
log "method row: waiting for $M2"
until [ -e $M2 ]; do
  grep -q "JOINT_STOP s2" $JL/stage2.log 2>/dev/null && { wait $PB; log "JOINT_STOP dev: stage 2 stopped (baseline row finished)"; exit 1; }
  sleep 60
done
read -r _ S2STEP _ S2RUN _ CK2 _ B2 _ <<< "$(cat $M2)"
RT2=${S2RUN}_s$(printf %05d $S2STEP)
log "method row: $RT2 ($CK2 on $B2)"
( row m $RT2 $B2 $CK2 $((PORT0 + 20)) ) &
PM=$!
if [ $BASE_TALKER = t1c ]; then
  wait $PV || log "separate v8 rows (T1c) failed"
else
for t in v8z v8e; do
  wait_done $DV/$t; [ -z "$(orch_failed $DV/$t)" ] || flag "$t decode failed"
done
chunk_text z jdev_v8sep_z; chunk_text e jdev_v8sep_e
fi
wait $PB; RB=$?
wait $PM; RM=$?
[ $RB = 0 ] && [ $RM = 0 ] && ! ls $DV/FAILED_* > /dev/null 2>&1 || { log "JOINT_STOP dev: a row failed (rc $RB / $RM; $(cat $DV/FAILED_* 2>/dev/null | tr '\n' ' ' | cut -c1-300))"; exit 1; }
# ---------------------------------------------------------------------------------------------- gates
NBZ=jdev_bz_$RT1; NBE=jdev_be_$RT1; NMZ=jdev_mz_$RT2; NME=jdev_me_$RT2
log "waiting for (a)'s EN2ZH_DEV_DONE (its A'rl + PL-EM dev outputs are the en2zh reference)"
until [ -e $E2Z/EN2ZH_DEV_DONE ] && [ -e $E2Z/thinker_selection.json ]; do
  [ -n "$(en2zh_failed)" ] && { log "JOINT_STOP dev: (a) stopped before EN2ZH_DEV_DONE"; exit 1; }
  sleep 60
done
AR=dev_$(cat $E2Z/RUN_ID)_s$($PY -c "import json; print('%05d' % json.load(open('$E2Z/thinker_selection.json'))['chosen'])")__rlplem
gate() {   # <cand> <ref> <tag>
  local O=$TE/$1; mkdir -p $O
  CUDA_VISIBLE_DEVICES= $PY -m experiments.mist_w3po_v1.evaluation.gate_2seed --cand $DEC/$1 $DEC/${1}_tts2 --ref $DEC/$2 $DEC/${2}_tts2 \
    --out $O/gate_$3.json > $O/gate_$3.log 2>&1
  log "gate $1 vs $2 rc=$?"
}
gate ${NMZ}__rlplem ${NBZ}$BSFX vs_joint_baseline
gate ${NME}__rlplem ${NBE}$BSFX vs_joint_baseline
XS=""; XG=""
if [ $BASE_TALKER = t1c ]; then   # descriptive only (修正 3): joint v8 vs separate v8, same T1c speech chain
  gate ${NBZ}__t1c jdev_v8sep_z__t1c vs_separate_v8_t1c
  gate ${NBE}__t1c jdev_v8sep_e__t1c vs_separate_v8_t1c
  XS=', "jdev_v8sep_z__t1c", "jdev_v8sep_e__t1c"'
  XG=' | {f"{c}__vs_separate_v8_t1c": j(TE / c / "gate_vs_separate_v8_t1c.json") for c in ("'${NBZ}__t1c'", "'${NBE}__t1c'")}'
fi
gate ${NMZ}__rlplem $APRL vs_separate_aprl_plem
gate ${NME}__rlplem $AR vs_separate_aprl_plem
# ---------------------------------------------------------------------------------------------- summary (CPU)
$PY - $SUM <<EOF > $DV/summary.log 2>&1 || log "summary failed (see $DV/summary.log)"
import json, sys
from pathlib import Path
DEC, TE = Path("$DEC"), Path("$TE")
def j(p):
    try:
        return json.load(open(p))
    except Exception as e:
        return None
def text(n):
    s = j(DEC / n / "score_text.json") or {}
    l = j(TE / n / f"textlat_{n}.json"); c = j(TE / n / f"textlat_chunk_{n}.json")
    g = lambda d: d["unit"]["text_laal_online"]["all"]["mean"] if d else None
    return {"all": s.get("tcpBLEU_all"), "ov": s.get("tcpBLEU_ov"), "1spk": s.get("tcpBLEU_1spk"), "text_laal": g(l), "text_laal_chunk": g(c)}
def speech(n):
    out = {}
    for nm in (n, n + "_tts2"):
        s = j(DEC / nm / "score_asr.json") or {}
        l = j(TE / n / f"lat_{nm}.json"); c = j(TE / n / f"lat_chunk_{nm}.json")
        g = lambda d: d["unit"]["speech_laal_online"]["all"]["mean"] if d else None
        out[nm] = {"asr_all": s.get("tcpBLEU_all"), "asr_ov": s.get("tcpBLEU_ov"), "speech_laal": g(l), "speech_laal_chunk": g(c)}
    return out
T = {k: text(k) for k in ("$NBZ", "$NBE", "jdev_v8sep_z", "jdev_v8sep_e", "$NMZ", "$NME", "dev_crf_v2p_s3000_finalpost_s8_c2g06", "${AR%__rlplem}")}
S = {k: speech(k) for k in ("${NBZ}$BSFX", "${NBE}$BSFX", "${NMZ}__rlplem", "${NME}__rlplem", "$APRL", "$AR"$XS)}
G = {f"{c}__{t}": j(TE / c / f"gate_{t}.json") for c in ("${NMZ}__rlplem", "${NME}__rlplem") for t in ("vs_joint_baseline", "vs_separate_aprl_plem")}$XG
d = lambda a, b, k="all": None if T[a][k] is None or T[b][k] is None else round(T[a][k] - T[b][k], 2)
res = {"text": T, "speech": S, "gates": G,
       "text_deltas_all": {"joint_v8_vs_separate_v8": {"zh2en": d("$NBZ", "jdev_v8sep_z"), "en2zh": d("$NBE", "jdev_v8sep_e")},
                           "method_vs_joint_v8": {"zh2en": d("$NMZ", "$NBZ"), "en2zh": d("$NME", "$NBE")},
                           "method_vs_separate_aprime": {"zh2en": d("$NMZ", "dev_crf_v2p_s3000_finalpost_s8_c2g06"), "en2zh": d("$NME", "${AR%__rlplem}")}},
       "gate_rule": "text tcpBLEU-all and speech ASR-BLEU (two seeds) >= reference - 0.5 per direction; LAAL reported only"}
with open(sys.argv[1], "x") as fh:
    json.dump(res, fh, indent=1)
print(json.dumps(res["text_deltas_all"]))
EOF
log "summary: $(tail -1 $DV/summary.log | cut -c1-300)"
( set -o noclobber; echo "s1 $RT1 s2 $RT2 utc $(date -u +%FT%TZ)" > $DONE_M ) || log "marker exists"
log "JOINT_DEV_DONE baseline $RT1 method $RT2"
