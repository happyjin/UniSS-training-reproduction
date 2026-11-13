#!/usr/bin/env bash
# Joint bidirectional run (b), TEST, run once (reports/mist_w3po_v1/joint_bidir_plan_prereg_v1.zh-CN.md §7): after
# logs/mist_w3po_v1/joint/JOINT_DEV_DONE, with everything fixed on dev (stage-1 / stage-2 selections, (a)'s PL-EM talker
# for en2zh, the existing zh2en PL-EM talker). Nothing is chosen on test. Sets: AMI zh->en, AliMeeting zh->en
# (final_test_orch's windows / references), AMI En->Zh (en2zh_orch's). Rows: joint baseline (stage-1 merged + identity)
# and method (stage-2 checkpoint on the stage-1 merged base). Per (row, set): text decode (8 shards by meeting, window
# 1.0, --cap135) -> read-level merge (zh2en ' ', en2zh ''; joint_lib.sh read_merge rule) -> PL-EM speech, two seeds ->
# chunk-protocol latencies (text and speech; zh2en via on_refset, en2zh via en2zh_lat) -> gate_2seed method vs baseline
# and method vs the separate A'rl + PL-EM test outputs (ft_ami / ft_ali_aprime_s3000__rlplem; (a)'s
# ft_en2zh_<run>_s<step>__rlplem after EN2ZH_TEST_DONE) -> eval_outputs/mist_w3po_v1/joint/test_summary_v1.json,
# logs/mist_w3po_v1/joint/JOINT_TEST_DONE, line JOINT_TEST_DONE. No COMET / MetricX.
# BASE_TALKER=t1c (prereg 修正 3; default plem = everything above, unchanged): the baseline row's speech is the original
#   single-speaker SI talker T1c@1500, frozen (thinker_t_eval_orch --talker t1c: --speed-policy fixed --history-ban 8,
#   no c1 / b1 / s2, no CTC scorers) on the plain text decode (no read-level merge, no PL-EM) ->
#   jtest_b<set>_<rt>__t1c[_tts2], both seeds; the "vs_joint_baseline" gate reads it. The method row is unchanged.
# TEST_TRIGGER=s2 (user 2026-11-13 ~20:10 Prague: baseline and method dev + test results by the morning; default empty
#   = wait for JOINT_DEV_DONE, unchanged): start as soon as JOINT_S2_SELECTED exists. Nothing on test is chosen either
#   way: the stage-1 / stage-2 checkpoints are fixed by their rules and the En->Zh PL-EM talker by (a)'s dev5 rule; the
#   method En->Zh chain decodes and merges first, then waits for that talker (EN2ZH_TALKER_SELECTED) before its speech.
# TEST_RESUME=1 (2026-11-13 ~21:35 Prague, after the user's stage-2 override 3500; default empty = unchanged): an
#   existing test dir is allowed and every orchestrator whose run dir already exists is adopted (not relaunched; the
#   later steps of its chain are redone, idempotent)
#   [PORT0=19400] [BASE_TALKER=t1c] [TEST_TRIGGER=s2] [TEST_RESUME=1] [DRY_RUN=1] bash experiments/mist_w3po_v1/scripts/joint_test.sh
# Run detached: setsid nohup bash ... > logs/mist_w3po_v1/joint/test.log 2>&1 < /dev/null &
# Checked with bash -n and DRY_RUN=1 only.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
. experiments/mist_w3po_v1/scripts/joint_lib.sh
PORT0=${PORT0:-19400}
BASE_TALKER=${BASE_TALKER:-plem}
case $BASE_TALKER in plem) BSFX=__rlplem;; t1c) BSFX=__t1c;; *) log "BASE_TALKER must be plem|t1c"; exit 2;; esac
M1=$JL/JOINT_S1_SELECTED; M2=$JL/JOINT_S2_SELECTED; DEVD=$JL/JOINT_DEV_DONE
DONE_M=$JL/JOINT_TEST_DONE
SUM=$JO/test_summary_v1.json
TV=$JL/test
X=eval_outputs/mist_w3po_v1/talker_crf/talker_align_20261111T001542Z/export   # DRY only
adopt() { [ -n "${TEST_RESUME:-}" ] && [ -e $1 ] && { log "$1 exists: adopted (TEST_RESUME)"; return 0; }; return 1; }
flag() { ( set -o noclobber; echo "$(date -u +%FT%TZ) $*" > $TV/FAILED_$(date -u +%s%N) ); log "JOINT_STOP test: $*"; }
setvars() {   # <set ami|ali|en> -> D (dir flag) K (joint_orch --set) WIN CW REF DR SEP
  case $1 in
    ami) D=zh2en; K=test; WIN=$WT_AMI; CW=$WT_AMIC; REF=$RT_AMI; DR=zh2en; SEP=' ';;
    ali) D=zh2en; K=ali;  WIN=$WT_ALI; CW=$WT_ALIC; REF=$RT_ALI; DR=ali; SEP=' ';;
    en)  D=en2zh; K=test; WIN=$WT_EN;  CW=$WT_ENC;  REF=$RE_TEST; DR=""; SEP='';;
  esac
}
lat() {   # <set> <run name> [<talker dir>] -> chunk-protocol LAAL json (text-only without a talker)
  local s=$1 n=$2 tk=${3:-} O f
  setvars $s
  O=$TE/${n%_tts2}; mkdir -p $O
  if [ -z "$tk" ]; then f=$O/textlat_chunk_$n.json; A="--text-only"; else f=$O/lat_chunk_$n.json; A="--talker $tk"; fi
  if [ $D = zh2en ]; then
    CUDA_VISIBLE_DEVICES= $PY -m experiments.mist_w3po_v1.evaluation.on_refset --ref-win $REF --dr $DR \
      experiments.mist_w3po_v1.evaluation.speech_latency_lit --run $DEC/$n $A --win $CW --out $f > $f.log 2>&1 || log "chunk LAAL failed: $n"
  else
    CUDA_VISIBLE_DEVICES= $PY -m experiments.mist_w3po_v1.evaluation.en2zh_lat --ref-win $REF --work $f.work --run $DEC/$n $A \
      --win $CW --out $f > $f.log 2>&1 || log "chunk LAAL failed: $n"
  fi
}
chain() {   # <row b|m> <set> <run tag> <base> <ckpt> <port0>
  local r=$1 s=$2 rt=$3 b=$4 ck=$5 p=$6 n tx pe=""
  setvars $s
  n=jtest_${r}${s}_$rt
  local C1="$JORCH --dir $D --set $K --base $b --run $TV/${r}$s --name $n --ckpt $ck --talker none --window 1 --cap135"
  if [ $r = b ] && [ $BASE_TALKER = t1c ]; then   # 修正 3: frozen T1c on the plain text, no merge, no PL-EM
    local C2="$JORCH --dir $D --set $K --base $b --run $TV/${r}${s}_sp --name ${n}__t1c --ckpt - --speech-from ${n} --talker t1c --window 1 --cap135"
    if [ -n "${DRY_RUN:-}" ]; then echo "${r}$s: $C1"; $C1 --dry | tail -1; echo "${r}${s}_sp: $C2"; $C2 --dry | tail -1; return 0; fi
    adopt $TV/${r}$s || setsid nohup $C1 < /dev/null > $TV/${r}$s.launch.log 2>&1 &
    log "$n decode orchestrator pid $!"
    wait_done $TV/${r}$s; [ -z "$(orch_failed $TV/${r}$s)" ] || { flag "$n decode failed ($(tail -1 $TV/${r}$s/orch.log | cut -c1-200))"; return 1; }
    lat $s $n
    adopt $TV/${r}${s}_sp || setsid nohup $C2 < /dev/null > $TV/${r}${s}_sp.launch.log 2>&1 &
    log "${n}__t1c speech orchestrator pid $!"
    wait_done $TV/${r}${s}_sp; [ -z "$(orch_failed $TV/${r}${s}_sp)" ] || { flag "${n}__t1c speech failed"; return 1; }
    for x in ${n}__t1c ${n}__t1c_tts2; do lat $s $x $SPD/$x/talker; done
    log "$n done (T1c)"
    return 0
  fi
  if [ $D = en2zh ] && [ "${TEST_TRIGGER:-}" = s2 ] && [ -z "${DRY_RUN:-}" ]; then   # talker after the decode + merge
    adopt $TV/${r}$s || setsid nohup $C1 < /dev/null > $TV/${r}$s.launch.log 2>&1 &
    log "$n decode orchestrator pid $!"
    wait_done $TV/${r}$s; [ -z "$(orch_failed $TV/${r}$s)" ] || { flag "$n decode failed ($(tail -1 $TV/${r}$s/orch.log | cut -c1-200))"; return 1; }
    lat $s $n
    read_merge $n "$SEP" $WIN $TV/merge_$n.log || { flag "merge $n ($(tail -1 $TV/merge_$n.log | cut -c1-200))"; return 1; }
    log "$n: waiting for (a)'s PL-EM talker"
    until [ -n "$(en2zh_talker)" ]; do
      [ -n "$(en2zh_failed)" ] && { flag "(a) stopped before its PL-EM talker ($(en2zh_failed | cut -c1-200))"; return 1; }
      sleep 60
    done
    tx=$(en2zh_talker)
    [ -e $tx/uniss/model.safetensors ] || { flag "$n: no En->Zh PL-EM export ('$tx')"; return 1; }
    local C2e="$JORCH --dir $D --set $K --base $b --run $TV/${r}${s}_sp --name ${n}__rlplem --ckpt - --speech-from ${n}_rl --talker plem --plem-export $tx --window 1 --cap135 --scorer-port0 $p"
    adopt $TV/${r}${s}_sp || setsid nohup $C2e < /dev/null > $TV/${r}${s}_sp.launch.log 2>&1 &
    log "${n}__rlplem speech orchestrator pid $!"
    wait_done $TV/${r}${s}_sp; [ -z "$(orch_failed $TV/${r}${s}_sp)" ] || { flag "${n}__rlplem speech failed"; return 1; }
    for x in ${n}__rlplem ${n}__rlplem_tts2; do lat $s $x $SPD/$x/talker; done
    log "$n done"
    return 0
  fi
  if [ $D = en2zh ]; then
    tx=$(en2zh_talker); [ -n "${DRY_RUN:-}" ] && tx=${tx:-$X}
    [ -n "$tx" ] && [ -e $tx/uniss/model.safetensors ] || { flag "$n: no En->Zh PL-EM export ('$tx')"; return 1; }
    pe="--plem-export $tx"
  fi
  local C2="$JORCH --dir $D --set $K --base $b --run $TV/${r}${s}_sp --name ${n}__rlplem --ckpt - --speech-from ${n}_rl --talker plem $pe --window 1 --cap135 --scorer-port0 $p"
  if [ -n "${DRY_RUN:-}" ]; then echo "${r}$s: $C1"; $C1 --dry | tail -1; echo "${r}${s}_sp: $C2"; $C2 --dry | tail -1; return 0; fi
  adopt $TV/${r}$s || setsid nohup $C1 < /dev/null > $TV/${r}$s.launch.log 2>&1 &
  log "$n decode orchestrator pid $!"
  wait_done $TV/${r}$s; [ -z "$(orch_failed $TV/${r}$s)" ] || { flag "$n decode failed ($(tail -1 $TV/${r}$s/orch.log | cut -c1-200))"; return 1; }
  lat $s $n
  read_merge $n "$SEP" $WIN $TV/merge_$n.log || { flag "merge $n ($(tail -1 $TV/merge_$n.log | cut -c1-200))"; return 1; }
  adopt $TV/${r}${s}_sp || setsid nohup $C2 < /dev/null > $TV/${r}${s}_sp.launch.log 2>&1 &
  log "${n}__rlplem speech orchestrator pid $!"
  wait_done $TV/${r}${s}_sp; [ -z "$(orch_failed $TV/${r}${s}_sp)" ] || { flag "${n}__rlplem speech failed"; return 1; }
  for x in ${n}__rlplem ${n}__rlplem_tts2; do lat $s $x $SPD/$x/talker; done
  log "$n done"
}
if [ -n "${DRY_RUN:-}" ]; then
  B=eval_outputs/uniss_qwen3_omni_v1/thinker_gstar_o2_20261029b/thinker_merged
  echo "(DRY: O2 + /nonexistent checkpoints stand in for the selections)"
  k=0
  for r in b m; do for s in ami ali en; do
    [ $r = b ] && ck=/nonexistent/identity || ck=/nonexistent/ckpt/step_03000
    chain $r $s DRY $B $ck $((PORT0 + 10 * k)); k=$((k + 1))
  done; done
  exit 0
fi
for f in $DONE_M $SUM $TV; do [ -n "${TEST_RESUME:-}" ] && [ $f = $TV ] && continue; [ -e $f ] && { log "拒绝覆盖已有输出: $f (the test runs once)"; exit 1; }; done
W=$DEVD; [ "${TEST_TRIGGER:-}" = s2 ] && W=$M2
log "test: waiting for $W"
until [ -e $W ]; do
  grep -q "JOINT_STOP" $JL/dev.log $JL/stage1.log $JL/stage2.log 2>/dev/null && { log "JOINT_STOP test: an earlier stage stopped"; exit 1; }
  sleep 60
done
read -r _ S1STEP _ S1RUN _ MERGED _ IDENT _ <<< "$(cat $M1)"
read -r _ S2STEP _ S2RUN _ CK2 _ B2 _ <<< "$(cat $M2)"
RT1=${S1RUN}_s$(printf %05d $S1STEP); RT2=${S2RUN}_s$(printf %05d $S2STEP)
mkdir -p $TV
log "TEST once: baseline $RT1 ($MERGED + identity), method $RT2 ($CK2), En->Zh talker $(en2zh_talker)"
PIDS=""
k=0
for r in b m; do for s in ami ali en; do
  if [ $r = b ]; then ( chain b $s $RT1 $MERGED $IDENT $((PORT0 + 10 * k)) ) & else ( chain m $s $RT2 $B2 $CK2 $((PORT0 + 10 * k)) ) & fi
  PIDS="$PIDS $!"; k=$((k + 1))
  sleep 120
done; done
rc=0; for p in $PIDS; do wait $p || rc=1; done
[ $rc = 0 ] && ! ls $TV/FAILED_* > /dev/null 2>&1 || { log "JOINT_STOP test: a chain failed ($(cat $TV/FAILED_* 2>/dev/null | tr '\n' ' ' | cut -c1-300))"; exit 1; }
# ---------------------------------------------------------------------------------------------- gates
log "waiting for (a)'s EN2ZH_TEST_DONE (en2zh reference)"
until grep -q EN2ZH_TEST_DONE $E2Z/test.log 2>/dev/null; do
  [ -n "$(en2zh_failed)" ] && { log "(a) stopped: en2zh gate vs the separate system skipped ($(en2zh_failed | cut -c1-200))"; break; }
  sleep 60
done
AE=ft_en2zh_$(cat $E2Z/RUN_ID)_s$($PY -c "import json; print('%05d' % json.load(open('$E2Z/thinker_selection.json'))['chosen'])" 2>/dev/null)__rlplem
gate() {   # <cand> <ref> <tag>
  [ -e $DEC/$2/score_asr.json ] || { log "gate $1 vs $2: reference missing"; return; }
  local O=$TE/$1; mkdir -p $O
  CUDA_VISIBLE_DEVICES= $PY -m experiments.mist_w3po_v1.evaluation.gate_2seed --cand $DEC/$1 $DEC/${1}_tts2 --ref $DEC/$2 $DEC/${2}_tts2 \
    --out $O/gate_$3.json > $O/gate_$3.log 2>&1
  log "gate $1 vs $2 rc=$?"
}
for s in ami ali en; do gate jtest_m${s}_${RT2}__rlplem jtest_b${s}_${RT1}$BSFX vs_joint_baseline; done
gate jtest_mami_${RT2}__rlplem ft_ami_aprime_s3000__rlplem vs_separate_aprl_plem
gate jtest_mali_${RT2}__rlplem ft_ali_aprime_s3000__rlplem vs_separate_aprl_plem
gate jtest_men_${RT2}__rlplem $AE vs_separate_aprl_plem
$PY - $SUM <<EOF > $TV/summary.log 2>&1 || log "summary failed (see $TV/summary.log)"
import json, sys
from pathlib import Path
DEC, TE = Path("$DEC"), Path("$TE")
def j(p):
    try:
        return json.load(open(p))
    except Exception:
        return None
res = {}
for r, rt, sf in (("b", "$RT1", "$BSFX"), ("m", "$RT2", "__rlplem")):
    for s in ("ami", "ali", "en"):
        n = f"jtest_{r}{s}_{rt}"
        t = j(DEC / n / "score_text.json") or {}
        c = j(TE / n / f"textlat_chunk_{n}.json")
        row = {"text_all": t.get("tcpBLEU_all"), "text_ov": t.get("tcpBLEU_ov"), "text_1spk": t.get("tcpBLEU_1spk"),
               "text_laal_chunk": c["unit"]["text_laal_online"]["all"]["mean"] if c else None}
        for nm in (n + sf, n + sf + "_tts2"):
            a = j(DEC / nm / "score_asr.json") or {}
            l = j(TE / (n + sf) / f"lat_{nm}.json"); lc = j(TE / (n + sf) / f"lat_chunk_{nm}.json")
            g = lambda d: d["unit"]["speech_laal_online"]["all"]["mean"] if d else None
            row[nm] = {"asr_all": a.get("tcpBLEU_all"), "asr_ov": a.get("tcpBLEU_ov"), "speech_laal": g(l), "speech_laal_chunk": g(lc)}
        if r == "m":
            row["gates"] = {t_: j(TE / (n + sf) / f"gate_{t_}.json") for t_ in ("vs_joint_baseline", "vs_separate_aprl_plem")}
        res[n] = row
with open(sys.argv[1], "x") as fh:
    json.dump(res, fh, indent=1)
print(json.dumps({k: v["text_all"] for k, v in res.items()}))
EOF
log "summary: $(tail -1 $TV/summary.log | cut -c1-300)"
( set -o noclobber; echo "s1 $RT1 s2 $RT2 utc $(date -u +%FT%TZ)" > $DONE_M ) || log "marker exists"
log "JOINT_TEST_DONE baseline $RT1 method $RT2"
