#!/usr/bin/env bash
# Plan A TEST, run once (user 2026-11-14 ~21:20 Prague: "先用A跑测试集出最终结果"): the method row of joint_test.sh
# (joint v2' stage 2 step 3500 on the joint v8 s6000 merged base) with the inference-time write bias fixed on dev
# (wb_dev5.sh calibration -> BZ = BE = -1, confirmed on the full dev by wb_dev.sh). Nothing is chosen on test.
# Sets: AMI zh->en, AliMeeting zh->en, AMI En->Zh (joint_test.sh's setvars). Per set: text decode through
# joint_orch_wb.py --wait-bias <BZ|BE> (8 shards by meeting, window 1.0, --cap135, all 8 cards) -> chunk text LAAL ->
# read-level merge -> PL-EM speech, two seeds (zh2en: the existing zh2en PL-EM talker; En->Zh: (a)'s selected
# export) -> chunk speech LAAL -> gate_2seed vs the strict joint baseline's existing test outputs
# (jtest_b<set>_joint_v8h6k_20261113T073658Z_s06000__t1c, frozen T1c) and vs the unbiased method
# (jtest_m<set>_joint_v2p_20261113T120415Z_s03500__rlplem). Names jtest_m<set>_<RT>, RT =
# joint_v2p_20261113T120415Z_s03500_wbz<tag BZ>e<tag BE>; run dir logs/mist_w3po_v1/joint/wb/test; summary
# eval_outputs/mist_w3po_v1/joint/wb_test_summary_v1.json; marker logs/mist_w3po_v1/joint/wb/WB_TEST_DONE. Refuses to
# overwrite any of them. No COMET / MetricX. Functions copied from joint_test.sh (chain's default PL-EM branch) with the
# text orchestrator swapped; joint_test.sh is not changed.
#   BZ=-1 BE=-1 [PORT0=19700] [DRY_RUN=1] bash experiments/mist_w3po_v1/scripts/wb_test.sh
# Run detached: setsid nohup bash ... > logs/mist_w3po_v1/joint/wb/wb_test.log 2>&1 < /dev/null &
# Checked with bash -n and DRY_RUN=1 only.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
. experiments/mist_w3po_v1/scripts/joint_lib.sh
BZ=${BZ:?}; BE=${BE:?}
PORT0=${PORT0:-19700}
TV=$JL/wb/test
SUM=$JO/wb_test_summary_v1.json
DONE_M=$JL/wb/WB_TEST_DONE
WORCH="$PY $SC/joint_orch_wb.py"
B2=$JO/joint_v8h6k_20261113T073658Z/merged_s06000/thinker_merged
CK2=$CRF/train/joint_v2p_20261113T120415Z/ckpt/step_03500
RT1=joint_v8h6k_20261113T073658Z_s06000; BSFX=__t1c
RT0=joint_v2p_20261113T120415Z_s03500
tag() { echo "$1" | sed 's/^-/m/; s/\./p/'; }
RT=${RT0}_wbz$(tag $BZ)e$(tag $BE)
flag() { ( set -o noclobber; echo "$(date -u +%FT%TZ) $*" > $TV/FAILED_$(date -u +%s%N) ); log "WB_STOP test: $*"; }
setvars() {   # <set ami|ali|en> -> D (dir flag) K (joint_orch --set) WIN CW REF DR SEP WB
  case $1 in
    ami) D=zh2en; K=test; WIN=$WT_AMI; CW=$WT_AMIC; REF=$RT_AMI; DR=zh2en; SEP=' '; WB=$BZ;;
    ali) D=zh2en; K=ali;  WIN=$WT_ALI; CW=$WT_ALIC; REF=$RT_ALI; DR=ali; SEP=' '; WB=$BZ;;
    en)  D=en2zh; K=test; WIN=$WT_EN;  CW=$WT_ENC;  REF=$RE_TEST; DR=""; SEP=''; WB=$BE;;
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
chain() {   # <set> <port0>
  local s=$1 p=$2 n tx pe=""
  setvars $s
  n=jtest_m${s}_$RT
  local C1="$WORCH --wait-bias $WB --dir $D --set $K --base $B2 --run $TV/m$s --name $n --ckpt $CK2 --talker none --window 1 --cap135"
  if [ $D = en2zh ]; then
    tx=$(en2zh_talker)
    [ -n "$tx" ] && [ -e $tx/uniss/model.safetensors ] || { flag "$n: no En->Zh PL-EM export ('$tx')"; return 1; }
    pe="--plem-export $tx"
  fi
  local C2="$JORCH --dir $D --set $K --base $B2 --run $TV/m${s}_sp --name ${n}__rlplem --ckpt - --speech-from ${n}_rl --talker plem $pe --window 1 --cap135 --scorer-port0 $p"
  if [ -n "${DRY_RUN:-}" ]; then echo "m$s: $C1"; $C1 --dry | tail -1; echo "m${s}_sp: $C2"; $C2 --dry | tail -1; return 0; fi
  setsid nohup $C1 < /dev/null > $TV/m$s.launch.log 2>&1 &
  log "$n decode orchestrator pid $!"
  wait_done $TV/m$s; [ -z "$(orch_failed $TV/m$s)" ] || { flag "$n decode failed ($(tail -1 $TV/m$s/orch.log | cut -c1-200))"; return 1; }
  lat $s $n
  read_merge $n "$SEP" $WIN $TV/merge_$n.log || { flag "merge $n ($(tail -1 $TV/merge_$n.log | cut -c1-200))"; return 1; }
  setsid nohup $C2 < /dev/null > $TV/m${s}_sp.launch.log 2>&1 &
  log "${n}__rlplem speech orchestrator pid $!"
  wait_done $TV/m${s}_sp; [ -z "$(orch_failed $TV/m${s}_sp)" ] || { flag "${n}__rlplem speech failed"; return 1; }
  for x in ${n}__rlplem ${n}__rlplem_tts2; do lat $s $x $SPD/$x/talker; done
  log "$n done"
}
for s in ami ali en; do [ -e $DEC/jtest_b${s}_$RT1$BSFX/score_asr.json ] || { log "baseline test output missing: jtest_b${s}_$RT1$BSFX"; exit 1; }; done
if [ -n "${DRY_RUN:-}" ]; then
  k=0; for s in ami ali en; do chain $s $((PORT0 + 10 * k)); k=$((k + 1)); done
  exit 0
fi
for f in $DONE_M $SUM $TV; do [ -e $f ] && { log "拒绝覆盖已有输出: $f (the test runs once)"; exit 1; }; done
mkdir -p $TV
log "plan A TEST once: method $RT0 + wait bias zh2en $BZ / en2zh $BE (run tag $RT), En->Zh talker $(en2zh_talker)"
PIDS=""; k=0
for s in ami ali en; do
  ( chain $s $((PORT0 + 10 * k)) ) &
  PIDS="$PIDS $!"; k=$((k + 1))
  sleep 120
done
rc=0; for p in $PIDS; do wait $p || rc=1; done
[ $rc = 0 ] && ! ls $TV/FAILED_* > /dev/null 2>&1 || { log "WB_STOP test: a chain failed ($(cat $TV/FAILED_* 2>/dev/null | tr '\n' ' ' | cut -c1-300))"; exit 1; }
gate() {   # <cand> <ref> <tag>
  [ -e $DEC/$2/score_asr.json ] || { log "gate $1 vs $2: reference missing"; return; }
  local O=$TE/$1; mkdir -p $O
  CUDA_VISIBLE_DEVICES= $PY -m experiments.mist_w3po_v1.evaluation.gate_2seed --cand $DEC/$1 $DEC/${1}_tts2 --ref $DEC/$2 $DEC/${2}_tts2 \
    --out $O/gate_$3.json > $O/gate_$3.log 2>&1
  log "gate $1 vs $2 rc=$?"
}
for s in ami ali en; do
  gate jtest_m${s}_${RT}__rlplem jtest_b${s}_${RT1}$BSFX vs_joint_baseline
  gate jtest_m${s}_${RT}__rlplem jtest_m${s}_${RT0}__rlplem vs_method_nobias
done
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
for r, rt, sf in (("b", "$RT1", "$BSFX"), ("m", "$RT0", "__rlplem"), ("m", "$RT", "__rlplem")):
    for s in ("ami", "ali", "en"):
        n = f"jtest_{r}{s}_{rt}"
        t = j(DEC / n / "score_text.json") or {}
        c = j(TE / n / f"textlat_chunk_{n}.json")
        row = {"text_all": t.get("tcpBLEU_all"), "text_ov": t.get("tcpBLEU_ov"), "text_1spk": t.get("tcpBLEU_1spk"),
               "text_laal_chunk": c["unit"]["text_laal_online"]["all"]["mean"] if c else None}
        for nm in (n + sf, n + sf + "_tts2"):
            a = j(DEC / nm / "score_asr.json") or {}
            lc = j(TE / (n + sf) / f"lat_chunk_{nm}.json")
            u = (lc or {}).get("unit") or {}
            g = lambda k: u[k]["all"]["mean"] if k in u else None
            row[nm] = {"asr_all": a.get("tcpBLEU_all"), "asr_ov": a.get("tcpBLEU_ov"), "speech_laal_chunk": g("speech_laal_online"),
                       "speech_end_chunk": g("speech_end_online"), "speech_start_chunk": g("speech_start_online")}
        if rt == "$RT":
            row["gates"] = {t_: j(TE / (n + sf) / f"gate_{t_}.json") for t_ in ("vs_joint_baseline", "vs_method_nobias")}
        res[n] = row
with open(sys.argv[1], "x") as fh:
    json.dump(res, fh, indent=1)
print(json.dumps({k: v["text_all"] for k, v in res.items()}))
EOF
log "summary: $(tail -1 $TV/summary.log | cut -c1-300)"
( set -o noclobber; echo "$RT utc $(date -u +%FT%TZ)" > $DONE_M ) || log "marker exists"
log "WB_TEST_DONE $RT"
