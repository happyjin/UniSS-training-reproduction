#!/usr/bin/env bash
# After a D1 / VE1 training (thinker_dagger_d1_prereg_v1 §5, thinker_v2p_evidence_ve1_prereg_v1 §4; = J1 prereg §6-§7):
# t_after_train.sh's flow with the registered differences, as a NEW script (t_after_train.sh untouched):
#   1. wait for CRF_RUN_EXIT; rc=0 with step_03000/DONE, else stop (slot kept for inspection); release the slot (rename)
#   2. dev5 quick check on CARDS (thinker_t_eval_orch --talker none --win dev5, 4 window shards, 1.0 s reads, --cap135,
#      the other 4 cards excluded), stall diag vs A' dev5, GO lines 1-3 (t_quick_go.py -> go.json) and line 4: the CPU
#      speech proxy (crf/joint_speech_proxy.py, read-level blocks) of the dev5 decode < A''s dev5 proxy (its own
#      segments) -> go4.json
#   3. GO (all four) -> dev decode (8 shards, --talker none: decode + text score + text latency), then the read-level
#      merged copy (crf/merge_read_segments.py -> <name>_rl), then the PRIMARY speech chain on it (--speech-from,
#      c1+b1+s2, 2 TTS seeds, gate vs A'), then the per-segment ablation chain on the unmerged decode (gate vs A')
#   Cards: the quick check on CARDS; the dev stages exclude the cards of any slot still owned at their start.
#   RUN=<training run id> TAG=<d1|ve1> CARDS=0,1,2,3|4,5,6,7 SLOT=slotA|slotB PORT0=<free scorer port base> [DRY_RUN=1] \
#     bash experiments/mist_w3po_v1/scripts/t_after_train_rl.sh
# Run detached (setsid nohup ... < /dev/null &). Checked with bash -n and DRY_RUN=1 only.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
RUN=${RUN:?}; TAG=${TAG:?}; CARDS=${CARDS:?}; SLOT=${SLOT:?}; PORT0=${PORT0:?}
export PYTHONPATH=.
PY=/opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python
TR=eval_outputs/mist_w3po_v1/crf/train
SL=logs/mist_w3po_v1/slots
E=logs/mist_w3po_v1/t_eval
DEC=eval_outputs/mist_w3po_v1/decode
W5=eval_outputs/mist_w3po_v1/data/windows_zh2en_c2g06_v1_dev5/windows_dev_online.jsonl
A=dev_crf_v2p_s3000_finalpost_s8_c2g06
AP5=logs/mist_w3po_v1/j1/proxy/proxy_dev5_aprime.json
Q=dev5_${TAG}_s3000
F=dev_${TAG}_s3000
QO=eval_outputs/mist_w3po_v1/crf/t_eval/$Q
CK=$TR/$RUN/ckpt/step_03000
case $CARDS in 4,5,6,7) EXQ="0 1 2 3";; 0,1,2,3) EXQ="4 5 6 7";; *) echo "CARDS 0,1,2,3 or 4,5,6,7"; exit 1;; esac
log() { echo "$(date -u +%FT%TZ) $*"; }
excl() { local x=""; [ -e $SL/slotA.owner ] && x="$x 0 1 2 3"; [ -e $SL/slotB.owner ] && x="$x 4 5 6 7"; echo $x; }
ORCH="$PY experiments/mist_w3po_v1/scripts/thinker_t_eval_orch.py"
QCMD="$ORCH --run $E/${TAG}quick --name $Q --ckpt $CK --talker none --win $W5 --shards 4 --shard-by window --window 1 \
--scorer-port0 $PORT0 --exclude $EXQ --cap135"
dcmd() { echo "$ORCH --run $E/$TAG --name $F --ckpt $CK --talker none --window 1 --scorer-port0 $((PORT0 + 10)) --cap135 --exclude $(excl)"; }
rlcmd() { echo "$ORCH --run $E/${TAG}rl --name ${F}__rl --ckpt - --speech-from ${F}_rl --talker c1b1s2 --window 1 --gate-ref $A ${A}_tts2 --scorer-port0 $((PORT0 + 20)) --cap135 --exclude $(excl)"; }
segcmd() { echo "$ORCH --run $E/${TAG}seg --name ${F}__seg --ckpt - --speech-from $F --talker c1b1s2 --window 1 --gate-ref $A ${A}_tts2 --scorer-port0 $((PORT0 + 30)) --cap135 --exclude $(excl)"; }
wait_done() { until [ -e $1/events/DONE ]; do sleep 30; done; }
if [ -n "${DRY_RUN:-}" ]; then
  echo "quick: $QCMD"; echo "decode: $(dcmd)"; echo "merge: $PY -m experiments.mist_w3po_v1.crf.merge_read_segments --src $F --dst ${F}_rl"
  echo "speech (read-level, primary): $(rlcmd)"; echo "speech (per segment, ablation): $(segcmd)"
  echo "A' dev5 proxy: $([ -e $AP5 ] && $PY -c "import json; print(json.load(open('$AP5'))['speech_laal'])" || echo "not yet (j1_decide writes it; else computed here)")"
  exit 0
fi
L=logs/mist_w3po_v1/$RUN.log
until [ -e $L ] && grep -q CRF_RUN_EXIT $L; do sleep 30; done
log "$RUN exit: $(grep CRF_RUN_EXIT $L)"
if ! grep -q 'CRF_RUN_EXIT rc=0' $L || [ ! -e $CK/DONE ]; then log "$RUN did not finish cleanly: no quick check (slot kept)"; exit 1; fi
[ -e $SL/$SLOT.owner ] && grep -q "$RUN" $SL/$SLOT.owner && \
  command mv -f $SL/$SLOT.owner $SL/$SLOT.released_$(date -u +%Y%m%dT%H%M%SZ) && log "$SLOT released"
mkdir -p $E
setsid nohup $QCMD < /dev/null > $E/${TAG}quick.launch.log 2>&1 &
log "quick check orchestrator pid $!"
wait_done $E/${TAG}quick
log "quick check DONE"
python3 eval_outputs/mist_w3po_v1/speech_laal_v1/analysis_20261111/t1_stall_diag.py $DEC/$Q/hyp.jsonl $DEC/plcrf_dev5_aprule/hyp.jsonl \
  eval_outputs/mist_w3po_v1/crf/t0_agree_dev5/analysis/pieces.jsonl $W5 > $QO/stall_diag.txt 2>&1
$PY experiments/mist_w3po_v1/scripts/t_quick_go.py $Q $QO $W5 > $QO/go.log 2>&1
if [ ! -e $AP5 ]; then
  mkdir -p $(dirname $AP5)
  $PY -m experiments.mist_w3po_v1.crf.joint_speech_proxy --run $DEC/plcrf_dev5_aprule --win $W5 --merge 0 --out $AP5 > /dev/null 2>&1
fi
$PY -m experiments.mist_w3po_v1.crf.joint_speech_proxy --run $DEC/$Q --win $W5 --merge 1 --out $QO/proxy_rl.json > $QO/proxy.log 2>&1
$PY - $QO $AP5 <<'EOF'
import json, sys
qo, ap = sys.argv[1], sys.argv[2]
g = json.load(open(f"{qo}/go.json"))
p, a = json.load(open(f"{qo}/proxy_rl.json"))["speech_laal"], json.load(open(ap))["speech_laal"]
r = {"go_lines_1_3": bool(g["GO"]), "proxy_rl": p, "aprime_proxy": a, "go_line_4": bool(p < a)}
r["GO"] = r["go_lines_1_3"] and r["go_line_4"]
with open(f"{qo}/go4.json", "x") as fh:
    json.dump(r, fh, indent=1)
print(json.dumps(r))
EOF
log "GO: $(cat $QO/go4.json | tr '\n' ' ')"
grep -q '"GO": true' $QO/go4.json || { log "NOT GO: no dev evaluation"; exit 0; }
C1=$(dcmd); setsid nohup $C1 < /dev/null > $E/$TAG.launch.log 2>&1 &
log "dev decode orchestrator pid $!: $C1"
wait_done $E/$TAG
$PY -m experiments.mist_w3po_v1.crf.merge_read_segments --src $F --dst ${F}_rl > $E/${TAG}_merge.log 2>&1 || { log "merge failed"; exit 1; }
C2=$(rlcmd); setsid nohup $C2 < /dev/null > $E/${TAG}rl.launch.log 2>&1 &
log "read-level speech orchestrator pid $!: $C2"
wait_done $E/${TAG}rl
C3=$(segcmd); setsid nohup $C3 < /dev/null > $E/${TAG}seg.launch.log 2>&1 &
log "per-segment speech orchestrator pid $!: $C3"
wait_done $E/${TAG}seg
log "all evaluations DONE"
