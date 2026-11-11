#!/usr/bin/env bash
# Resume T1's paused dev eval (start_t5_v2.sh paused it for T5 on 0-3; coordinator 2026-11-11 18:30 UTC): on 4-7 after
# T4's training, stacked with T4's dev5 quick check (<= 135 GiB per card, --cap135) but never ahead of it:
#   1. wait (poll 60 s) for a T1 pause record (t_eval/t1/paused_*.txt) and T4's CRF_RUN_EXIT; T4 rc != 0 -> exit, no
#      resume (t_after_train.sh keeps slotB; a decision for the coordinator);
#   2. wait until the T4 quick check has started all 4 of its decodes (t_eval/t4quick/state/dev5_t4_s3000__dec<k>.a*.pid)
#      or written events/DONE;
#   3. t_eval/t1/exclude_cards (read every tick, the only exclusion on --resume) <- the cards of the slots still owned at
#      that moment (slotA -> 0 1 2 3, slotB -> 4 5 6 7), written as a new file and moved into place (command mv -f);
#   4. relaunch the T1 orchestrator with its original argv (copied from PID 1272714) + nothing else: it already carried
#      --resume --cap135 --dec-after <t2>. Its stopped jobs come back as retries (decodes resume from partial hyps).
# No orchestrator may be live on t_eval/t1 at that moment (checked by exact python argv). bash -n / DRY_RUN only.
#   [DRY_RUN=1] bash experiments/mist_w3po_v1/scripts/t1_resume_after_t4.sh
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
PY=/opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python
E=logs/mist_w3po_v1/t_eval
SL=logs/mist_w3po_v1/slots
T1R=$E/t1
T4=t4_dart_t0_w10_20261111T173134Z
L4=logs/mist_w3po_v1/$T4.log
Q4=$E/t4quick
log() { echo "$(date -u +%FT%TZ) $*"; }
orch_pids() {
  ps -ww -eo pid=,args= | awk -v r="$1" '$2 ~ /\/python[0-9.]*$/ && $3 == "experiments/mist_w3po_v1/scripts/thinker_t_eval_orch.py" \
    && $4 == "--run" && $5 == r {print $1}'
}
q4_decodes_started() {
  [ -e $Q4/events/DONE ] && return 0
  local k
  for k in 0 1 2 3; do ls $Q4/state/dev5_t4_s3000__dec$k.a*.pid > /dev/null 2>&1 || return 1; done
  return 0
}
ARGS=(experiments/mist_w3po_v1/scripts/thinker_t_eval_orch.py --run $T1R --name dev_t1_piece_t0_s3000_w05
  --scorer-port0 18801 --exclude 4 5 6 7
  --ckpt eval_outputs/mist_w3po_v1/crf/train/t1_piece_t0_20261111T094743Z/ckpt/step_03000 --talker c1b1s2
  --gate-ref dev_crf_v2p_s3000_finalpost_s8_c2g06 dev_crf_v2p_s3000_finalpost_s8_c2g06_tts2
  --resume --cap135 --dec-after $E/t2 dev_t2_latent_t0_s3000_w05)
excl() {
  local x=""
  [ -e $SL/slotA.owner ] && x="$x 0 1 2 3"
  [ -e $SL/slotB.owner ] && x="$x 4 5 6 7"
  echo "${x# }"
}
if [ -n "${DRY_RUN:-}" ]; then
  echo "pause records: $(ls $T1R/paused_*.txt 2>/dev/null | tr '\n' ' ')"
  echo "T4 exit: $(grep -o 'CRF_RUN_EXIT rc=[0-9-]*' $L4 2>/dev/null || echo none); T4 quick decodes started: $(q4_decodes_started && echo yes || echo no)"
  echo "T1 orchestrator pids now: $(orch_pids $T1R | tr '\n' ' '); exclude_cards now: $(cat $T1R/exclude_cards); would write: $(excl)"
  echo "CMD: $PY ${ARGS[*]}"
  exit 0
fi
until ls $T1R/paused_*.txt > /dev/null 2>&1 && [ -e $L4 ] && grep -q CRF_RUN_EXIT $L4; do sleep 60; done
log "T1 paused ($(ls $T1R/paused_*.txt | tr '\n' ' ')); T4: $(grep -o 'CRF_RUN_EXIT rc=[0-9-]*' $L4)"
grep -q 'CRF_RUN_EXIT rc=0' $L4 || { log "T4 did not exit cleanly: T1 not resumed (decision needed)"; exit 1; }
until q4_decodes_started; do sleep 60; done
log "T4 quick check: all 4 decodes started (or DONE)"
[ -z "$(orch_pids $T1R)" ] || { log "a T1 orchestrator is already live: $(orch_pids $T1R | tr '\n' ' '); not relaunching"; exit 1; }
X=$(excl)
TMP=$T1R/exclude_cards.new_$(date -u +%Y%m%dT%H%M%SZ)
( set -o noclobber; echo "$X" > $TMP ) && command mv -f $TMP $T1R/exclude_cards && log "t1 exclude_cards <- '$X'"
setsid nohup $PY "${ARGS[@]}" < /dev/null >> $E/t1.launch.log 2>&1 &
log "T1 orchestrator resumed: pid $!"
