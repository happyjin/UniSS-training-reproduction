#!/usr/bin/env bash
# v3 = start_t2rl_eval_v2.sh + one rule (coordinator ~23:00 Prague: D1 takes 4-7 right after its roll-in passes R4):
#   4-7 also excluded while logs/mist_w3po_v1/d1/claiming_4567 exists (start_d1_after.sh: D1 is about to claim 4-7)
#        and no slotB.released_* is newer than it (after D1's own release the marker no longer applies).
# v2: start_t2rl_eval.sh (539ed495, left unchanged) with a card-priority scheduler in place of "wait for T5's exit, then
# 0-3" (coordinator 2026-11-11 ~22:30 Prague: T5's dev5 quick check and, on GO, its dev full eval go first; 4-7 may be
# given to t2rl once T4's quick check and T1's resume have started). The orchestrator command is the same:
# thinker_t_eval_orch --speech-from dev_t2_latent_t0_s3000_w05_rl, talker c1+b1+s2, 2 TTS seeds, gate_0 vs A',
# gate_1 vs T2 (unmerged), scorer ports 18941-18944, --cap135 (<= 135 GiB per card, shared ledger).
# Every 60 s the cards t2rl must not take NEW jobs on are recomputed:
#   0-3  slotA owned (T5 training; a failed T5 keeps it)
#        | T5 exited rc=0 and its quick check (t_eval/t5quick) has neither started all 4 decodes nor written DONE
#   4-7  slotB owned (T4 training)
#        | T4 exited rc=0 and its quick check (t_eval/t4quick) has neither started all 4 decodes nor written DONE
#        | T4 exited and T1's resume has not happened yet (t1_resume_after_t4.log: "resumed" or "not resumed")
#   0-7  T4's or T5's quick check said GO (go.json) and that dev full eval (t_eval/t4 | t_eval/t5) has neither started
#        all 8 decodes nor written DONE (at most 90 min after go.json)
# The orchestrator starts (with --exclude <set>) the first time some card is allowed; afterwards a changed set is
# written to t_eval/t2rl/exclude_cards (new file + command mv -f; re-read every tick by the orchestrator; running jobs
# are never touched). Ends when t2rl writes events/DONE (or its orchestrator is gone without DONE). bash -n / DRY_RUN only.
#   [DRY_RUN=1] bash experiments/mist_w3po_v1/scripts/start_t2rl_eval_v3.sh
#   setsid nohup bash .../start_t2rl_eval_v3.sh > logs/mist_w3po_v1/t_eval/start_t2rl_eval_v3.log 2>&1 < /dev/null &
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
PY=/opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python
E=logs/mist_w3po_v1/t_eval
SL=logs/mist_w3po_v1/slots
TE=eval_outputs/mist_w3po_v1/crf/t_eval
T4LOG=logs/mist_w3po_v1/t4_dart_t0_w10_20261111T173134Z.log
T5LOG=logs/mist_w3po_v1/t5_evid_p4_20261111T154210Z.log
T1RES=$E/t1_resume_after_t4.log
MARK=logs/mist_w3po_v1/d1/claiming_4567
RUN=$E/t2rl
SRC=dev_t2_latent_t0_s3000_w05
A=dev_crf_v2p_s3000_finalpost_s8_c2g06
log() { echo "$(date -u +%FT%TZ) $*"; }
orch_pids() {
  ps -ww -eo pid=,args= | awk -v r="$1" '$2 ~ /\/python[0-9.]*$/ && $3 == "experiments/mist_w3po_v1/scripts/thinker_t_eval_orch.py" \
    && $4 == "--run" && $5 == r {print $1}'
}
# all <name>__dec<k> (k < n) of run dir $1 have a pid file, or the run wrote DONE
decs_started() {
  local r=$1 name=$2 n=$3 k
  [ -e $r/events/DONE ] && return 0
  for k in $(seq 0 $((n - 1))); do ls $r/state/${name}__dec$k.a*.pid > /dev/null 2>&1 || return 1; done
  return 0
}
exited_ok() { [ -e "$1" ] && grep -q 'CRF_RUN_EXIT rc=0' "$1"; }
exited() { [ -e "$1" ] && grep -q 'CRF_RUN_EXIT' "$1"; }
# GO phase of tag $1: go.json says GO and the full eval has not started its 8 decodes, for at most 90 min
go_phase() {
  local g=$TE/dev5_$1_s3000/go.json
  [ -e $g ] && grep -q '"GO": true' $g || return 1
  [ $(( $(date +%s) - $(stat -c %Y $g) )) -lt 5400 ] || return 1
  decs_started $E/$1 dev_$1_s3000 8 && return 1
  return 0
}
excl() {
  local lo=0 hi=0
  [ -e $SL/slotA.owner ] && lo=1
  exited_ok $T5LOG && ! decs_started $E/t5quick dev5_t5_s3000 4 && lo=1
  [ -e $SL/slotB.owner ] && hi=1
  exited_ok $T4LOG && ! decs_started $E/t4quick dev5_t4_s3000 4 && hi=1
  exited $T4LOG && ! grep -qE 'T1 orchestrator resumed|T1 not resumed' $T1RES 2>/dev/null && hi=1
  [ -e $MARK ] && ! find $SL -maxdepth 1 -name 'slotB.released_*' -newer $MARK | grep -q . && hi=1
  { go_phase t4 || go_phase t5; } && lo=1 && hi=1
  local x=""
  [ $lo = 1 ] && x="0 1 2 3"
  [ $hi = 1 ] && x="$x 4 5 6 7"
  echo "${x# }"
}
CMD0="$PY experiments/mist_w3po_v1/scripts/thinker_t_eval_orch.py --run $RUN --name ${SRC}__rl --ckpt - \
--speech-from ${SRC}_rl --talker c1b1s2 --window 0.5 --gate-ref $A ${A}_tts2 --gate-ref $SRC ${SRC}_tts2 \
--scorer-port0 18941 --cap135 --exclude"
[ -s eval_outputs/mist_w3po_v1/decode/${SRC}_rl/hyp.jsonl ] && [ -s eval_outputs/mist_w3po_v1/decode/${SRC}_rl/score_text.json ] \
  || { log "merged decode ${SRC}_rl missing"; exit 1; }
if [ -n "${DRY_RUN:-}" ]; then
  X=$(excl)
  echo "excluded now: '$X'"; echo "CMD: $CMD0 $X"
  echo "T4 exit: $(grep -o 'CRF_RUN_EXIT rc=[0-9-]*' $T4LOG 2>/dev/null || echo none); T5 exit: $(grep -o 'CRF_RUN_EXIT rc=[0-9-]*' $T5LOG 2>/dev/null || echo none)"
  echo "t2rl run dir exists: $([ -e $RUN ] && echo yes || echo no)"
  exit 0
fi
[ -e $RUN ] && { log "拒绝覆盖已有输出: $RUN"; exit 1; }
log "waiting for a card t2rl may use (rules in the header)"
X=$(excl)
until [ "$X" != "0 1 2 3 4 5 6 7" ]; do sleep 60; X=$(excl); done
setsid nohup $CMD0 $X < /dev/null > $RUN.launch.log 2>&1 &
log "t2rl orchestrator pid $! (excluded: '$X')"
CUR=$X
sleep 120
while [ ! -e $RUN/events/DONE ]; do
  [ -n "$(orch_pids $RUN)" ] || { log "t2rl orchestrator gone without DONE: scheduler stops"; exit 1; }
  X=$(excl)
  if [ "$X" != "$CUR" ]; then
    T=$RUN/exclude_cards.new_$(date -u +%Y%m%dT%H%M%SZ)
    ( set -o noclobber; echo "$X" > $T ) && command mv -f $T $RUN/exclude_cards && log "t2rl exclude_cards '$CUR' -> '$X'" && CUR=$X
  fi
  sleep 60
done
log "t2rl DONE"
