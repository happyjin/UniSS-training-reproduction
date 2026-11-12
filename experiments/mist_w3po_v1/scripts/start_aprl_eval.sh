#!/usr/bin/env bash
# A'rl (reports/mist_w3po_v1/aprime_rl_prereg_v1.zh-CN.md, 7fa0aba3): speech chains only on A''s dev decode with
# read-level talker blocks (decode/dev_crf_v2p_s3000_finalpost_s8_c2g06_rl, merge_read_segments.py), two talkers x two TTS
# seeds, each with gate_0 vs A' (c1+b1+s2) and gate_1 vs A' + PL-EM:
#   t_eval/aprl      --talker c1b1s2 -> decode/dev_crf_v2p_s3000_finalpost_s8_c2g06__rl      (scorer ports 19101-19104)
#   t_eval/aprlplem  --talker plem   -> decode/dev_crf_v2p_s3000_finalpost_s8_c2g06__rlplem  (scorer ports 19111-19114)
# Cards: waits until T4's dev5 quick check has started all 4 decodes (t_eval/t4quick/state) or written DONE, then starts
# both orchestrators (--cap135, shared ledger) with --exclude 0 1 2 3 (VE1 trains there; the shared GLOBAL_EXCLUDE
# keeps every owned slot's cards out anyway). Polls every 20 s. bash -n / DRY_RUN only.
#   [DRY_RUN=1] bash experiments/mist_w3po_v1/scripts/start_aprl_eval.sh
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
PY=/opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python
E=logs/mist_w3po_v1/t_eval
A=dev_crf_v2p_s3000_finalpost_s8_c2g06
P=${A}__plem_c1b1s2
SRC=${A}_rl
log() { echo "$(date -u +%FT%TZ) $*"; }
q4_started() {
  [ -e $E/t4quick/events/DONE ] && return 0
  local k
  for k in 0 1 2 3; do ls $E/t4quick/state/dev5_t4_s3000__dec$k.a*.pid > /dev/null 2>&1 || return 1; done
  return 0
}
cmd() {   # <run> <name> <talker> <port0>
  echo "$PY experiments/mist_w3po_v1/scripts/thinker_t_eval_orch.py --run $E/$1 --name $2 --ckpt - --speech-from $SRC \
--talker $3 --window 1 --gate-ref $A ${A}_tts2 --gate-ref $P ${P}_tts2 --scorer-port0 $4 --cap135 --exclude 0 1 2 3"
}
C1=$(cmd aprl ${A}__rl c1b1s2 19101)
C2=$(cmd aprlplem ${A}__rlplem plem 19111)
[ -s eval_outputs/mist_w3po_v1/decode/$SRC/hyp.jsonl ] && [ -s eval_outputs/mist_w3po_v1/decode/$SRC/score_text.json ] \
  || { log "merged decode $SRC missing"; exit 1; }
if [ -n "${DRY_RUN:-}" ]; then
  echo "T4 quick decodes started: $(q4_started && echo yes || echo no)"
  for c in "$C1" "$C2"; do echo "CMD: $c"; $c --dry | tail -1; done
  exit 0
fi
for r in aprl aprlplem; do [ -e $E/$r ] && { log "拒绝覆盖已有输出: $E/$r"; exit 1; }; done
log "waiting for T4's quick-check decodes"
until q4_started; do sleep 20; done
setsid nohup $C1 < /dev/null > $E/aprl.launch.log 2>&1 &
log "aprl (c1b1s2) orchestrator pid $!"
sleep 30
setsid nohup $C2 < /dev/null > $E/aprlplem.launch.log 2>&1 &
log "aprlplem (PL-EM) orchestrator pid $!"
