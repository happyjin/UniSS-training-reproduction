#!/usr/bin/env bash
# Rule C NOT triggered: replaces t3_eval_wait.sh (stopped by PID before this is armed). When the T3 training exits
# (CRF_RUN_EXIT in its log), T4 takes cards 4-7 at once (start_t4.sh without STOP_T3: slotB release + claim, takeover
# of any placeholder, run_crf_v2p), then the T3 dev eval orchestrator is started with EVERY card excluded
# (<run>/exclude_cards, re-read each tick): it holds no GPU until cards are released to it by editing that file (after
# a training slot frees; coordinator's call), so it can never collide with the T4 / T5 trainings. bash -n / DRY_RUN only.
#   T4_RUN=<id> [DRY_RUN=1] bash experiments/mist_w3po_v1/scripts/t3_end_then_t4.sh
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
T4_RUN=${T4_RUN:?}
R3=t3_joint_t0_20261111T095841Z
TR=eval_outputs/mist_w3po_v1/crf/train
C=eval_outputs/mist_w3po_v1/crf
A=dev_crf_v2p_s3000_finalpost_s8_c2g06
P=${A}__plem_c1b1s2
L=logs/mist_w3po_v1/$R3.log
log() { echo "$(date -u +%FT%TZ) $*"; }
ORCH="/opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python experiments/mist_w3po_v1/scripts/thinker_t_eval_orch.py \
--run logs/mist_w3po_v1/t_eval/t3 --name dev_t3_joint_t0_s3000_w05 --scorer-port0 18841 --ckpt $TR/$R3/ckpt/step_03000 \
--talker plem --gate-ref $P ${P}_tts2 --gate-ref $A ${A}_tts2 --cap135 --exclude 0 1 2 3 4 5 6 7"
if [ -n "${DRY_RUN:-}" ]; then
  SCHED=$C/sched_piece_t0_g1000_v1 DART=$C/t4_dart_w10_v1 RUN_ID=$T4_RUN DRY_RUN=1 bash experiments/mist_w3po_v1/scripts/start_t4.sh
  echo "then: $ORCH"; exit 0
fi
until grep -q CRF_RUN_EXIT $L; do sleep 10; done
log "T3 exit: $(grep CRF_RUN_EXIT $L)"
SCHED=$C/sched_piece_t0_g1000_v1 DART=$C/t4_dart_w10_v1 RUN_ID=$T4_RUN bash experiments/mist_w3po_v1/scripts/start_t4.sh
if grep -q 'CRF_RUN_EXIT rc=0' $L && [ -e $TR/$R3/ckpt/step_03000/DONE ]; then
  mkdir -p logs/mist_w3po_v1/t_eval
  setsid nohup $ORCH < /dev/null > logs/mist_w3po_v1/t_eval/t3.launch.log 2>&1 &
  log "T3 eval orchestrator pid $! (all cards excluded until released)"
else
  log "T3 did not finish cleanly: no eval"
fi
