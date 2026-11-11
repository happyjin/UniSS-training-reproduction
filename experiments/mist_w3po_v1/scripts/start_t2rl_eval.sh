#!/usr/bin/env bash
# T2 with the read-level talker input (reports/mist_w3po_v1/block_merge_bound_v1.zh-CN.md §3.5,
# thinker_joint_crf_prereg_v1.zh-CN.md §1.7 / §8; coordinator 2026-11-11 ~22:15 Prague): speech chain only on T2's dev
# decode with each speaker's same-read segments merged (crf/merge_read_segments.py ->
# decode/dev_t2_latent_t0_s3000_w05_rl, made and checked on CPU): thinker_t_eval_orch --speech-from (talker c1+b1+s2,
# 2 TTS seeds, ASR, speech score, speech_latency_lit), gate_0 vs A' and gate_1 vs T2 itself (unmerged; = the merge's
# paired effect; text identical by construction). Eval-class job: stacks on non-training cards (--cap135, <= 135 GiB per
# card), never on a training card: it waits for T5's training exit (CRF_RUN_EXIT in its log) and excludes 4-7 (T4 trains
# there until ~01:25 Prague); the orchestrator re-reads <run>/exclude_cards every tick, so the cards can be widened later
# (write a new file + command mv -f). Scorer ports 18941-18944 (unused by the other T evals).
#   [EXCL="4 5 6 7"] [WAIT_T5=1] [DRY_RUN=1] bash experiments/mist_w3po_v1/scripts/start_t2rl_eval.sh
# Run detached: setsid nohup bash .../start_t2rl_eval.sh > logs/mist_w3po_v1/t_eval/start_t2rl_eval.log 2>&1 < /dev/null &
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
PY=/opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python
T5LOG=logs/mist_w3po_v1/t5_evid_p4_20261111T154210Z.log
RUN=logs/mist_w3po_v1/t_eval/t2rl
SRC=dev_t2_latent_t0_s3000_w05
A=dev_crf_v2p_s3000_finalpost_s8_c2g06
EXCL=${EXCL:-4 5 6 7}
log() { echo "$(date -u +%FT%TZ) $*"; }
CMD="$PY experiments/mist_w3po_v1/scripts/thinker_t_eval_orch.py --run $RUN --name ${SRC}__rl --ckpt - \
--speech-from ${SRC}_rl --talker c1b1s2 --window 0.5 --gate-ref $A ${A}_tts2 --gate-ref $SRC ${SRC}_tts2 \
--scorer-port0 18941 --cap135 --exclude $EXCL"
[ -s eval_outputs/mist_w3po_v1/decode/${SRC}_rl/hyp.jsonl ] && [ -s eval_outputs/mist_w3po_v1/decode/${SRC}_rl/score_text.json ] \
  || { log "merged decode ${SRC}_rl missing"; exit 1; }
if [ -n "${DRY_RUN:-}" ]; then
  echo "CMD: $CMD"
  $CMD --dry | tail -1
  echo "T5 exited: $(grep -c CRF_RUN_EXIT $T5LOG 2>/dev/null)"
  exit 0
fi
[ -e $RUN ] && { log "拒绝覆盖已有输出: $RUN"; exit 1; }
if [ "${WAIT_T5:-1}" = 1 ]; then
  log "waiting for T5's training exit ($T5LOG)"
  until grep -q CRF_RUN_EXIT $T5LOG 2>/dev/null; do sleep 30; done
  log "T5: $(grep CRF_RUN_EXIT $T5LOG | tail -1)"
fi
setsid nohup $CMD < /dev/null > $RUN.launch.log 2>&1 &
log "t2rl orchestrator pid $! (excluded: $EXCL)"
