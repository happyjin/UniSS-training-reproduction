#!/usr/bin/env bash
# Step2 v4: guided-CE + blankpen train → decode probe → commit/push.
# Isolated under simul_s2st_route_v1; does not touch Stage09-11 / Phase3 shipping.
set -euo pipefail

ROOT=/opt/dlami/nvme/jasonleeeli/projects/UniSS
USER_ROOT=/opt/dlami/nvme/jasonleeeli
PYTHON=$USER_ROOT/conda_envs/uniss-train/bin/python
RUN_NAME=${RUN_NAME:-step2_nar_ctc_15shard_v4_guided_ce}
DECODE_RUN=${DECODE_RUN:-step2_trained_nar_decode_v4_guided_ce}
TRAIN_TMUX=${TRAIN_TMUX:-step2_nar_ctc_v4}
CKPT_ROOT=$ROOT/checkpoints/simul_s2st_route_v1/${RUN_NAME}
TRAIN_LOG=$ROOT/logs/simul_s2st_route_v1/${RUN_NAME}.log
CHAIN_LOG=$ROOT/logs/simul_s2st_route_v1/${RUN_NAME}_chain.log
MASTER_PORT=${MASTER_PORT:-29814}

export PATH=$USER_ROOT/conda_envs/uniss-train/bin:$PATH
export PYTHONPATH=$ROOT/third_party/Megatron-LM:$ROOT:${PYTHONPATH:-}
export HF_HOME=$USER_ROOT/cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export GIT_TERMINAL_PROMPT=0

mkdir -p "$(dirname "$CHAIN_LOG")"
exec > >(tee -a "$CHAIN_LOG") 2>&1
log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

cd "$ROOT"
log "launch train ${RUN_NAME}"
tmux kill-session -t "$TRAIN_TMUX" 2>/dev/null || true
tmux new-session -d -s "$TRAIN_TMUX" \
  "cd $ROOT && RUN_NAME=$RUN_NAME BLANK_PENALTY=1.0 GUIDED_CE_WEIGHT=1.0 MASTER_PORT=$MASTER_PORT \
   bash experiments/simul_s2st_route_v1/step2_nar_ctc_head/run_15shard_8gpu.sh; echo TRAIN_EXIT=\$?; sleep 3600"

while true; do
  # Only the Megatron entrypoint counts as "training". The train tmux session
  # deliberately sleeps after exit, so do not gate on tmux liveness.
  alive=0
  pgrep -f "pretrain_nar_ctc_megatron.py.*${RUN_NAME}" >/dev/null 2>&1 && alive=1
  latest=""
  [[ -f "$CKPT_ROOT/latest_checkpointed_iteration.txt" ]] && latest=$(tr -d '[:space:]' <"$CKPT_ROOT/latest_checkpointed_iteration.txt")
  done_marker=0
  grep -q 'after training is done' "$TRAIN_LOG" 2>/dev/null && done_marker=1
  if [[ "$latest" == "3000" && "$done_marker" -eq 1 && "$alive" -eq 0 ]]; then
    log "train finished latest=${latest}"
    break
  fi
  # Fail fast if train process and tmux both gone before completion.
  tmux_alive=0
  tmux has-session -t "$TRAIN_TMUX" 2>/dev/null && tmux_alive=1
  if [[ "$alive" -eq 0 && "$tmux_alive" -eq 0 && "$done_marker" -eq 0 ]]; then
    log "ERROR: train died before completion (latest=${latest:-none})"
    exit 2
  fi
  log "waiting train alive=${alive} latest=${latest:-none} done=${done_marker}"
  sleep 60
done

sleep 15
DECODE_JSON=$ROOT/reports/simul_s2st_route_v1/${DECODE_RUN}.json
DECODE_MD=$ROOT/reports/simul_s2st_route_v1/${DECODE_RUN}.md
if [[ -e "$DECODE_JSON" || -e "$DECODE_MD" ]]; then
  log "decode already exists — skip"
else
  log "decode probe ${DECODE_RUN}"
  CUDA_VISIBLE_DEVICES=0 "$PYTHON" \
    "$ROOT/experiments/simul_s2st_route_v1/step2_nar_ctc_head/evaluate_trained_head.py" \
    --run-name "$DECODE_RUN" \
    --output-json "$DECODE_JSON" \
    --output-md "$DECODE_MD" \
    --checkpoint "iter1000=$CKPT_ROOT/iter_0001000" \
    --checkpoint "iter2000=$CKPT_ROOT/iter_0002000" \
    --checkpoint "iter3000=$CKPT_ROOT/iter_0003000" \
    --samples-per-direction 16
fi

log "commit + push"
git add \
  experiments/simul_s2st_route_v1/step2_nar_ctc_head/pretrain_nar_ctc_megatron.py \
  experiments/simul_s2st_route_v1/step2_nar_ctc_head/run_15shard_8gpu.sh \
  experiments/simul_s2st_route_v1/step2_nar_ctc_head/tests/test_guided_duration_ce.py \
  experiments/simul_s2st_route_v1/run_step2_v4_train_and_probe.sh \
  experiments/simul_s2st_route_v1/run_post_train_pipeline_v1.sh \
  docs/uniss_training_reproduction/simul_s2st_route_execution_report_step4_decision.md \
  "$DECODE_JSON" "$DECODE_MD" || true
if ! git diff --cached --quiet; then
  git commit -m "$(cat <<'EOF'
feat: Step2 v4 guided-CE train/probe and Step4 stay-on-A decision

Duration-stretched unit CE fights CTC blank collapse without editing
shipping Stage09-11 or Phase3 joint trainers.
EOF
)"
  GIT_TERMINAL_PROMPT=0 git push private HEAD:main
  log "pushed $(git rev-parse --short HEAD)"
else
  log "nothing new to commit after decode"
fi

echo CHAIN_OK >"$ROOT/logs/simul_s2st_route_v1/${RUN_NAME}_chain.ok"
log "v4 chain complete"
