#!/usr/bin/env bash
# Step2 v5: CE-dominant loss (ctc_weight=0.25, guided_ce=5, blank_pen=2) → decode.
set -euo pipefail

ROOT=/opt/dlami/nvme/jasonleeeli/projects/UniSS
USER_ROOT=/opt/dlami/nvme/jasonleeeli
PYTHON=$USER_ROOT/conda_envs/uniss-train/bin/python
RUN_NAME=${RUN_NAME:-step2_nar_ctc_15shard_v5_guided_ce_dominant}
DECODE_RUN=${DECODE_RUN:-step2_trained_nar_decode_v5_guided_ce_dominant}
TRAIN_TMUX=${TRAIN_TMUX:-step2_nar_ctc_v5}
CKPT_ROOT=$ROOT/checkpoints/simul_s2st_route_v1/${RUN_NAME}
TRAIN_LOG=$ROOT/logs/simul_s2st_route_v1/${RUN_NAME}.log
CHAIN_LOG=$ROOT/logs/simul_s2st_route_v1/${RUN_NAME}_chain.log
MASTER_PORT=${MASTER_PORT:-29815}

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
  "cd $ROOT && \
   RUN_NAME=$RUN_NAME \
   BLANK_PENALTY=2.0 GUIDED_CE_WEIGHT=5.0 CTC_WEIGHT=0.25 \
   BASE_LR=5e-4 MIN_LR=5e-4 LR_WARMUP_ITERS=200 \
   LR_DECAY_STYLE=constant MASTER_PORT=$MASTER_PORT \
   bash experiments/simul_s2st_route_v1/step2_nar_ctc_head/run_15shard_8gpu.sh; \
   echo TRAIN_EXIT=\$?; sleep 3600"

while true; do
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

# Write short report
"$PYTHON" - <<PY
from pathlib import Path
from datetime import datetime, timezone
import json, re
root = Path("$ROOT")
decode = json.loads((root / "reports/simul_s2st_route_v1/${DECODE_RUN}.json").read_text())
logt = (root / "logs/simul_s2st_route_v1/${RUN_NAME}.log").read_text(errors="ignore")
m = re.findall(
    r"validation loss at iteration 3000 on validation set \\| nar_ctc value: ([0-9.E+-]+).*?nar_blank_mass value: ([0-9.E+-]+).*?nar_guided_ce value: ([0-9.E+-]+)",
    logt,
)
ctc, blank, guided = m[-1] if m else ("?", "?", "?")
lines = [
    "# Simul-S2ST route — Step2 v5 CE-dominant result",
    "",
    f"> {datetime.now(timezone.utc).isoformat()}",
    "",
    "- Settings: ctc_weight=0.25, guided_ce=5.0, blank_penalty=2.0, lr=5e-4 constant",
    f"- Final valid nar_ctc / blank_mass / guided_ce: `{ctc}` / `{blank}` / `{guided}`",
    "",
    "| Ckpt | UER | Empty | Blank frames | Blank-sup UER | Distinct (blank-sup) |",
    "|---|---:|---:|---:|---:|---:|",
]
for e in decode["results"]:
    p = e["pooled"]
    lines.append(
        f"| `{e['label']}` | {p['unit_error_rate']*100:.1f}% | {p['empty_predictions']}/32 | "
        f"{p['mean_blank_fraction']*100:.1f}% | {p['blank_suppressed_unit_error_rate']*100:.1f}% | "
        f"{p['mean_blank_suppressed_distinct']:.1f} |"
    )
lines.append("")
out = root / "docs/uniss_training_reproduction/simul_s2st_route_execution_report_step2_v5_guided_ce.md"
out.write_text("\\n".join(lines), encoding="utf-8")
print("wrote", out)
PY

log "commit + push"
git add \
  experiments/simul_s2st_route_v1/step2_nar_ctc_head/pretrain_nar_ctc_megatron.py \
  experiments/simul_s2st_route_v1/step2_nar_ctc_head/run_15shard_8gpu.sh \
  experiments/simul_s2st_route_v1/step2_nar_ctc_head/overfit_guided_ce.py \
  experiments/simul_s2st_route_v1/run_step2_v5_train_and_probe.sh \
  reports/simul_s2st_route_v1/${DECODE_RUN}.json \
  reports/simul_s2st_route_v1/${DECODE_RUN}.md \
  docs/uniss_training_reproduction/simul_s2st_route_execution_report_step2_v5_guided_ce.md \
  || true
if ! git diff --cached --quiet; then
  git commit -m "$(cat <<'EOF'
feat: Step2 v5 CE-dominant NAR train/probe against blank collapse

Lower CTC weight and raise guided CE so frame-level unit peaks can form
at 15-shard scale; defaults keep prior runs unchanged.
EOF
)"
  GIT_TERMINAL_PROMPT=0 git push private HEAD:main
  log "pushed $(git rev-parse --short HEAD)"
else
  log "nothing to commit"
fi

echo CHAIN_OK >"$ROOT/logs/simul_s2st_route_v1/${RUN_NAME}_chain.ok"
log "v5 chain complete"
