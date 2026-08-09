#!/usr/bin/env bash
# Isolated post-train chain for simul_s2st_route_v1.
# Does not modify Stage09-11 / Phase3 joint shipping scripts.
set -euo pipefail

ROOT=/opt/dlami/nvme/jasonleeeli/projects/UniSS
USER_ROOT=/opt/dlami/nvme/jasonleeeli
PYTHON=$USER_ROOT/conda_envs/uniss-train/bin/python
TRAIN_NAME=${TRAIN_NAME:-step2_nar_ctc_15shard_v3_blankpen}
TRAIN_LOG=$ROOT/logs/simul_s2st_route_v1/${TRAIN_NAME}.log
TRAIN_TMUX=${TRAIN_TMUX:-step2_nar_ctc_15shard}
CKPT_ROOT=$ROOT/checkpoints/simul_s2st_route_v1/${TRAIN_NAME}
PIPELINE_LOG=$ROOT/logs/simul_s2st_route_v1/${TRAIN_NAME}_post_pipeline.log
DECODE_RUN=${DECODE_RUN:-step2_trained_nar_decode_v3_blankpen}
PARETO_RUN=${PARETO_RUN:-step3_ar_pareto_smoke8_v2}
REPORT=$ROOT/docs/uniss_training_reproduction/simul_s2st_route_execution_report_step2_v3_and_step3_smoke.md

export PATH=$USER_ROOT/conda_envs/uniss-train/bin:$PATH
export PYTHONPATH=$ROOT/third_party/Megatron-LM:$ROOT:${PYTHONPATH:-}
export HF_HOME=$USER_ROOT/cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

mkdir -p "$(dirname "$PIPELINE_LOG")"
exec > >(tee -a "$PIPELINE_LOG") 2>&1

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

log "pipeline start: wait for train=${TRAIN_NAME}"

# 1) Wait for training to finish (tmux gone + final checkpoint + log marker).
while true; do
  train_alive=0
  tmux has-session -t "$TRAIN_TMUX" 2>/dev/null && train_alive=1
  pgrep -f "pretrain_nar_ctc_megatron.py.*${TRAIN_NAME}" >/dev/null 2>&1 && train_alive=1
  latest=""
  [[ -f "$CKPT_ROOT/latest_checkpointed_iteration.txt" ]] && latest=$(tr -d '[:space:]' <"$CKPT_ROOT/latest_checkpointed_iteration.txt")
  done_marker=0
  grep -q 'after training is done' "$TRAIN_LOG" 2>/dev/null && done_marker=1
  if [[ "$train_alive" -eq 0 && "$latest" == "3000" && "$done_marker" -eq 1 ]]; then
    log "train finished: latest=${latest}"
    break
  fi
  log "waiting train_alive=${train_alive} latest=${latest:-none} done_marker=${done_marker}"
  sleep 30
done

# Give NCCL teardown a moment so GPUs are free.
sleep 15
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader || true

# 2) Unit decode probe on v3 checkpoints (isolated; new report name).
DECODE_JSON=$ROOT/reports/simul_s2st_route_v1/${DECODE_RUN}.json
DECODE_MD=$ROOT/reports/simul_s2st_route_v1/${DECODE_RUN}.md
if [[ -e "$DECODE_JSON" || -e "$DECODE_MD" ]]; then
  log "decode outputs already exist — skipping probe"
else
  log "starting decode probe ${DECODE_RUN}"
  CUDA_VISIBLE_DEVICES=0 "$PYTHON" \
    "$ROOT/experiments/simul_s2st_route_v1/step2_nar_ctc_head/evaluate_trained_head.py" \
    --run-name "$DECODE_RUN" \
    --output-json "$DECODE_JSON" \
    --output-md "$DECODE_MD" \
    --checkpoint "iter1000=$CKPT_ROOT/iter_0001000" \
    --checkpoint "iter2000=$CKPT_ROOT/iter_0002000" \
    --checkpoint "iter3000=$CKPT_ROOT/iter_0003000" \
    --samples-per-direction 16
  log "decode probe done"
fi

# 3) Step3 AR Pareto on 8 GPUs (import-only Stage11 wrapper).
PARETO_JSON=$ROOT/reports/simul_s2st_route_v1/${PARETO_RUN}.json
PARETO_MD=$ROOT/reports/simul_s2st_route_v1/${PARETO_RUN}.md
if [[ -e "$PARETO_JSON" || -e "$PARETO_MD" ]]; then
  log "pareto outputs already exist — skipping"
else
  log "starting Step3 AR Pareto ${PARETO_RUN}"
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  RUN_NAME="$PARETO_RUN" MAX_SAMPLES=8 MASTER_PORT=29832 \
    bash "$ROOT/experiments/simul_s2st_route_v1/step3_waitk_pareto/run_ar_pareto_8gpu.sh"
  log "pareto done"
fi

# 4) Write a short combined report (new file only).
if [[ ! -e "$REPORT" ]]; then
  log "writing combined report"
  "$PYTHON" - <<'PY'
import json
from pathlib import Path
from datetime import datetime, timezone

root = Path("/opt/dlami/nvme/jasonleeeli/projects/UniSS")
decode = json.loads((root / "reports/simul_s2st_route_v1/step2_trained_nar_decode_v3_blankpen.json").read_text())
pareto_path = root / "reports/simul_s2st_route_v1/step3_ar_pareto_smoke8_v1.json"
pareto = json.loads(pareto_path.read_text()) if pareto_path.exists() else None
train_log = (root / "logs/simul_s2st_route_v1/step2_nar_ctc_15shard_v3_blankpen.log").read_text(errors="ignore")
import re
valid = re.findall(r"validation loss at iteration 3000 on validation set \| nar_ctc value: ([0-9.E+-]+).*?nar_blank_mass value: ([0-9.E+-]+)", train_log)
valid_ctc, valid_blank = (valid[-1] if valid else ("?", "?"))

lines = [
    "# Simul-S2ST route — Step2 v3 blankpen + Step3 AR Pareto smoke",
    "",
    f"> Auto-generated {datetime.now(timezone.utc).isoformat()}",
    "",
    "## Step2 v3 blankpen train",
    "",
    "- Data: 15-shard joint (`pilot_15shard_joint`), mbs=64 / gbs=512, blank_penalty=1.0",
    f"- Final valid `nar_ctc` / `nar_blank_mass`: {valid_ctc} / {valid_blank}",
    "- Checkpoint: `checkpoints/simul_s2st_route_v1/step2_nar_ctc_15shard_v3_blankpen/iter_0003000`",
    "",
    "## Decode probe",
    "",
    "| Ckpt | UER | Empty | Blank frames | Blank-sup UER | Distinct (blank-sup) |",
    "|---|---:|---:|---:|---:|---:|",
]
for entry in decode["results"]:
    p = entry["pooled"]
    lines.append(
        f"| `{entry['label']}` | {p['unit_error_rate']*100:.1f}% | {p['empty_predictions']}/32 | "
        f"{p['mean_blank_fraction']*100:.1f}% | {p['blank_suppressed_unit_error_rate']*100:.1f}% | "
        f"{p['mean_blank_suppressed_distinct']:.1f} |"
    )
lines += ["", "## Step3 AR Pareto smoke", ""]
if pareto:
    lines += [
        "| k | Λ window | BLEU | chrF | First WRITE ms | Fallback | RTF |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for b in pareto["summary"]:
        lines.append(
            f"| {b['lagging_k']} | {b['lambda_window']} | {b['mean_text_bleu']:.2f} | "
            f"{b['mean_chrf']:.2f} | {b['mean_first_write_ms']:.0f} | "
            f"{b['fallback_rate']*100:.0f}% | {b['mean_compute_rtf']:.2f} |"
        )
else:
    lines.append("_Pareto report missing._")
lines.append("")
out = root / "docs/uniss_training_reproduction/simul_s2st_route_execution_report_step2_v3_and_step3_smoke.md"
out.write_text("\n".join(lines), encoding="utf-8")
print("wrote", out)
PY
fi

# 5) Commit + push only route artefacts (never touch unrelated dirty files).
log "git commit + push"
cd "$ROOT"
git add \
  experiments/simul_s2st_route_v1/run_post_train_pipeline_v1.sh \
  reports/simul_s2st_route_v1/${DECODE_RUN}.json \
  reports/simul_s2st_route_v1/${DECODE_RUN}.md \
  reports/simul_s2st_route_v1/${PARETO_RUN}.json \
  reports/simul_s2st_route_v1/${PARETO_RUN}.md \
  docs/uniss_training_reproduction/simul_s2st_route_execution_report_step2_v3_and_step3_smoke.md \
  || true
if git diff --cached --quiet; then
  log "nothing to commit"
else
  git commit -m "$(cat <<'EOF'
docs: auto-chain Step2 v3 decode and Step3 AR Pareto smoke

Post-train pipeline under simul_s2st_route_v1 only; no shipping Stage09-11
or Phase3 joint scripts were modified.
EOF
)"
  git push private HEAD:main
  log "pushed $(git rev-parse --short HEAD)"
fi

log "pipeline complete"
echo PIPELINE_OK >"$ROOT/logs/simul_s2st_route_v1/${TRAIN_NAME}_post_pipeline.ok"
