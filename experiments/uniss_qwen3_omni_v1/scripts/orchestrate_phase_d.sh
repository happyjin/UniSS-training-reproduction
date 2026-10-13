#!/usr/bin/env bash
# Phase D, unattended: train, probe at the planned iterations, extend if the
# curve has not flattened, and stop when it has.
#
# A probe cannot share a card with training -- training reserves 92 GiB and
# the probe wants 62, against 139.8 -- so each measurement is a stop, a
# probe, and a resume. Stopping right after a save costs nothing, which is
# why the orchestrator waits for the checkpoint rather than for the step.
#
# What it decides, and on what:
#   * probes write ASR-BLEU per direction to probes.tsv
#   * at the last planned iteration, if the mean gain over the previous
#     probe is still above GAIN_STOP, TRAIN_ITERS grows by EXTEND_BY and a
#     probe is scheduled at the new end
#   * it never grows past MAX_ITERS, and never extends on a *fall* -- a
#     drop is a reason to stop and look, not to train longer
#
# 128 utterances carry roughly +/-3 BLEU of noise, so this gauge is a
# tripwire and a shape, not a checkpoint selector. Selection happens after,
# on a random 1,000, and the final number on the full split.
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"; cd "${REPO_ROOT}"

RUN_DIR="${RUN_DIR:?set RUN_DIR to the checkpoint directory}"
TRAIN_ITERS="${TRAIN_ITERS:-10000}"
PROBE_AT="${PROBE_AT:-5000 7000}"       # plus TRAIN_ITERS, always
MAX_ITERS="${MAX_ITERS:-20000}"
EXTEND_BY="${EXTEND_BY:-4000}"
GAIN_STOP="${GAIN_STOP:-2.0}"           # mean BLEU gain below this = flat
MBS="${MBS:-128}"; GBS="${GBS:-1024}"
LR="${LR:-6e-5}"; MIN_LR="${MIN_LR:-2e-6}"
STATE="${RUN_DIR}/orchestrator"
mkdir -p "${STATE}"
TSV="${STATE}/probes.tsv"
[ -f "${TSV}" ] || printf "iter\tzh2en\ten2zh\tmean\n" > "${TSV}"
log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "${STATE}/orchestrator.log"; }

latest_iter() { cat "${RUN_DIR}/latest_checkpointed_iteration.txt" 2>/dev/null || echo 0; }
training_up() { pgrep -f "pretrain_phase[_]d" >/dev/null; }

start_training() {
  local iters="$1" extra=""
  [ -f "${RUN_DIR}/latest_checkpointed_iteration.txt" ] && extra="--override-opt-param-scheduler"
  log "启动训练 -> ${iters} 步"
  OVERRIDE_SCHED="${extra}" MBS="${MBS}" GBS="${GBS}" TRAIN_ITERS="${iters}" \
    WARMUP_ITERS=200 LR="${LR}" MIN_LR="${MIN_LR}" NUM_WORKERS=4 \
    EVAL_INTERVAL=500 EVAL_ITERS=4 SAVE_INTERVAL=1000 LOG_INTERVAL=20 \
    MASTER_PORT="${MASTER_PORT:-29657}" \
    SAVE_DIR="${RUN_DIR}" TB_DIR="${REPO_ROOT}/runs/uniss_qwen3_omni_v1/$(basename "${RUN_DIR}")" \
    setsid nohup bash experiments/uniss_qwen3_omni_v1/scripts/run_phase_d.sh \
      >> "${STATE}/train.log" 2>&1 < /dev/null &
  disown
  sleep 60
}

stop_training() {
  log "停止训练"
  pkill -f "pretrain_phase[_]d" 2>/dev/null
  local w=0
  while training_up && [ $w -lt 40 ]; do sleep 5; w=$((w+1)); done
  sleep 20
}

run_probe() {
  local it="$1" tag ckpt out
  tag=$(printf "%07d" "${it}")
  ckpt="${RUN_DIR}/exported/phase_d_iter${tag}.pt"
  out="${REPO_ROOT}/eval_outputs/uniss_qwen3_omni_v1/probe_iter${it}"
  log "导出 iter ${it}"
  PYTHONPATH=.:third_party/Megatron-LM CUDA_VISIBLE_DEVICES="" \
    /opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval/bin/python \
    -m experiments.uniss_qwen3_omni_v1.training.export_phase_d \
    --checkpoint-dir "${RUN_DIR}/iter_${tag}" --output "${ckpt}" \
    >> "${STATE}/export.log" 2>&1 || { log "导出失败 iter ${it}"; return 1; }
  log "探针 iter ${it}"
  STAGGER=12 PHASE_D_CKPT="${ckpt}" \
    TALKER="${REPO_ROOT}/checkpoints/uniss_omni_s2st_v1/stage1_transfer/exported/talker_iter0040000.pt" \
    OUT="${out}" LIMIT=128 \
    bash experiments/uniss_qwen3_omni_v1/scripts/run_probe.sh >> "${STATE}/probe.log" 2>&1
  local line
  line=$(/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval/bin/python - "${out}/asr_bleu.json" "${it}" <<'PY'
import json, sys
try:
    g = json.load(open(sys.argv[1]))["groups"]
except Exception:
    sys.exit(1)
z = next((v["score"] for k, v in g.items() if "cmn->eng" in k), None)
e = next((v["score"] for k, v in g.items() if "eng->cmn" in k), None)
if z is None or e is None:
    sys.exit(1)
print(f"{sys.argv[2]}\t{z:.2f}\t{e:.2f}\t{(z+e)/2:.2f}")
PY
) || { log "打分失败 iter ${it}"; return 1; }
  echo "${line}" >> "${TSV}"
  log "iter ${it}: ${line}"
}

# The gain between the last two probe rows, mean over directions.
last_gain() {
  /opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval/bin/python - "${TSV}" <<'PY'
import sys
rows = [l.split("\t") for l in open(sys.argv[1]).read().splitlines()[1:] if l.strip()]
print(f"{float(rows[-1][3]) - float(rows[-2][3]):.2f}" if len(rows) >= 2 else "999")
PY
}

targets=""
for t in ${PROBE_AT}; do [ "${t}" -lt "${TRAIN_ITERS}" ] && targets="${targets} ${t}"; done
targets="${targets} ${TRAIN_ITERS}"
log "计划探针点:${targets}  上限 ${MAX_ITERS} 步"

training_up || start_training "${TRAIN_ITERS}"

while :; do
  next=""
  for t in ${targets}; do
    grep -qP "^${t}\t" "${TSV}" || { next="${t}"; break; }
  done
  [ -z "${next}" ] && { log "全部探针点已完成"; break; }

  log "等待 checkpoint >= ${next}（当前 $(latest_iter)）"
  while [ "$(latest_iter)" -lt "${next}" ]; do
    training_up || { log "训练意外退出，重启"; start_training "${TRAIN_ITERS}"; }
    sleep 60
  done

  stop_training
  run_probe "${next}" || log "iter ${next} 探针未完成，继续"

  if [ "${next}" = "${TRAIN_ITERS}" ]; then
    gain=$(last_gain)
    log "最后一个探针点，较上一次增益 ${gain}"
    flat=$(/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval/bin/python -c "
print('1' if float('${gain}') < float('${GAIN_STOP}') else '0')")
    if [ "${flat}" = "1" ] || [ "${TRAIN_ITERS}" -ge "${MAX_ITERS}" ]; then
      log "曲线已平或已达上限，训练结束于 ${TRAIN_ITERS} 步"
      break
    fi
    TRAIN_ITERS=$((TRAIN_ITERS + EXTEND_BY))
    [ "${TRAIN_ITERS}" -gt "${MAX_ITERS}" ] && TRAIN_ITERS="${MAX_ITERS}"
    targets="${targets} ${TRAIN_ITERS}"
    log "仍在上升，延长到 ${TRAIN_ITERS} 步"
  fi
  start_training "${TRAIN_ITERS}"
done

log "ORCHESTRATOR_DONE"
cat "${TSV}"
