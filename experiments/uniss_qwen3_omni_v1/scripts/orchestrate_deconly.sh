#!/usr/bin/env bash
# Phase E, unattended: train, probe end to end at the planned iterations,
# extend while the curve climbs, stop when it flattens.
#
# Same shape as Phase D's orchestrator and the same reason for the stop
# and resume: training reserves 97 GiB a card and a probe wants 62, so a
# measurement cannot share the GPUs. Stopping right after a save costs
# nothing, which is why it waits for the checkpoint and not the step.
#
# What is different is the probe. Phase D's was TTS mode -- reference text
# in -- because the question was whether the Talker could read Qwen3 at
# all. This one is end to end, because Phase E's question is whether it
# can read Qwen3 on the states a real translation produces, and the line
# to beat is Phase B's cascade at 27.68 and 34.14.
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"; cd "${REPO_ROOT}"

RUN_DIR="${RUN_DIR:?}"
TRAIN_ITERS="${TRAIN_ITERS:-15000}"
PROBE_AT="${PROBE_AT:-4000 8000 12000}"
MAX_ITERS="${MAX_ITERS:-25000}"
EXTEND_BY="${EXTEND_BY:-5000}"
GAIN_STOP="${GAIN_STOP:-1.5}"
MBS="${MBS:-32}"; GBS="${GBS:-512}"
LR="${LR:-6e-5}"; MIN_LR="${MIN_LR:-2e-6}"
PROBE_LIMIT="${PROBE_LIMIT:-64}"
# Every probe point needs a checkpoint sitting exactly on it: run_probe
# exports iter_<target>, and a save interval coarser than the probe
# spacing leaves it exporting a directory that was never written.
SAVE_INTERVAL="${SAVE_INTERVAL:-1000}"
RUN_SCRIPT="${RUN_SCRIPT:-experiments/uniss_qwen3_omni_v1/scripts/run_deconly.sh}"
WARMUP_ITERS="${WARMUP_ITERS:-200}"
# Checking a config should not cost eight GPUs. start_training setsids and
# disowns, so a timeout or a Ctrl-C on this script leaves the run it
# already launched alive -- there is no way to "just see what it would
# do" without this.
DRY_RUN="${DRY_RUN:-0}"
# The exporter refuses an adapter-free checkpoint unless told the run was
# meant to be one, because that is also what a checkpoint that never
# trained looks like.
NO_LORA=""
[ "${UNISS_DECONLY_LORA_FROM:-}" = "none" ] && NO_LORA="--no-lora"

# Checked before anything starts, because neither failure is legible once
# it happens. A probe point off the save grid sends the exporter at a
# directory that was never written; a first segment shorter than the
# warmup trips assert lr_warmup_steps < lr_decay_steps inside the restart
# loop, so the orchestrator retries it forever.
# The final target is probed and the extensions land on the grid too, so
# they are held to it as well as PROBE_AT.
for t in ${PROBE_AT} "${TRAIN_ITERS}" "${EXTEND_BY}"; do
  if [ $(( t % SAVE_INTERVAL )) -ne 0 ]; then
    echo "${t}（PROBE_AT / TRAIN_ITERS / EXTEND_BY）不能被 SAVE_INTERVAL=${SAVE_INTERVAL} 整除；" >&2
    echo "对应的 iter 永远不会被写出，导出会失败。" >&2
    exit 1
  fi
done
# Every launch is given the whole horizon -- start_training "${TRAIN_ITERS}"
# -- and a probe point is reached by stopping the run, not by ending it. So
# Megatron's lr_warmup_steps < lr_decay_steps compares the horizon with the
# warmup, and an early probe point is fine. (This guard once compared the
# first probe point instead, carried over from a run whose horizon really
# was 100, and refused a 100-step probe that would have worked.)
if [ "${TRAIN_ITERS}" -le "${WARMUP_ITERS}" ]; then
  echo "TRAIN_ITERS=${TRAIN_ITERS} 不大于 WARMUP_ITERS=${WARMUP_ITERS}；" >&2
  echo "Megatron 要求 lr_warmup_steps < lr_decay_steps，会崩溃重启循环。" >&2
  exit 1
fi

STATE="${RUN_DIR}/orchestrator"; mkdir -p "${STATE}"
TSV="${STATE}/probes.tsv"
[ -f "${TSV}" ] || printf "iter\tzh2en\ten2zh\tmean\ttext_zh2en\ttext_en2zh\n" > "${TSV}"
log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "${STATE}/orchestrator.log"; }

latest_iter() { cat "${RUN_DIR}/latest_checkpointed_iteration.txt" 2>/dev/null || echo 0; }
# Training is a python process with pretrain_deconly.py among its
# arguments. Matching the bare string also matched any shell whose command
# line mentioned it -- a grep of the logs, or the very command that ran a
# DRY_RUN of this script -- and a false "up" leaves the loop waiting for a
# checkpoint nothing is writing; stop_training would have killed those
# shells too.
deconly_pids() {
  ps -eo pid=,args= 2>/dev/null \
    | awk '$2 ~ /\/python[0-9.]*$/ && / experiments\/uniss_qwen3_omni_v1\/training\/pretrain_deconly\.py( |$)/ {print $1}'
}
training_up() { [ -n "$(deconly_pids)" ]; }

start_training() {
  local iters="$1" sched=""
  [ -f "${RUN_DIR}/latest_checkpointed_iteration.txt" ] && sched="--override-opt-param-scheduler"
  log "启动训练 -> ${iters} 步"
  if [ "${DRY_RUN}" = "1" ]; then
    log "DRY_RUN=1，不启动。配置检查通过。"
    exit 0
  fi
  OVERRIDE_SCHED="${sched}" MBS="${MBS}" GBS="${GBS}" \
    TRAIN_ITERS="${iters}" WARMUP_ITERS="${WARMUP_ITERS}" LR="${LR}" MIN_LR="${MIN_LR}" \
    NUM_WORKERS=6 EVAL_INTERVAL=500 EVAL_ITERS=4 \
    SAVE_INTERVAL="${SAVE_INTERVAL}" \
    LOG_INTERVAL=20 MASTER_PORT="${MASTER_PORT:-29705}" \
    SAVE_DIR="${RUN_DIR}" TB_DIR="${REPO_ROOT}/runs/uniss_qwen3_omni_v1/$(basename "${RUN_DIR}")" \
    setsid nohup bash "${RUN_SCRIPT}" \
      >> "${STATE}/train.log" 2>&1 < /dev/null &
  disown
  sleep 90
}

stop_training() {
  log "停止训练"
  deconly_pids | while read p; do kill "$p" 2>/dev/null; done
  local w=0
  while training_up && [ $w -lt 40 ]; do sleep 5; w=$((w+1)); done
  sleep 25
}

run_probe() {
  local it="$1" tag ckpt out
  tag=$(printf "%07d" "${it}")
  ckpt="${RUN_DIR}/exported/phase_e_iter${tag}.pt"
  # Namespaced by run: run_e2e_probe rm -rf's its output directory, so a
  # bare e2e_iter${it} let v2's probe delete v1's outputs at the same
  # iteration. The scores survived in probes.tsv; the waveforms did not.
  out="${REPO_ROOT}/eval_outputs/uniss_qwen3_omni_v1/$(basename "${RUN_DIR}")/e2e_iter${it}"
  mkdir -p "$(dirname "${out}")"
  # A retry gets its own directory. The probe refuses to write into a
  # non-empty one now -- it no longer deletes -- so a second attempt
  # must not aim at the first attempt's leftovers.
  att=$(cat "${STATE}/fail_${it}" 2>/dev/null || echo 0)
  [ "${att}" != "0" ] && out="${out}_r${att}"
  log "导出 iter ${it}"
  # export_deconly, not export_phase_d: there is no talker here and no
  # markers, and the two-stream prefixes would match nothing and leave an
  # empty payload.
  PYTHONPATH=.:third_party/Megatron-LM CUDA_VISIBLE_DEVICES="" \
    /opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval/bin/python \
    -m experiments.uniss_qwen3_omni_v1.training.export_deconly \
    --checkpoint-dir "${RUN_DIR}/iter_${tag}" --output "${ckpt}" ${NO_LORA} \
    >> "${STATE}/export.log" 2>&1 || { log "导出失败 ${it}"; return 1; }
  log "端到端探针 iter ${it}"
  # No TALKER and no SINGLE_PASS: the thinker writes the reply and then
  # keeps going into codes, so there is no second module to load and no
  # choice of how many passes to make.
  CKPT="${ckpt}" OUT="${out}" LIMIT="${PROBE_LIMIT}" \
    bash experiments/uniss_qwen3_omni_v1/scripts/run_deconly_probe.sh \
    >> "${STATE}/probe.log" 2>&1
  local line
  line=$(/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval/bin/python - "${out}" "${it}" <<'PY'
import json, sys
def pair(path):
    try: g = json.load(open(path))["groups"]
    except Exception: return None, None
    return (next((v["score"] for k, v in g.items() if "cmn->eng" in k), None),
            next((v["score"] for k, v in g.items() if "eng->cmn" in k), None))
z, e = pair(f"{sys.argv[1]}/asr_bleu.json")
if z is None or e is None: sys.exit(1)
tz, te = pair(f"{sys.argv[1]}/text_bleu.json")
f = lambda v: "NA" if v is None else f"{v:.2f}"
print(f"{sys.argv[2]}\t{z:.2f}\t{e:.2f}\t{(z+e)/2:.2f}\t{f(tz)}\t{f(te)}")
PY
) || { log "打分失败 ${it}"; return 1; }
  echo "${line}" >> "${TSV}"
  log "iter ${it}: ${line}   （对照 Thinker–Talker 24.96 @14k，级联 31.08，论文 Dec-only 27.12）"
}

last_gain() {
  /opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval/bin/python - "${TSV}" <<'PY'
import sys
rows=[l.split("\t") for l in open(sys.argv[1]).read().splitlines()[1:] if l.strip()]
rows=[r for r in rows if r[3]!="NA" and "-single" not in r[0]]
print(f"{float(rows[-1][3])-float(rows[-2][3]):.2f}" if len(rows)>=2 else ("999" if rows else "NA"))
PY
}

targets=""
for t in ${PROBE_AT}; do [ "${t}" -lt "${TRAIN_ITERS}" ] && targets="${targets} ${t}"; done
targets="${targets} ${TRAIN_ITERS}"
log "计划探针点:${targets}  上限 ${MAX_ITERS}"
training_up || start_training "${TRAIN_ITERS}"

while :; do
  next=""
  for t in ${targets}; do grep -qP "^${t}\t" "${TSV}" || { next="${t}"; break; }; done
  [ -z "${next}" ] && { log "全部探针点完成"; break; }
  log "等待 checkpoint >= ${next}（当前 $(latest_iter)）"
  while [ "$(latest_iter)" -lt "${next}" ]; do
    training_up || { log "训练意外退出，重启"; start_training "${TRAIN_ITERS}"; }
    sleep 60
  done
  stop_training
  if ! run_probe "${next}"; then
    # Without this the loop finds the same target again, sees a
    # checkpoint already past it, and stops training on the spot --
    # start, stop, fail, forever, with the run burning in between.
    # Two attempts, then write a sentinel so the target is retired and
    # training gets to keep going.
    n=$(cat "${STATE}/fail_${next}" 2>/dev/null || echo 0); n=$((n+1))
    echo "${n}" > "${STATE}/fail_${next}"
    log "iter ${next} 探针未完成（第 ${n} 次）"
    if [ "${n}" -ge 2 ]; then
      printf "%s\tNA\tNA\tNA\tNA\tNA\n" "${next}" >> "${TSV}"
      log "iter ${next} 放弃探针，继续训练"
    fi
  fi
  if [ "${next}" = "${TRAIN_ITERS}" ]; then
    gain=$(last_gain); log "最后一点，较上次增益 ${gain}"
    [ "${gain}" = "NA" ] && { log "最后一点无有效分数，不延长"; break; }
    flat=$(/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval/bin/python -c "
print('1' if float('${gain}') < float('${GAIN_STOP}') else '0')")
    if [ "${flat}" = "1" ] || [ "${TRAIN_ITERS}" -ge "${MAX_ITERS}" ]; then
      log "曲线已平或达上限，结束于 ${TRAIN_ITERS} 步"; break
    fi
    TRAIN_ITERS=$((TRAIN_ITERS + EXTEND_BY))
    [ "${TRAIN_ITERS}" -gt "${MAX_ITERS}" ] && TRAIN_ITERS="${MAX_ITERS}"
    targets="${targets} ${TRAIN_ITERS}"; log "仍在上升，延长到 ${TRAIN_ITERS}"
  fi
  start_training "${TRAIN_ITERS}"
done
log "ORCHESTRATOR_DONE"; cat "${TSV}"
