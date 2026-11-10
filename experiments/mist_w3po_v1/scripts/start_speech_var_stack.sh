#!/usr/bin/env bash
# A speech_var_orch.py run stacked next to another running launcher (2026-11-10 ~15:55 UTC, user: use the free GPU
# memory next to C2). Unlike start_speech_var.sh it does NOT stop the workload, arm idle fill or restart the workload
# (the running launcher owns those); it starts its OWN CTC scorers (ports SCORER_PORT0.., cards SCORER_CARDS) so they
# outlive nothing else's run, launches the orchestrator, and stops its scorers by PID when the orchestrator exits.
# Checked with bash -n only; never test-run.
#   RUN=logs/mist_w3po_v1/speech_var_b3 VARS="c1+b1k4@<base>,c1+b1+s2@<base>" \
#     setsid nohup bash experiments/mist_w3po_v1/scripts/start_speech_var_stack.sh > logs/.../start_<UTC>.out 2>&1 < /dev/null &
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
J=/opt/dlami/nvme/jasonleeeli
RUN=${RUN:?set RUN}
VARS=${VARS:?set VARS}
SCORER_CARDS=${SCORER_CARDS:-"1 3 5 7"}
SCORER_PORT0=${SCORER_PORT0:-18769}
PY=$J/conda_envs/qwen3omni/bin/python
LOCK=$J/work/mist_w3po_v1/gpu_launch.lock
ENVS="export TMPDIR=$J/work/mist_w3po_v1/tmp HF_HOME=$J/cache/huggingface TORCH_HOME=$J/cache/torch XDG_CACHE_HOME=$J/cache PYTHONPATH=.:third_party/Megatron-LM OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false"
D=logs/mist_w3po_v1/speech_var_stack_at_$(date -u +%Y%m%dT%H%M%SZ)
if [ -e "$D" ] || [ -e "$RUN" ]; then echo "refusing: $D or $RUN exists"; exit 1; fi
mkdir -p "$D"; echo $$ > "$D/launcher.pid"
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$D/launcher.log"; }

NS=0; SPIDS=""
for c in $SCORER_CARDS; do
  port=$((SCORER_PORT0 + NS))
  flock -o "$LOCK" -c "$ENVS; CUDA_VISIBLE_DEVICES=$c setsid nohup $PY -m experiments.mist_w3po_v1.evaluation.si_ctc_scorer --port $port > $D/scorer$NS.log 2>&1 < /dev/null & echo \$! > $D/scorer$NS.pid" 9>&-
  SPIDS="$SPIDS $(cat $D/scorer$NS.pid)"; NS=$((NS + 1)); sleep 5
done
for i in $(seq 1 60); do
  r=0; for k in $(seq 0 $((NS - 1))); do grep -q SCORER_READY $D/scorer$k.log 2>/dev/null && r=$((r+1)); done
  [ $r = $NS ] && break; sleep 5
done
log "scorers ready $r/$NS ports $SCORER_PORT0.. pids [$SPIDS]"
[ "$r" = "$NS" ] || { log "SCORER_FAILED; stop scorers, not launching"; kill -TERM $SPIDS 2>/dev/null; exit 1; }

# ORCH_EXTRA (default empty = unchanged): extra speech_var_orch args, e.g. "--win <C2 windows>" for a C2 base
# ORCH (default empty = speech_var_orch.py with --vars, unchanged): another orchestrator taking --run --scorers
# --scorer-port0 --spread (c2_dev_orch.py), its own arguments in ORCH_EXTRA; VARS then only names the run
if [ -n "${ORCH:-}" ]; then
  setsid "$PY" "$ORCH" --run "$RUN" --scorers "$NS" --scorer-port0 "$SCORER_PORT0" --spread ${ORCH_EXTRA:-} \
    > "$D/orch.out" 2>&1 < /dev/null 9>&- &
else
setsid "$PY" experiments/mist_w3po_v1/scripts/speech_var_orch.py --run "$RUN" --vars "$VARS" --scorers "$NS" \
  --scorer-port0 "$SCORER_PORT0" --spread ${ORCH_EXTRA:-} > "$D/orch.out" 2>&1 < /dev/null 9>&- &
fi
OP=$!; echo $OP > "$D/orch.pid"; log "orchestrator pid $OP run $RUN vars $VARS"
while [ -d /proc/$OP ]; do sleep 60; done
log "orchestrator exited; DONE=$([ -e "$RUN/events/DONE" ] && cat "$RUN/events/DONE" || echo no)"
log "stop scorers [$SPIDS]"; kill -TERM $SPIDS 2>/dev/null
log "launcher done"
