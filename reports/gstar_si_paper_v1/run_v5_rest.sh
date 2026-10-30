#!/usr/bin/env bash
# Paper v5: finish Zh->En S2-dl 3k at 0.5 s (its first decode was stopped by PID when runaway generations grew the
# batch-16 workers to 35-56 GB). make_v5_rest.py writes the remaining windows per shard; here one worker per card, batch 4
# (REST_WORKER_MIB=36000 -> one per card), and a watchdog: a card above 134 GB (124 GB with a colleague's process) loses
# our largest worker on it (by PID, logged). Then the system is scored if complete. Never touches other users' processes.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
exec >> logs/multispk/v5_rest.log 2>&1
log() { echo "[$(date -u +%F' '%T) UTC] $*"; }
EV=eval_outputs/uniss_qwen3_omni_v1; OUT=${EV}/gstar_v5_20261030; NAME=zh2en_S2dl3k_d0.5; JR=reports/gstar_si_paper_v1/jobs/v5_runs
PYE=/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval/bin/python; PYM=/opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python
M=experiments.uniss_qwen3_omni_v1.evaluation; S2DLZ=${EV}/thinker_gstar_s2dl_zh2en_3k_20261030/thinker_merged
WZ=data/ami_multispk_si_v1/e2e_gstar_zh2en_v1/windows_test_ref.jsonl

REST_WORKER_MIB=36000 python3 reports/gstar_si_paper_v1/make_v5_rest.py ${NAME} 7 0.5 0,1,2,3,4,5,6,7 4 || { log "make_v5_rest failed"; exit 1; }
pids=()
for f in ${JR}/rest_${NAME}_card*.json; do
  c=${f##*_card}; c=${c%.json}
  CUDA_VISIBLE_DEVICES=$c PYTHONPATH=.:third_party/Megatron-LM ${PYM} -m ${M}.gstar_multi --thinker ${S2DLZ} $f \
    > logs/multispk/v5/rest_card$c.log 2>&1 &
  pids+=($!); log "card $c: gstar_multi $!"; sleep 10
done

alive() { for p in "${pids[@]}"; do kill -0 $p 2>/dev/null && return 0; done; return 1; }
while alive; do
  for c in 0 1 2 3 4 5 6 7; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $c)
    lim=$(python3 -c "import sys; sys.path.insert(0, 'reports/gstar_si_paper_v1'); from make_v5_runs import foreign; print(124000 if foreign($c) else 134000)")
    [ "$used" -gt "$lim" ] || continue
    bus=$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader -i $c)
    big=$(nvidia-smi --query-compute-apps=pid,gpu_bus_id,used_memory --format=csv,noheader,nounits | awk -F', ' -v b="$bus" 'tolower($2)==tolower(b) {print $3, $1}' \
          | sort -rn | while read m p; do case "$(readlink /proc/$p/cwd)" in /opt/dlami/nvme/jasonleeeli/*) ;; *) continue ;; esac
              tr '\0' ' ' < /proc/$p/cmdline | grep -q "multiprocessing" && { echo "$p $m"; break; }; done)
    [ -n "$big" ] || continue
    kill ${big%% *}; log "card $c at ${used} MiB > ${lim}: stopped our worker ${big} by PID"
    echo "[$(date -u +%F' '%T) UTC] M rest watchdog: card $c at ${used} MiB > ${lim}; stopped our worker ${big}" >> logs/gpu_coord/events.log
  done
  sleep 20
done
n=$(cat ${OUT}/${NAME}/hyp_s*.jsonl | wc -l)
if [ "$n" -eq "$(wc -l < ${WZ})" ] && [ ! -e ${OUT}/${NAME}/hyp.jsonl ]; then
  cat ${OUT}/${NAME}/hyp_s*.jsonl > ${OUT}/${NAME}/hyp.jsonl
  PYTHONPATH=. ${PYE} -m ${M}.gstar_score --windows ${WZ} --hyp ${OUT}/${NAME}/hyp.jsonl --post-key lowlat \
    --out ${OUT}/${NAME}/score.json > ${OUT}/${NAME}/score.log 2>&1
  log "${NAME}: $(grep -m1 cpBLEU_session ${OUT}/${NAME}/score.json) $(grep -m1 LAAL_s ${OUT}/${NAME}/score.json)"
else log "${NAME}: $n / $(wc -l < ${WZ}) windows, not scored"; fi
log "V5_REST_DONE"
