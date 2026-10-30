#!/usr/bin/env bash
# Second half of run_cvss_table1.sh, whose top-level bash was stopped by PID (it would have restarted the GPU holder on
# cards 0-3 while the tier queue, jobs/tier_jobs.json, still needs them): wait for its two speech subshells (PIDs given),
# then A.PCP + SIM-O for cvss_base and cvss_s0 on cards 0-3 (small encoders, beside the tier queue). No holder here.
# Log: logs/si_v4/cvss_table1.log
set -u
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
LOG=logs/si_v4/cvss_table1.log
log() { echo "$(date -u +%FT%TZ) $*" >> $LOG; }
V4=eval_outputs/uniss_qwen3_omni_v1/si_v4
EPY=/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval/bin/python
for p in "$@"; do while [ -e /proc/$p ]; do sleep 30; done; done
log "speech subshells done; A.PCP + SIM-O"
g=0
for m in base s0; do for d in en2zh zh2en; do
  run=$V4/speech/cvss_$m/cvss_$d
  if [ -f $run/asr_speech.jsonl ]; then
    PYTHONPATH=. $EPY reports/uniss_si_v4/cvss_table1_metrics.py $run $V4/cvss_table1/${m}_$d $g >> $V4/cvss_table1_${m}_$d.log 2>&1 &
    g=$((g+1)); sleep 20
  else log "missing $run/asr_speech.jsonl"; fi
done; done
wait
log "metrics: $(grep -l CVSS_TABLE1_METRICS_DONE $V4/cvss_table1_*.log 2>/dev/null | wc -l)/4 done"
log "CVSS_TABLE1_DONE"
