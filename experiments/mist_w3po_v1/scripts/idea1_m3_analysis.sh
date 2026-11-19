#!/usr/bin/env bash
# Idea 1 M3: waits for logs/mist_w3po_v1/voicecond_idea1/m3/M3_RUNS_DONE (or M3_RUNS_FAILED), then runs
# idea1_m3_analysis bench (prereg v4.3/v4.4) -> eval_outputs/mist_w3po_v1/voicecond_idea1/m3/test_summary.json and the
# marker logs/mist_w3po_v1/voicecond_idea1/m3/M3_ANALYSIS_DONE (M3_ANALYSIS_FAILED on error).
#   [DRY_RUN=1] bash experiments/mist_w3po_v1/scripts/idea1_m3_analysis.sh     (checked with bash -n / DRY_RUN only)
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
L=logs/mist_w3po_v1/voicecond_idea1/m3
[ -n "${DRY_RUN:-}" ] && { echo "[dry] wait $L/M3_RUNS_{DONE,FAILED}; idea1_m3_analysis bench -> m3/test_summary.json; $L/M3_ANALYSIS_DONE"; exit 0; }
until [ -e $L/M3_RUNS_DONE ] || [ -e $L/M3_RUNS_FAILED ]; do sleep 60; done
if PYTHONPATH=. /opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval/bin/python -m experiments.mist_w3po_v1.voicecond.idea1_m3_analysis bench > $L/m3_analysis.log 2>&1; then
  date > $L/M3_ANALYSIS_DONE
else
  date > $L/M3_ANALYSIS_FAILED
fi
