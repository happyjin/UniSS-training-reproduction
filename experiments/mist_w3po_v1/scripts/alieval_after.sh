#!/usr/bin/env bash
# v11 P0-2: after alieval_run.sh (marker ALIEVAL_DONE, or a FAILED_* file / ALIEVAL_STOP) -> alieval_postproc.sh ->
# alieval_analysis.py -> marker logs/mist_w3po_v1/joint/alieval/ALIEVAL_ANALYSIS_DONE (or ALIEVAL_ANALYSIS_FAILED).
# Read-only on the runs; refuses to overwrite the analysis json.
#   [DRY_RUN=1] bash experiments/mist_w3po_v1/scripts/alieval_after.sh
# Checked with bash -n and DRY_RUN=1 only.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
. experiments/mist_w3po_v1/scripts/joint_lib.sh
TV=$JL/alieval
OUT=$JO/alieval_analysis/analysis_v1.json
if [ -n "${DRY_RUN:-}" ]; then echo "wait $TV/ALIEVAL_DONE; postproc; analysis -> $OUT"; exit 0; fi
until [ -e $TV/ALIEVAL_DONE ] || ls $TV/FAILED_* > /dev/null 2>&1; do sleep 60; done
if [ ! -e $TV/ALIEVAL_DONE ]; then log "driver failed: $(cat $TV/FAILED_* | tr '\n' ' ' | cut -c1-300)"; touch $TV/ALIEVAL_ANALYSIS_FAILED; exit 1; fi
bash experiments/mist_w3po_v1/scripts/alieval_postproc.sh
CUDA_VISIBLE_DEVICES= $PY -m experiments.mist_w3po_v1.evaluation.alieval_analysis --out $OUT > $JO/alieval_analysis/analysis_v1.log 2>&1 \
  && { touch $TV/ALIEVAL_ANALYSIS_DONE; log "ALIEVAL_ANALYSIS_DONE"; } \
  || { touch $TV/ALIEVAL_ANALYSIS_FAILED; log "analysis failed (see $JO/alieval_analysis/analysis_v1.log)"; }
