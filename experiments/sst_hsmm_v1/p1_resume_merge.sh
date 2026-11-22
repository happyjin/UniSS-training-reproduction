#!/usr/bin/env bash
# sst_hsmm_v1: resume after the 2026-11-22 22:18 CET merge failure (merge.py REPO path was parents[3], copied from a deeper
# file; training itself finished and saved). p1/FAILED renamed to FAILED_merge_path_20261122 by hand. Runs the same merge +
# check as p1_orch.sh, touches p1/DONE, then starts p2_orch.sh, p1_diag_orch.sh and p3_chain.sh detached (each refuses
# existing outputs). Checked with bash -n / DRY_RUN only.
#   setsid nohup bash experiments/sst_hsmm_v1/p1_resume_merge.sh > logs/sst_hsmm_v1/p1_resume_merge.log 2>&1 < /dev/null 8>&- 9>&- &
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
R=eval_outputs/sst_hsmm_v1/p1; LOGD=logs/sst_hsmm_v1/p1
source experiments/sst_hsmm_v1/lib.sh
[ -n "${DRY_RUN:-}" ] && { echo "[dry] merge -> DONE -> p2_orch, p1_diag_orch, p3_chain"; exit 0; }
[ -e $R/FAILED ] && { log "p1/FAILED still present"; exit 1; }
[ -e $R/merged ] && { log "exists: $R/merged"; exit 1; }
[ -s $R/train/adapters_final.pt ] || { log "no adapters"; exit 1; }
PYTHONPATH=. $PYQ experiments/sst_hsmm_v1/merge.py --adapters $R/train/adapters_final.pt --out $R/merged > $LOGD/merge_resume.log 2>&1 || { log "merge failed"; touch $R/FAILED; exit 1; }
grep -q '"adapted"' $LOGD/merge_resume.log || { log "merge check failed"; touch $R/FAILED; exit 1; }
touch $R/DONE
log "p1 done (merge resumed): $(tail -n 1 $LOGD/merge_resume.log)"
for s in p2_orch p1_diag_orch p3_chain; do
  setsid nohup bash experiments/sst_hsmm_v1/$s.sh >> logs/sst_hsmm_v1/$s.log 2>&1 < /dev/null 8>&- 9>&- &
  log "$s pid $!"; sleep 5
done
