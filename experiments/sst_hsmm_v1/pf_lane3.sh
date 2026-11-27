#!/usr/bin/env bash
# portfolio 2026-11-27 lane 3: C1 = calibrate the theta map on the B0 dev runs (CPU), then the C1 pipeline (no rollouts, h14, one spec).
source experiments/sst_hsmm_v1/pf_lane_lib.sh
until [ -e $B0W ]; do [ -e $D0/pfb0/FAILED ] && { log "B0 failed"; exit 1; }; sleep 60; done
PYTHONPATH=. /opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval/bin/python experiments/sst_hsmm_v1/pf_c1_calib.py --ref $D0/pfb0 --out $D0/pfc1_theta.json > logs/sst_hsmm_v1/pfc1_calib.log 2>&1 || { log "calib failed"; exit 1; }
log "theta map: $(tr '\n' ' ' < logs/sst_hsmm_v1/pfc1_calib.log | cut -c1-300)"
run pfc1 ROLL=0 GRID="0.75" XF="--theta-map $D0/pfc1_theta.json"
touch $D0/PF_LANE3_DONE
