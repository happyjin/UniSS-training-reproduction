#!/usr/bin/env bash
# portfolio 2026-11-27 lane 2: B2 -> A1 -> D1 -> E1 (E1 also waits for the P2F-T training).
source experiments/sst_hsmm_v1/pf_lane_lib.sh
run pfb2 WAIT=$B0W ROLL=1 FIT=hsmm_anchor XF="--bnd-cut"
run pfa1 WAIT=$B0W ROLL=1 FIT=hsmm_anchor XF="--bias-redraft 1.0"
run pfd1 WAIT=$B0W ROLL=1 FIT=hsmm_anchor XF="--mid-mbr 6 --mbr-util support"
run pfe1 WAIT=$D0/p2ft/TRAIN_DONE ROLL=1 FIT=hsmm_anchor XF="--draft-lora $D0/p2ft/peft/a1.0000 --draft-tag"
touch $D0/PF_LANE2_DONE
