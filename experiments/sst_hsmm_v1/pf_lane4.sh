#!/usr/bin/env bash
# portfolio 2026-11-27 lane 4: E1 started as soon as the P2F-T training finished (lane 2 skips pfe1 later: refuse-to-overwrite).
source experiments/sst_hsmm_v1/pf_lane_lib.sh
run pfe1 WAIT=$D0/p2ft/TRAIN_DONE ROLL=1 FIT=hsmm_anchor XF="--draft-lora $D0/p2ft/peft/a1.0000 --draft-tag"
touch $D0/PF_LANE4_DONE
