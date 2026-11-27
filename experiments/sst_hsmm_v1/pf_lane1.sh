#!/usr/bin/env bash
# portfolio 2026-11-27 lane 1: B1 -> A2 -> D2 -> C2 (each waits for the B0 reference runs).
source experiments/sst_hsmm_v1/pf_lane_lib.sh
run pfb1 WAIT=$B0W ROLL=1 FIT=hsmm_ctr XF="--ctr-cert"
run pfa2 WAIT=$B0W ROLL=1 FIT=hsmm_anchor XF="--mid-mbr 6 --mbr-util bid"
run pfd2 WAIT=$B0W ROLL=1 FIT=hsmm_anchor XF="--mid-mbr 6 --mbr-util support --sentcut"
run pfc2 WAIT=$B0W ROLL=1 FIT=hsmm_map ROLL_ITEMS=$D0/cvdev_a/data ROLL_VIEWS=eval_outputs/mist_w3po_v1/wave15/views FITX="--items-dir $D0/cvdev_a/data --views eval_outputs/mist_w3po_v1/wave15/views" CVDEV_DATA=$D0/cvdev_b/data
touch $D0/PF_LANE1_DONE
