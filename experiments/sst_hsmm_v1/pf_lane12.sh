#!/usr/bin/env bash
# plan v3 lane 12: H3 only (after the gate fix and a new smoke), so H3 does not wait for lane 9's pfe1d1.
source experiments/sst_hsmm_v1/pf_lane_lib2.sh
source experiments/sst_hsmm_v1/pv3_lane_lib.sh
H3
touch $D0/PV3_LANE12_DONE
