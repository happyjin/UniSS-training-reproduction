#!/usr/bin/env bash
# portfolio 2026-11-27 (11:55 CET): replaces lanes 1/2 (stopped by PID after their current B1/B2 pipelines; old run() truncated logs of
# methods another lane had already run). Three lanes share the remaining methods A1 A2 D1 D2 C2 in rotated order (work-stealing).
source experiments/sst_hsmm_v1/pf_lane_lib2.sh
A1; D1; A2; C2; D2
touch $D0/PF_LANE6_DONE
