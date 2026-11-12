#!/usr/bin/env bash
# dev calibration round 2 (lower thresholds / waits / unit chunks), same as pilot_dev.sh; staggered 60 s per card
set -u
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
P=experiments/s2st_baselines_v1/seamless/pilot_dev.sh
D="0.1:192:50,0.2:192:50,0.1:32:25"; E="0.2:32:25,0.1:0:10,0.3:0:10"
bash $P 5 en2zh D "$D" &
bash $P 7 zh2en D "$D" &
sleep 65
bash $P 5 zh2en E "$E" &
bash $P 7 en2zh E "$E" &
wait
