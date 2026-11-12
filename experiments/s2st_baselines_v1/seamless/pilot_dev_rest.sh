#!/usr/bin/env bash
# the other five dev-calibration processes of pilot_dev.sh (first one, en2zh A on card 5, started by hand), staggered
set -u
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
P=experiments/s2st_baselines_v1/seamless/pilot_dev.sh
B="0.9:192:50,0.3:64:50,0.5:64:50"; C="0.7:64:50,0.5:64:25,0.3:32:25"; A="0.3:192:50,0.5:192:50,0.7:192:50"
bash $P 5 en2zh B "$B" & 
bash $P 4 en2zh C "$C" &
bash $P 7 zh2en A "$A" &
bash $P 6 zh2en C "$C" &
sleep 65
bash $P 7 zh2en B "$B" &
wait
