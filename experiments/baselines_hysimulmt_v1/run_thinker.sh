#!/usr/bin/env bash
# The registered Hy-SimulMT Thinker runs (2 native points x 2 directions) against one server; run under setsid nohup.
# Usage: URL=http://127.0.0.1:18814/v1 bash experiments/baselines_hysimulmt_v1/run_thinker.sh
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
URL="${URL:?}"; PYH=/opt/dlami/nvme/jasonleeeli/venvs/hysimulmt/bin/python
export NO_PROXY='*' no_proxy='*' PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
E=eval_outputs/baselines_hysimulmt_v1/thinker; L=logs/baselines_hysimulmt
for p in hy-std-w4 hy-sc-w4; do
  for d in en2zh zh2en; do
    "$PYH" experiments/baselines_hysimulmt_v1/hy_realsi.py --direction $d --point $p --url "$URL" --out $E/$p \
      --concurrency 16 > $L/thinker_${p}_${d}.log 2>&1 &
  done
done
wait
echo "[$(TZ=Europe/Prague date +%H:%M:%S)] THINKER_ALL_DONE"
