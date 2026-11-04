#!/usr/bin/env bash
# when the 4 SDM diarizer shards are done: build the SDM online windows, then queue the S / Rel SDM decodes
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
until [ "$(grep -h '^\[dump\] DONE' logs/multispk/v9/sdm_diar_s*.log | wc -l)" = 4 ]; do sleep 60; done
O=eval_outputs/uniss_qwen3_omni_v1/gstar_v9_20261104/sdm
/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval/bin/python reports/gstar_si_v9/build_online_windows_v9.py --set sdm --out $O > logs/multispk/v9/sdm_windows.log 2>&1 && \
  cp -n reports/gstar_si_v9/jobs/sdm_decode.json logs/multispk/v9/q/050_sdm_decode.json && echo SDM_QUEUED
