#!/usr/bin/env bash
# Resume the B3 prefix-model LA2 runs stopped by the v7 memory watchdog at 12:42 UTC (watchdog then stopped by its
# until-file): the windows not yet decoded, one 8-card spread decoder per direction, 60 s apart.
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
bash reports/gstar_si_v8/run_spread.sh en2zh_la2_r1 eval_outputs/uniss_qwen3_omni_v1/thinker_gstar_s2dl_20261029b/thinker_merged eval_outputs/uniss_qwen3_omni_v1/gstar_v8_20261103/runs/on_en2zh_S2dlLA2/resume1.windows.jsonl eval_outputs/uniss_qwen3_omni_v1/gstar_v8_20261103/runs/on_en2zh_S2dlLA2/hyp_r1.jsonl eval_outputs/uniss_qwen3_omni_v1/gstar_v8_20261103/runs/on_en2zh_S2dlLA2/infer_r1.log la2 en2zh &
sleep 60
bash reports/gstar_si_v8/run_spread.sh zh2en_la2_r1 eval_outputs/uniss_qwen3_omni_v1/thinker_gstar_s2dl_zh2en_3k_20261030/thinker_merged eval_outputs/uniss_qwen3_omni_v1/gstar_v8_20261103/runs/on_zh2en_S2dlLA2/resume1.windows.jsonl eval_outputs/uniss_qwen3_omni_v1/gstar_v8_20261103/runs/on_zh2en_S2dlLA2/hyp_r1.jsonl eval_outputs/uniss_qwen3_omni_v1/gstar_v8_20261103/runs/on_zh2en_S2dlLA2/infer_r1.log la2 zh2en &
wait
