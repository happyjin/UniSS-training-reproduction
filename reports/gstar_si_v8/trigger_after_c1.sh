#!/usr/bin/env bash
# run after_c1.py once both C1 trainings have finished and merged
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
until grep -q THINKER_GSTAR_DONE logs/multispk/v8/thinker_gstar_s2dlnosd_en2zh_v8.log && grep -q THINKER_GSTAR_DONE logs/multispk/v8/thinker_gstar_s2dlnosd_zh2en_v8.log; do sleep 60; done
python3 reports/gstar_si_v8/after_c1.py && echo AFTER_C1_DONE
