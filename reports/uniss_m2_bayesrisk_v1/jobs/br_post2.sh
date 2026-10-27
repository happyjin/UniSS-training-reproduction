#!/bin/bash
# UTMOS + continuous abrupt analysis for one bayesrisk system: <cmp name> <system name> <dir> <window> <suffix> [blocks dir]
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
R=reports/uniss_m2_bayesrisk_v1; E=eval_outputs/uniss_qwen3_omni_v1/si_t1c_20261021T181440Z/full_iter0001500
TB=${6:-eval_outputs/uniss_qwen3_omni_v1/bayesrisk_v1/realsi}; EV=/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval/bin/python
c=$1; n=$2; d=$3; w=$4; s=$5; run=$E/system_${n}_m${w}/realsi_$d
[ -e $R/abrupt/${c}_s1b1.2q.txt ] || PYTHONPATH=. $EV reports/uniss_m2_fluency_results_v2/abrupt/abrupt.py "$c=$run=realsi:$d=$TB/blocks_${d}_w${w}${s}.jsonl=s1b1.2q" > $R/abrupt/${c}_s1b1.2q.txt 2>&1
[ -e $R/utmos/$c.jsonl ] || UTMOS_THREADS=12 $EV reports/uniss_m2_fluency_analysis_v1/scripts/utmos_runs.py $R/utmos/$c.jsonl "$c=$run" > $R/utmos/$c.log 2>&1
