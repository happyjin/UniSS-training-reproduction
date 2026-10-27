# usage: bash compare_cmd.sh <prefix> [--utmos file]   (RealSI, paired bootstrap vs v3)
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
E=eval_outputs/uniss_qwen3_omni_v1/si_t1c_20261021T181440Z/full_iter0001500; R=reports/uniss_m2_bayesrisk_v1
PY="env PYTHONPATH=.:.tmp/pylib_jieba /opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval/bin/python -m experiments.uniss_qwen3_omni_v1.evaluation.si_fluency_compare"
en=""; for b in 1 1.5 2 2.5; do en="$en --run brA-b${b}_en2zh=$E/system_brA-b${b}_m0.5/realsi_en2zh"; done
for b in 0.5 0.75 1; do en="$en --run brPF-b${b}_en2zh=$E/system_brPF-b${b}_m0.5/realsi_en2zh"; done
for b in 0.5 0.75; do en="$en --run brPFO-b${b}_en2zh=$E/system_brPFO-b${b}_m0.5/realsi_en2zh"; done
for b in 2.5 3 3.5; do en="$en --run v3b${b}_en2zh=$E/system_bay-v3-b${b}_m0.5/realsi_en2zh"; done
zh=""; for b in -0.5 0 0.5 1; do zh="$zh --run brA-b${b}_zh2en=$E/system_brA-zh-b${b}_m1/realsi_zh2en --run brP-b${b}_zh2en=$E/system_brP-zh-b${b}_m1/realsi_zh2en"; done
for b in -0.5 0 0.5; do zh="$zh --run brPO-b${b}_zh2en=$E/system_brPO-zh-b${b}_m1/realsi_zh2en"; done
for b in 1 1.5; do zh="$zh --run v3b${b}_zh2en=$E/system_bay-v3-zh-b${b}_m1/realsi_zh2en"; done
$PY --base v3_en2zh "${@:2}" --run v3_en2zh=$E/system_flu7-sq-b1_m0.5/realsi_en2zh --run m2_en2zh=$E/system_swb-q-b1_m0.5/realsi_en2zh \
 --run v3b0.5_en2zh=$E/system_flu7-sq_m0.5/realsi_en2zh --run v3b1.5_en2zh=$E/system_bay-v3-b1.5_m0.5/realsi_en2zh --run v3b2_en2zh=$E/system_bay-v3-b2_m0.5/realsi_en2zh \
 $en --out $R/$1_en2zh.json > $R/$1_en2zh.log 2>&1 &
$PY --base v3_zh2en "${@:2}" --run v3_zh2en=$E/system_flu7-sq-zh_m1/realsi_zh2en --run m2_zh2en=$E/system_swb-q-b0-zh_m1/realsi_zh2en \
 --run v3bm0.5_zh2en=$E/system_flu7-sq-bm0p5-zh_m1/realsi_zh2en --run v3b0_zh2en=$E/system_bay-v3-zh-b0_m1/realsi_zh2en --run v3b0.5_zh2en=$E/system_bay-v3-zh-b0.5_m1/realsi_zh2en \
 $zh --out $R/$1_zh2en.json > $R/$1_zh2en.log 2>&1
wait
