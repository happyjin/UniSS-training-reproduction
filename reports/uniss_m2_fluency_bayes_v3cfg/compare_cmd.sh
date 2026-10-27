# usage: bash compare_cmd.sh <suffix> [--utmos file]
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
E=eval_outputs/uniss_qwen3_omni_v1/si_t1c_20261021T181440Z/full_iter0001500; R=reports/uniss_m2_fluency_bayes_v3cfg; PY="env PYTHONPATH=. /opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval/bin/python -m experiments.uniss_qwen3_omni_v1.evaluation.si_fluency_compare"
$PY --base v3_en2zh "${@:2}" --run v3_en2zh=$E/system_flu7-sq-b1_m0.5/realsi_en2zh --run m2_en2zh=$E/system_swb-q-b1_m0.5/realsi_en2zh \
 --run v3b0.5_en2zh=$E/system_flu7-sq_m0.5/realsi_en2zh --run v3b1.5_en2zh=$E/system_bay-v3-b1.5_m0.5/realsi_en2zh --run v3b2_en2zh=$E/system_bay-v3-b2_m0.5/realsi_en2zh \
 --run nosdr_en2zh=$E/system_bay-nosdr_m0.5/realsi_en2zh --run sdrcal_en2zh=$E/system_bay-sdrcal_m0.5/realsi_en2zh \
 --run betab0.5_en2zh=$E/system_bay-beta-b0.5_m0.5/realsi_en2zh --run betab1_en2zh=$E/system_bay-beta_m0.5/realsi_en2zh \
 --run la2w0.5_en2zh=$E/system_la2_m0.5/realsi_en2zh --run la2w0.75_en2zh=$E/system_la2_m0.75/realsi_en2zh --run la2w1_en2zh=$E/system_la2_m1/realsi_en2zh \
 --run la2v3w0.5_en2zh=$E/system_la2v3_m0.5/realsi_en2zh --run la2v3w0.6_en2zh=$E/system_la2v3_m0.6/realsi_en2zh --run la2v3w0.75_en2zh=$E/system_la2v3_m0.75/realsi_en2zh --run la2v3w1_en2zh=$E/system_la2v3_m1/realsi_en2zh \
 --out $R/$1_en2zh.json > $R/$1_en2zh.log 2>&1 &
$PY --base v3_zh2en "${@:2}" --run v3_zh2en=$E/system_flu7-sq-zh_m1/realsi_zh2en --run m2_zh2en=$E/system_swb-q-b0-zh_m1/realsi_zh2en \
 --run v3bm0.5_zh2en=$E/system_flu7-sq-bm0p5-zh_m1/realsi_zh2en --run v3b0_zh2en=$E/system_bay-v3-zh-b0_m1/realsi_zh2en --run v3b0.5_zh2en=$E/system_bay-v3-zh-b0.5_m1/realsi_zh2en \
 --run nosdr_zh2en=$E/system_bay-nosdr-zh_m1/realsi_zh2en --run sdrcal_zh2en=$E/system_bay-sdrcal-zh_m1/realsi_zh2en \
 --run betabm1.5_zh2en=$E/system_bay-beta-zh-b-1.5_m1/realsi_zh2en --run betabm1_zh2en=$E/system_bay-beta-zh_m1/realsi_zh2en \
 --run la2w0.5_zh2en=$E/system_la2_m0.5/realsi_zh2en --run la2w0.75_zh2en=$E/system_la2_m0.75/realsi_zh2en --run la2w1_zh2en=$E/system_la2_m1/realsi_zh2en \
 --run la2v3w0.4_zh2en=$E/system_la2v3-zh_m0.4/realsi_zh2en --run la2v3w0.5_zh2en=$E/system_la2v3-zh_m0.5/realsi_zh2en --run la2v3w0.75_zh2en=$E/system_la2v3-zh_m0.75/realsi_zh2en \
 --out $R/$1_zh2en.json > $R/$1_zh2en.log 2>&1
wait
