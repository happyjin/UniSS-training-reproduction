# usage: bash pair_cmd.sh [--utmos file]   paired bootstrap of each bayesrisk run against the nearest-LAAL v3 point (RealSI)
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
E=eval_outputs/uniss_qwen3_omni_v1/si_t1c_20261021T181440Z/full_iter0001500; R=reports/uniss_m2_bayesrisk_v1
PY="env PYTHONPATH=.:.tmp/pylib_jieba /opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval/bin/python -m experiments.uniss_qwen3_omni_v1.evaluation.si_fluency_compare"
dir_of() { case $1 in v3_en2zh) echo system_flu7-sq-b1_m0.5;; v3b0.5_en2zh) echo system_flu7-sq_m0.5;; v3_zh2en) echo system_flu7-sq-zh_m1;;
  v3bm0.5_zh2en) echo system_flu7-sq-bm0p5-zh_m1;; v3b*_en2zh) b=${1#v3b}; echo system_bay-v3-b${b%_en2zh}_m0.5;; v3b*_zh2en) b=${1#v3b}; echo system_bay-v3-zh-b${b%_zh2en}_m1;;
  br*_en2zh) echo system_${1%_en2zh}_m0.5;; br*_zh2en) n=${1%_zh2en}; echo system_${n%%-b*}-zh-b${n#*-b}_m1;; esac; }
pair() {  # base run, then bayesrisk runs
  local base=$1 d=${1##*_} args=""; shift
  for r in $base "$@"; do args="$args --run $r=$E/$(dir_of $r)/realsi_$d"; done
  $PY --base $base $UT $args --out $R/pair_${base}.json > $R/pair_${base}.log 2>&1
}
UT="$*"
pair v3b0.5_en2zh brA-b1_en2zh &
pair v3_en2zh brA-b1.5_en2zh brPF-b0.5_en2zh brPFO-b0.5_en2zh &
pair v3b2_en2zh brA-b2_en2zh brPF-b0.75_en2zh brPFO-b0.75_en2zh &
pair v3b2.5_en2zh brA-b2.5_en2zh &
pair v3b3_en2zh brPF-b1_en2zh &
pair v3_zh2en brA-b-0.5_zh2en brP-b-0.5_zh2en brPO-b-0.5_zh2en &
pair v3bm0.5_zh2en brA-b0_zh2en brP-b0_zh2en &
pair v3b0_zh2en brA-b0.5_zh2en brPO-b0_zh2en &
pair v3b0.5_zh2en brP-b0.5_zh2en brA-b1_zh2en brPO-b0.5_zh2en &
pair v3b1_zh2en brP-b1_zh2en &
wait
