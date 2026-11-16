#!/usr/bin/env bash
# Table 1 (paper v14; reports/mist_w3po_v1/multispk_s2st_baselines_prereg_v2.zh-CN.md §2): chunk-protocol speech latency
# units of external / baseline runs, exactly s4000_postproc.sh latu()'s commands (lat_units_dump, LATU_MODE perword and
# blockstart, speech_latency_lit on the chunk windows). CPU only. Existing outputs are kept.
#   bash table1_lat.sh <set: ami|ali|en> <run name> [<run name> ...]      [DRY_RUN=1]
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
. experiments/mist_w3po_v1/scripts/joint_lib.sh
A=$JO/table1_v14
run() { if [ -n "${DRY_RUN:-}" ]; then echo "DRY: $*" | cut -c1-300; else "$@"; fi; }
s=$1; shift
for n in "$@"; do
  for mode in perword blockstart; do
    f=$A/lat/$mode/$n.json
    [ -e $f ] && continue
    [ -n "${DRY_RUN:-}" ] || mkdir -p $A/lat/$mode
    case $s in
      ami) run env LATU_MODE=$mode CUDA_VISIBLE_DEVICES= $PY -m experiments.mist_w3po_v1.evaluation.lat_units_dump \
             experiments.mist_w3po_v1.evaluation.on_refset --ref-win $RT_AMI --dr zh2en \
             experiments.mist_w3po_v1.evaluation.speech_latency_lit --run $DEC/$n --talker $SPD/$n/talker --win $WT_AMIC --out $f;;
      ali) run env LATU_MODE=$mode CUDA_VISIBLE_DEVICES= $PY -m experiments.mist_w3po_v1.evaluation.lat_units_dump \
             experiments.mist_w3po_v1.evaluation.on_refset --ref-win $RT_ALI --dr ali \
             experiments.mist_w3po_v1.evaluation.speech_latency_lit --run $DEC/$n --talker $SPD/$n/talker --win $WT_ALIC --out $f;;
      en)  run env LATU_MODE=$mode CUDA_VISIBLE_DEVICES= $PY -m experiments.mist_w3po_v1.evaluation.lat_units_dump \
             experiments.mist_w3po_v1.evaluation.en2zh_lat --ref-win $RE_TEST --work $f.work --run $DEC/$n \
             --talker $SPD/$n/talker --win $WT_ENC --out $f;;
    esac
  done
done
