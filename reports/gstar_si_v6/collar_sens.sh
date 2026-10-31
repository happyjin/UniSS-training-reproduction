#!/usr/bin/env bash
# Paper v6 (review of the v6 draft, M07/T22): tcpBLEU collar sensitivity (0 / 0.5 / 1 / 2 s) of the whole-meeting online
# runs and of the window runs on the same meetings; idempotent, new files only (scores/collar/). CPU.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
PY=/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval/bin/python
EV=eval_outputs/uniss_qwen3_omni_v1; V6=$EV/gstar_v6_20261031; O=$V6/scores/collar; ON=data/online_v6_20261031
mkdir -p $O
REFE=$ON/windows_test_ref_en2zh_offline_sub.jsonl; REFS=$ON/windows_test_ref_en2zh_stream_sub.jsonl
REFZ=$ON/windows_test_ref_zh2en_sub.jsonl; REFA=$ON/windows_test_ref_ali_sub.jsonl
run() {   # name set windows hyp ref extra
  for c in 0 0.5 1 2; do
    out=$O/$1_c$c.json
    [ -e $out ] || [ ! -e $4 ] || $PY reports/gstar_si_v6/meeting_score.py --dir $2 --windows $3 --hyp $4 --ref-windows $5 $6 \
      --collar $c --out $out > /dev/null 2>&1
  done
}
for st in en2zh zh2en ali; do
  R=$([ $st = en2zh ] && echo $REFE || ([ $st = zh2en ] && echo $REFZ || echo $REFA))
  for s in S2dl O2_LA2; do run on_${st}_$s $st $ON/windows_test_online_${st}_sub.jsonl $V6/on_${st}_$s/hyp.jsonl $R --online; done
done
run winsub_en2zh_S2dl en2zh $REFS $EV/gstar_v3_20261029/en2zh_S2dl_d1/hyp.jsonl $REFE ""
run winsub_en2zh_O2_LA2 en2zh $REFS $EV/gstar_v3_20261029/en2zh_O2_LA2_d1/hyp.jsonl $REFE ""
run winsub_zh2en_S2dl zh2en $REFZ $EV/gstar_v5_20261030/zh2en_S2dl3k_d1/hyp.jsonl $REFZ ""
run winsub_zh2en_O2_LA2 zh2en $REFZ $EV/gstar_v3_20261029/zh2en_O2_LA2_d1/hyp.jsonl $REFZ ""
run winsub_ali_S2dl ali $REFA $EV/gstar_v5_20261030/ali_S2dl3k_d1/hyp.jsonl $REFA ""
run winsub_ali_O2_LA2 ali $REFA $EV/gstar_v3_20261029/ali_O2_LA2_d1/hyp.jsonl $REFA ""
echo COLLAR_DONE
