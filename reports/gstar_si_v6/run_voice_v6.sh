#!/usr/bin/env bash
# Paper v6, review M09 / T14: voice similarity, confusion and in-meeting identification of the existing speech outputs
# with the corrected Zh->En matching (gstar_voice_v6.py), with WavLM-SV (as before) and TitaNet-large (second SV model).
# Small models (~2-3 GB); CARD_W / CARD_T pick the cards. Outputs: eval_outputs/.../gstar_v6_20261031/voice/ (new files).
#   CARD_W=5 CARD_T=7 setsid nohup bash reports/gstar_si_v6/run_voice_v6.sh > /dev/null 2>&1 < /dev/null &
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
exec >> logs/multispk/v6/voice.log 2>&1
EV=eval_outputs/uniss_qwen3_omni_v1; O=$EV/gstar_v6_20261031/voice; D=data/ami_multispk_si_v1
mkdir -p $O
PYE=/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval/bin/python; PYN=/opt/dlami/nvme/jasonleeeli/venvs/nemo_diar/bin/python
M=experiments.uniss_qwen3_omni_v1.evaluation
SYS="en2zh_O2:$EV/gstar_v3_20261029/en2zh_O2:$D/e2e_gstar_v1:$EV/gstar_v4_speech_20261030/en2zh_O2
en2zh_C-ft2:$EV/gstar_v3_20261029/en2zh_C-ft2:$D/e2e_gstar_v1:$EV/gstar_v4_speech_20261030/en2zh_C-ft2
en2zh_S2dl_d1:$EV/gstar_v3_20261029/en2zh_S2dl_d1:$D/e2e_gstar_stream_v1:$EV/gstar_v4_speech_20261030/en2zh_S2dl_d1
en2zh_O2_LA2_d1:$EV/gstar_v3_20261029/en2zh_O2_LA2_d1:$D/e2e_gstar_stream_v1:$EV/gstar_v4_speech_20261030/en2zh_O2_LA2_d1
zh2en_O2:$EV/gstar_v3_20261029/zh2en_O2:$D/e2e_gstar_zh2en_v1:$EV/gstar_v4_speech_20261030/zh2en_O2
zh2en_C-ft2:$EV/gstar_v3_20261029/zh2en_C-ft2:$D/e2e_gstar_zh2en_v1:$EV/gstar_v4_speech_20261030/zh2en_C-ft2
zh2en_S2dl_d1:$EV/gstar_v5_20261030/zh2en_S2dl3k_d1:$D/e2e_gstar_zh2en_v1:$EV/gstar_v5_speech_20261030/zh2en_S2dl3k_d1
zh2en_O2_LA2_d1:$EV/gstar_v3_20261029/zh2en_O2_LA2_d1:$D/e2e_gstar_zh2en_v1:$EV/gstar_v4_speech_20261030/zh2en_O2_LA2_d1"
for line in $SYS; do
  IFS=: read -r name hyp win sp <<< "$line"
  [ -e $sp/talker/results.jsonl ] || { echo "skip $name (no speech)"; continue; }
  [ -e $O/${name}_wavlm.json ] || CUDA_VISIBLE_DEVICES=${CARD_W:-5} PYTHONPATH=. $PYE -m $M.gstar_voice_v6 --windows $win/windows_test_ref.jsonl \
    --hyp $hyp/hyp.jsonl --results $sp/talker/results.jsonl --out $O/${name}_wavlm.json
  [ -e $O/${name}_titanet.json ] || CUDA_VISIBLE_DEVICES=${CARD_T:-7} PYTHONPATH=. $PYN -m $M.gstar_voice_v6 --sv titanet \
    --windows $win/windows_test_ref.jsonl --hyp $hyp/hyp.jsonl --results $sp/talker/results.jsonl --out $O/${name}_titanet.json
  echo "$name done"
done
echo VOICE_V6_DONE
