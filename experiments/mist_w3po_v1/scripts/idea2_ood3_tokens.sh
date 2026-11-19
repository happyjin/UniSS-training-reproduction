#!/usr/bin/env bash
# Idea 2 prereg v3 §C: shrink+Vfb token tables for the OOD-3 subsets from the B T1c full runs (items / blocks / prompts),
# restricted to the pre-registered meetings. Card 5 (stacked). Never overwrites (idea2_stream_tokens refuses).
#   [DRY_RUN=1] bash experiments/mist_w3po_v1/scripts/idea2_ood3_tokens.sh
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
export HF_HOME=/opt/dlami/nvme/jasonleeeli/cache/huggingface
VPY=/opt/dlami/nvme/jasonleeeli/venvs/vllm_omni_030/bin/python
FIT=eval_outputs/mist_w3po_v1/voicecond_idea2/shrink_fit_all
RT=joint_v8h6k_20261113T073658Z_s06000__t1c
DD=eval_outputs/mist_w3po_v1/data
run() { if [ -n "${DRY_RUN:-}" ]; then echo "[dry] $*"; else echo "$(date -u +%FT%TZ) $*"; eval "$@"; fi; }
run "CUDA_VISIBLE_DEVICES=5 $VPY -I experiments/mist_w3po_v1/voicecond/idea2_stream_tokens.py --data data/ami_multispk_si_v1 --dir zh2en --session zh_mix_v8 --diar nemotron_lowlat_zh8 --lang zh --run jtest_bami_$RT --windows $DD/windows_test_c2g06_v1/windows_test_online_zh2en.jsonl --fit $FIT --tag ood3_ami --meetings EN2002d IS1009c"
run "CUDA_VISIBLE_DEVICES=5 $VPY -I experiments/mist_w3po_v1/voicecond/idea2_stream_tokens.py --data data/ami_multispk_si_v1 --dir en2zh --session ihm_mix --diar nemotron_lowlat --lang en --run jtest_ben_$RT --windows $DD/windows_test_en2zh_c2g06_v1/windows_test_online_en2zh.jsonl --fit $FIT --tag ood3_en --meetings EN2002d IS1009c"
run "CUDA_VISIBLE_DEVICES=5 $VPY -I experiments/mist_w3po_v1/voicecond/idea2_stream_tokens.py --data data/alimeeting_si_v1 --dir zh2en --session near_mix --diar nemotron_lowlat --lang zh --run jtest_bali_$RT --windows $DD/windows_test_c2g06_v1/windows_test_online_ali.jsonl --fit $FIT --tag ood3_ali --meetings R8009_M8021 R8009_M8024"
