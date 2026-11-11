#!/usr/bin/env bash
# dev5 text numbers of a FULL dev decode, the way T1-quick was scored (prereg T1-quick / T4 GO lines / rule C):
# hyp.jsonl cut to the 1092 dev5 windows -> decode/dev5cut_<name>; score_run (tcpBLEU all / ov) and
# speech_latency_lit --text-only on the dev5 windows; final-read word share; t1_stall_diag vs A' dev5.
# Also the final-read word share of the FULL dev decode (T1-ctrl rule). Waits for <decode>/score_text.json.
#   bash experiments/mist_w3po_v1/scripts/t_dev5_cut.sh <decode name> [DRY_RUN=1]
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
J=/opt/dlami/nvme/jasonleeeli
export TMPDIR=$J/work/mist_w3po_v1/tmp PYTHONPATH=.:third_party/Megatron-LM PATH=$J/venvs/vllm_omni_030/bin:$PATH \
  OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=
PY=$J/conda_envs/qwen3omni/bin/python
NAME=${1:?decode name}
D=eval_outputs/mist_w3po_v1/decode/$NAME
C=eval_outputs/mist_w3po_v1/decode/dev5cut_$NAME
O=eval_outputs/mist_w3po_v1/crf/t_eval/dev5cut_$NAME
W5=eval_outputs/mist_w3po_v1/data/windows_zh2en_c2g06_v1_dev5/windows_dev_online.jsonl
WF=eval_outputs/mist_w3po_v1/data/windows_zh2en_c2g06_v1/windows_dev_online.jsonl
REF=data/ami_multispk_si_v1/e2e_gstar_zh2en_v1/windows_validation_ref.jsonl
A5=eval_outputs/mist_w3po_v1/decode/plcrf_dev5_aprule/hyp.jsonl
PIECES=eval_outputs/mist_w3po_v1/crf/t0_agree_dev5/analysis/pieces.jsonl
DIAG=eval_outputs/mist_w3po_v1/speech_laal_v1/analysis_20261111/t1_stall_diag.py
FS=experiments/mist_w3po_v1/scripts/final_read_share.py
if [ -n "${DRY_RUN:-}" ]; then echo "$D -> $C, $O"; exit 0; fi
if [ -e "$C" ] || [ -e "$O" ]; then echo "拒绝覆盖已有输出: $C / $O" >&2; exit 1; fi
until [ -e "$D/score_text.json" ]; do sleep 30; done
mkdir -p "$C" "$O"
$PY -c "
import json,sys
ids={json.loads(l)['id'] for l in open('$W5')}
n=0
with open('$C/hyp.jsonl','x') as fh:
    for l in open('$D/hyp.jsonl'):
        if json.loads(l)['id'] in ids:
            fh.write(l); n+=1
print('dev5 windows', n, 'of', len(ids))
" > "$O/cut.log" 2>&1
$PY -m experiments.mist_w3po_v1.evaluation.score_run --run "$C" --windows "$W5" --ref-windows "$REF" --set zh2en \
  > "$C/score.log" 2>&1
$PY -m experiments.mist_w3po_v1.evaluation.speech_latency_lit --run "$C" --text-only --win "$W5" \
  --out "$O/textlat.json" > "$O/textlat.log" 2>&1
$PY $FS "$C/hyp.jsonl" "$W5" > "$O/final_share_dev5.json"
$PY $FS "$D/hyp.jsonl" "$WF" > "$O/final_share_fulldev.json"
$PY $DIAG "$C/hyp.jsonl" "$A5" "$PIECES" "$W5" > "$O/stall_diag.txt" 2>&1
echo "T_DEV5_CUT_DONE $NAME" >> "$O/cut.log"
