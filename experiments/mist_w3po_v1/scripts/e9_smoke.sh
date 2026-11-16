#!/usr/bin/env bash
# E9 smoke test (reports/mist_w3po_v1/multispk_s2st_baselines_prereg_v2.zh-CN.md §4 feasibility): the cascade's GPU and
# CPU steps on 3 DEV windows (zh->en c2g06 dev, ES2011a), every command the test chain's (e9_orch.py dry-run
# commands) with smoke paths under $W: speaker-blind decode (online_decode_finalpost_zerosd, B's flags, 1 shard) ->
# causal relabel (relabel_causal_run) -> score_run -> Talker prep (gstar_talker_v6_unk, B's flags) -> drop_long_streams
# -> T1c si_infer (B's flags, 1 shard) -> speech-only -> ASR (Whisper-large-v3) -> asr_hyp -> score_run --asr.
# One card ($CARD). The card's no-checkpoint placeholder (supervisor + worker PIDs given) is stopped by PID after a
# cmdline check, right before the decode. Outputs never overwritten (every tool refuses).
#   CARD=4 SUP=<supervisor pid> WRK=<worker pid> bash experiments/mist_w3po_v1/scripts/e9_smoke.sh
# Run detached: setsid nohup bash ... > /opt/dlami/nvme/jasonleeeli/work/e9_smoke/smoke.log 2>&1 < /dev/null &
# Checked with bash -n only.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
. experiments/mist_w3po_v1/scripts/joint_lib.sh
export TMPDIR=/opt/dlami/nvme/jasonleeeli/work/mist_w3po_v1/tmp HF_HOME=/opt/dlami/nvme/jasonleeeli/cache/huggingface \
  TORCH_HOME=/opt/dlami/nvme/jasonleeeli/cache/torch XDG_CACHE_HOME=/opt/dlami/nvme/jasonleeeli/cache \
  TRITON_CACHE_DIR=/opt/dlami/nvme/jasonleeeli/cache/triton VLLM_CACHE_ROOT=/opt/dlami/nvme/jasonleeeli/cache/vllm_omni_030 \
  PYTHONPATH=.:third_party/Megatron-LM PATH=/opt/dlami/nvme/jasonleeeli/venvs/vllm_omni_030/bin:$PATH OMP_NUM_THREADS=8 \
  TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CARD=${CARD:?}; SUP=${SUP:?}; WRK=${WRK:?}
PYE=/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval/bin/python
PYV=/opt/dlami/nvme/jasonleeeli/venvs/vllm_omni_030/bin/python
W=/opt/dlami/nvme/jasonleeeli/work/e9_smoke
WIN=$W/windows_dev3.jsonl
NZ=eval_outputs/uniss_qwen3_omni_v1/thinker_gstar_s2dlnosd_zh2en_v8/thinker_merged
T1C=eval_outputs/uniss_qwen3_omni_v1/si_t1c_20261021T181440Z/autoext/export_1500/uniss
GUARD="--loop-guard-k 6 --loop-guard-maxp 12 --budget-tps 25 --budget-slack 50"   # finalpost_dev_orch.GUARD
RDEV=data/ami_multispk_si_v1/e2e_gstar_zh2en_v1/windows_validation_ref.jsonl   # finalpost_dev_orch.REF_WIN (zh->en dev)
N=smoke_blind; D=$W/dec/$N; R=$W/dec/${N}_rel; SP=$W/spd/${N}_rel__t1c
[ -e $W/dec ] && { log "拒绝覆盖已有输出: $W/dec"; exit 1; }
for p in $SUP $WRK; do
  c=$(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null)
  case "$c" in *scripts/gpu_load/train_mist_workload.py*"--devices $CARD "*) ;; *) log "pid $p is not card $CARD's placeholder: '$c'"; exit 1;; esac
done
kill -TERM $WRK $SUP; log "stopped placeholder worker $WRK + supervisor $SUP on card $CARD"
for i in $(seq 1 60); do kill -0 $WRK 2>/dev/null || kill -0 $SUP 2>/dev/null || break; sleep 1; done
mkdir -p $D
log "decode"
CUDA_VISIBLE_DEVICES=$CARD $PY -m experiments.mist_w3po_v1.evaluation.online_decode_finalpost_zerosd --final-read-post close_detect \
  --thinker $NZ --windows $WIN --out $D/hyp_s0_a1.jsonl --policy wait --lag-ms 2000 --context 0 --dir zh2en --batch 16 \
  --max-gib 120 --shard 0 --num-shards 1 --shard-by meeting $GUARD > $D/infer_s0_a1.log 2>&1
log "decode rc=$? $(grep -E 'ZERO_SD|W3PO_ONLINE_DONE|FINALPOST' $D/infer_s0_a1.log | tr '\n' ' ' | cut -c1-300)"
CUDA_VISIBLE_DEVICES= $PY -m experiments.mist_w3po_v1.evaluation.score_run --run $D --windows $WIN --ref-windows $RDEV \
  --set zh2en > $D/score.log 2>&1; log "score raw rc=$?"
CUDA_VISIBLE_DEVICES= $PY -m experiments.mist_w3po_v1.evaluation.relabel_causal_run --src $N --windows $WIN \
  --dec-root $W/dec > $W/relabel.log 2>&1; log "relabel rc=$? $(tail -c 400 $W/relabel.log)"
CUDA_VISIBLE_DEVICES= $PY -m experiments.mist_w3po_v1.evaluation.score_run --run $R --windows $WIN --ref-windows $RDEV \
  --set zh2en > $R/score.log 2>&1; log "score rel rc=$? $(tail -c 300 $R/score.log)"
mkdir -p $SP/talker
GSTAR_DATA=data/ami_multispk_si_v1 $PYE -m experiments.mist_w3po_v1.evaluation.gstar_talker_v6_unk prep --hyp $R/hyp.jsonl \
  --windows $WIN --causal --blocks --preset lowlat --session zh_mix_v8 --diar nemotron_lowlat_zh8 --sep ' ' \
  --out $SP/talker_in_all > $SP/prep.log 2>&1 && python3 reports/gstar_si_v7/drop_long_streams.py $SP/talker_in_all \
  $SP/talker_in 400 >> $SP/prep.log 2>&1; log "prep rc=$? $(tail -c 300 $SP/prep.log)"
CUDA_VISIBLE_DEVICES=$CARD VLLM_USE_FLASHINFER_SAMPLER=0 OMP_NUM_THREADS=16 $PYV -m experiments.uniss_qwen3_omni_v1.evaluation.si_infer \
  --model $T1C --direction zh2en --window 1 --output $SP/talker --text-policy file --blocks-file $SP/talker_in/blocks.jsonl \
  --items-file $SP/talker_in/items.jsonl --speed-policy fixed --history-ban 8 --max-model-len 32768 --gpu-mem 0.1 \
  --shard 0 --num-shards 1 > $SP/talk0.log 2>&1; log "talker rc=$? $(tail -c 300 $SP/talk0.log | tr '\n' ' ')"
cat $SP/talker/part_*.jsonl > $SP/talker/results.jsonl && $PYE scripts/gpu_queue/p80_speechonly.py $SP/talker > $SP/asrprep.log 2>&1
log "speechonly rc=$?"
CUDA_VISIBLE_DEVICES=$CARD $PYE -m evaluation.asr_transcribe --input $SP/talker/results_speechonly.jsonl \
  --output $SP/talker/asr_speech.jsonl --device cuda:0 --batch-size 8 > $SP/asr.log 2>&1; log "asr rc=$?"
CUDA_VISIBLE_DEVICES= $PY -m experiments.mist_w3po_v1.evaluation.asr_hyp $R/hyp.jsonl $SP/talker/asr_speech.jsonl \
  $R/hyp_asr.jsonl && CUDA_VISIBLE_DEVICES= $PY -m experiments.mist_w3po_v1.evaluation.score_run --run $R --windows $WIN \
  --ref-windows $RDEV --set zh2en --asr $R/hyp_asr.jsonl > $R/score_asr.log 2>&1; log "score asr rc=$? $(tail -c 300 $R/score_asr.log)"
log "E9_SMOKE_DONE"
