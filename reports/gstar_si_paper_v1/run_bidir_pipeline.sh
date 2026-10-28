#!/usr/bin/env bash
# Plan v9 training chain, after run_zh_data_final.sh:
#   gate   A8 (diarizer on the Chinese meetings): for every split, mean zh DER - en DER <= 10 points, else stop
#   rows   en->zh rows (e2e_gstar_v1) + zh->en rows (e2e_gstar_zh2en_v1) -> e2e_gstar_bidir_v1/rows_train.jsonl
#   O2     bidirectional offline MIST, 2,000 steps from the original Thinker
#   ABC    4,000 training windows per direction labelled by O2 (--policy oracle, reads every 0.5/1/1/2 s), 8 shards each
#   S2     MIST-Simul-ABC from O2 (new LoRA), 3,000 steps on the ABC rows + offline and single-speaker rows
#   O2n    O2 without the speaker stream (NOSD), 2,000 steps
# Every stage writes new directories only. Launch detached (setsid nohup); check with bash -n.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
PYE=/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval/bin/python
PYM=/opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python
D=data/ami_multispk_si_v1; EV=eval_outputs/uniss_qwen3_omni_v1
TS=${TS:-20261029}
O2=thinker_gstar_o2_${TS}; S2=thinker_gstar_s2abc_${TS}; O2N=thinker_gstar_o2nosd_${TS}
ABC=${D}/e2e_gstar_abc_v1
log() { echo "[$(TZ=Europe/Prague date +%H:%M:%S)] $*"; }
for p in ${D}/e2e_gstar_bidir_v1 ${ABC} ${EV}/${O2} ${EV}/${S2} ${EV}/${O2N}; do
  [ -e "$p" ] && { log "拒绝覆盖已有输出: $p"; exit 1; }
done

until grep -q -E "ZH_DATA_DONE|failed" logs/zh_data_final.log; do sleep 30; done
grep -q ZH_DATA_DONE logs/zh_data_final.log || { log "zh data failed"; exit 1; }
${PYE} - <<'PY' || { log "A8_FAIL: the diarizer tracks the Chinese meetings much worse than the English ones"; exit 2; }
import json, sys
s = json.load(open("data/ami_multispk_si_v1/zh_mix_v7/audit_diar.json"))["summary"]
gaps = {k: v["der_zh"] - v["der_en"] for k, v in s.items() if isinstance(v, dict)}
print("A8 gaps", gaps)
sys.exit(0 if all(g <= 10 for g in gaps.values()) else 1)
PY
log "A8 passed"

mkdir -p ${D}/e2e_gstar_bidir_v1
${PYE} - <<'PY' || { log "rows failed"; exit 1; }
import json, random
d = "data/ami_multispk_si_v1"
rows = [l for f in (f"{d}/e2e_gstar_v1/rows_train.jsonl", f"{d}/e2e_gstar_zh2en_v1/rows_train.jsonl") for l in open(f)]
random.Random(20261029).shuffle(rows)
with open(f"{d}/e2e_gstar_bidir_v1/rows_train.jsonl", "x") as fh:
    fh.writelines(rows)
print(len(rows))
PY
log "bidir rows: $(wc -l < ${D}/e2e_gstar_bidir_v1/rows_train.jsonl)"

train() {   # name, extra env
  log "train $1"
  env RUN_ID=$1 "${@:2}" bash experiments/uniss_qwen3_omni_v1/scripts/run_thinker_gstar.sh > logs/multispk/$1.log 2>&1
  grep -q THINKER_GSTAR_DONE logs/multispk/$1.log || { log "FAILED $1"; return 1; }
  log "done $1"
}
train ${O2} ROWS=${D}/e2e_gstar_bidir_v1/rows_train.jsonl TRAIN_ITERS=2000 || exit 1

PYTHONPATH=. ${PYE} -m experiments.uniss_qwen3_omni_v1.data_clean.build_ami_gstar_abc_rows pick --out ${ABC} --n 4000 \
  > logs/multispk/abc_pick.log 2>&1 || { log "abc pick failed"; exit 1; }
for d in en2zh zh2en; do
  log "oracle labels ${d}"
  for k in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES=$k PYTHONPATH=.:third_party/Megatron-LM ${PYM} -m experiments.uniss_qwen3_omni_v1.evaluation.gstar_si_stream_online \
      --thinker ${EV}/${O2}/thinker_merged --windows ${ABC}/windows_${d}.jsonl --out ${ABC}/dump_${d}_s${k}.jsonl \
      --policy oracle --dir ${d} --window-set 0.5,1,1,2 --max-new 160 --post-key lowlat --batch 16 \
      --shard $k --num-shards 8 > logs/multispk/abc_${d}_s${k}.log 2>&1 &
  done
  wait
  log "dumped ${d}: $(cat ${ABC}/dump_${d}_s*.jsonl | wc -l) windows"
done
PYTHONPATH=. ${PYE} -m experiments.uniss_qwen3_omni_v1.data_clean.build_ami_gstar_abc_rows rows --out ${ABC} \
  --dumps "${ABC}/dump_*_s*.jsonl" --offline-rows ${D}/e2e_gstar_bidir_v1/rows_train.jsonl \
  > logs/multispk/abc_rows.log 2>&1 || { log "abc rows failed"; exit 1; }
log "abc rows: $(cat logs/multispk/abc_rows.log | tail -1)"

train ${S2} BASE=${EV}/${O2}/thinker_merged INIT_MERGED=1 ROWS=${ABC}/rows_train.jsonl TRAIN_ITERS=3000 || exit 1
train ${O2N} NOSD=1 ROWS=${D}/e2e_gstar_bidir_v1/rows_train.jsonl TRAIN_ITERS=2000 || exit 1
log "BIDIR_PIPELINE_DONE"
