#!/bin/bash
# 第三轮后处理：复现 SimulS2ST-Omni（zeyangsong_share/Qwen2.5_3B_DualCodec）的参考评测。
#
# ！！基座是官方已训练的 SimulS2ST-Omni/offline（含作者 Stage 1+2），不是从零复现，
#     结果只能当「官方 offline 上继续训练」的参考，不可作为复现结论。
#
#   merge                 合并分片、整理流式分段、剔除预热句，并统计各系统覆盖率（CPU）
#   gpu <k> <n> <gpu>     第 k 片（共 n 片）的 ASR / LAAL 词对齐 / UTMOS / SIM-o·ESim·AutoPCP
#   score                 打分。覆盖完整的系统与第二轮 7 个系统同表；覆盖不全的（分片没跑完）
#                         另起一张表，连同官方同档系统一起**只在其覆盖的句子子集上**打分，
#                         避免把缺的句子当空假设拉低 BLEU
set -uo pipefail
E=/fsx/stevehguo/eval
O=${OUT:-$E/out3}
O2=$E/out2
# 新版复现评测时：LABEL 给本轮系统起名；PREV_OUT 指向上一版复现产物（如 out3），一并进主表对照
LABEL=${LABEL:-Repro-Omni}
PREV_OUT=${PREV_OUT:-}
PREV_LABEL=${PREV_LABEL:-Repro-Omni}
PY=/opt/venv/bin/python
C=$E/code/custom/streaming_si/eval
export HF_HOME=$E/hf TORCH_HOME=$E/models/torch_hub PYTHONWARNINGS=ignore TOKENIZERS_PARALLELISM=false
MAN=$E/work/manifests/all.jsonl
WH=$E/models/whisper-large-v3
DSS="realsi_en2zh realsi_zh2en acl6060_en2zh"

# 实际跑出结果的复现系统（按 _merged 下的文件定）
present() { for s in rep_off rep_L1 rep_L2 rep_L3 rep_L4; do [ -s "$O/_merged/$s.jsonl" ] && echo $s; done; }
streams() { present | grep -v rep_off; }

case $1 in
merge)
  mkdir -p $O/_merged
  for L in 1 2 3 4; do
    ls -d $O/rep_L$L/shard* >/dev/null 2>&1 || continue
    $PY $C/stream_segments.py omni --runs "$O/rep_L$L/shard*" --out $O/_seg/rep_L$L
  done
  $PY - "$O" "$E" <<'EOF'
import glob, json, os, sys
O, E = sys.argv[1], sys.argv[2]
man = {json.loads(l)['id']: json.loads(l) for l in open(f'{E}/work/manifests/all.jsonl', encoding='utf-8')}
srcs = {}
off = sorted(glob.glob(f'{O}/rep_off/results.*.jsonl'))
if off:
    srcs['rep_off'] = off
for L in (1, 2, 3, 4):
    p = f'{O}/_seg/rep_L{L}/results.jsonl'
    if os.path.isfile(p):
        srcs[f'rep_L{L}'] = [p]
cov = {}
for name, paths in srcs.items():
    rows, seen = [], set()
    for p in paths:
        for l in open(p, encoding='utf-8'):
            if not l.strip():
                continue
            r = json.loads(l)
            if r['id'].startswith('_warmup_') or r['id'] in seen or r['id'] not in man:
                continue
            seen.add(r['id'])
            r.setdefault('tgt_lang', man[r['id']]['tgt_lang'])
            r.setdefault('source_sec', man[r['id']]['source_sec'])
            rows.append(r)
    with open(f'{O}/_merged/{name}.jsonl', 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    ok = sum(1 for r in rows if r.get('status') in (None, 'ok'))
    cov[name] = dict(n=len(rows), total=len(man), ok=ok, full=len(rows) == len(man),
                     ids=sorted(seen) if len(rows) != len(man) else None)
    print(f'{name}: {len(rows)}/{len(man)} 条（ok {ok}）{"" if cov[name]["full"] else "  <- 覆盖不全，另表打分"}')
json.dump(cov, open(f'{O}/_merged/coverage.json', 'w'), ensure_ascii=False)
EOF
  ;;
gpu)
  K=$2; N=$3; export CUDA_VISIBLE_DEVICES=$4
  mkdir -p $O/_asr $O/_lat $O/_utmos $O/_para
  for s in $(present); do
    PYTHONPATH=$E/code/custom $PY $C/run_asr.py --wav_jsonl $O/_merged/$s.jsonl --manifest $MAN \
      --out $O/_asr/$s.$K.jsonl --model $WH --device cuda:0 --shard $K --num_shards $N --batch 8
  done
  for s in $(streams); do
    $PY $C/latency_eval.py --results $O/_merged/$s.jsonl --manifest $MAN --model $WH \
      --out $O/_lat/$s.$K.jsonl --shard $K --num_shards $N
  done
  for s in $(present); do
    $PY $C/run_utmos.py --wav_jsonl $O/_merged/$s.jsonl --out $O/_utmos/$s.$K.jsonl --shard $K --num_shards $N
  done
  for s in $(present); do
    PYTHONPATH=$E/code/site_para:$E/code/custom $PY $C/run_para_metrics.py --results $O/_merged/$s.jsonl \
      --manifest $MAN --out $O/_para/$s.$K.jsonl --wavlm $E/models/wavlm-base-plus-sv \
      --emotion2vec /fsx/stevehguo/tts_bench/models/emotion2vec_plus_large \
      --autopcp $E/models/autopcp/model.config --w2v $E/models/wav2vec2-large-xlsr-53 \
      --device cuda:0 --batch 8 --shard $K --num_shards $N 2>&1 | grep -E "FAIL|/" | tail -1
  done
  echo GPU_STAGE_DONE
  ;;
score)
  mkdir -p $O/_report $O/logs
  CK=$(ls $E/models/comet/models--Unbabel--wmt22-comet-da/snapshots/*/checkpoints/model.ckpt | head -1)
  j() { ls $1 2>/dev/null | paste -sd, -; }
  # 第二轮系统（中间产物原样复用，不重算）
  sys2() {
    local asr="$(j "$O2/_asr/$2.*.jsonl")"
    [ "$2" = omni_off ] && asr="$(j "$E/out/_asr/omni.*.jsonl")"
    echo "$1=$O2/_merged/$2.jsonl:$asr:$(j "$O2/_utmos/$2.*.jsonl"):$(j "$O2/_lat/$2.*.jsonl"):$(j "$O2/_para/$2.*.jsonl")"
  }
  sys3() { echo "$1=$O/_merged/$2.jsonl:$(j "$O/_asr/$2.*.jsonl"):$(j "$O/_utmos/$2.*.jsonl"):$(j "$O/_lat/$2.*.jsonl"):$(j "$O/_para/$2.*.jsonl")"; }
  sysp() { local P=$PREV_OUT; echo "$1=$P/_merged/$2.jsonl:$(j "$P/_asr/$2.*.jsonl"):$(j "$P/_utmos/$2.*.jsonl"):$(j "$P/_lat/$2.*.jsonl"):$(j "$P/_para/$2.*.jsonl")"; }
  label() { case $1 in rep_off) echo "$LABEL-offline";; rep_L*) echo "$LABEL-L${1#rep_L}";; esac; }

  FULL=(); PART=()
  for s in $(present); do
    if $PY -c "import json,sys; sys.exit(0 if json.load(open('$O/_merged/coverage.json'))['$s']['full'] else 1)"; then
      FULL+=($s); else PART+=($s); fi
  done
  echo "覆盖完整: ${FULL[*]:-无} | 覆盖不全: ${PART[*]:-无}"

  base_args=()
  for pair in "S4-offline s4" "SimulS2ST-Omni-offline omni_off" "S5-streaming s5" \
              "SimulS2ST-Omni-L1 omni_L1" "SimulS2ST-Omni-L2 omni_L2" \
              "SimulS2ST-Omni-L3 omni_L3" "SimulS2ST-Omni-L4 omni_L4"; do
    set -- $pair
    base_args+=(--sys "$(sys2 "$1" "$2")")
  done
  # 上一版复现（中间产物原样复用，不重算），放在本轮系统之前
  if [ -n "$PREV_OUT" ]; then
    for s in rep_off rep_L1 rep_L2 rep_L3 rep_L4; do
      [ -s "$PREV_OUT/_merged/$s.jsonl" ] || continue
      case $s in rep_off) n="$PREV_LABEL-offline";; *) n="$PREV_LABEL-L${s#rep_L}";; esac
      base_args+=(--sys "$(sysp "$n" "$s")")
    done
  fi

  # ---- 主表：第二轮 7 个系统 + 覆盖完整的复现系统，全量清单
  main_args=("${base_args[@]}")
  for s in ${FULL[@]+"${FULL[@]}"}; do main_args+=(--sys "$(sys3 "$(label $s)" "$s")"); done
  g=0
  for ds in $DSS; do
    (CUDA_VISIBLE_DEVICES=$g PYTHONPATH=$E/code/site_comet:$E/code/custom $PY $C/score_eval.py \
      --manifest $E/work/manifests/$ds.jsonl "${main_args[@]}" \
      --source_utmos "$(j "$O2/_utmos/source.*.jsonl")" \
      --comet_model "$CK" --out $O/_report/$ds.json > $O/logs/score.$ds.log 2>&1) &
    g=$((g + 1))
  done
  wait
  for ds in $DSS; do grep -A18 "^$ds" $O/logs/score.$ds.log; done

  # ---- 子集表：覆盖不全的系统，只在其共同覆盖的句子上，与全部对照系统一起打分
  if [ ${#PART[@]} -gt 0 ]; then
    mkdir -p $O/_report_subset $O/work_subset
    $PY - "$O" "$E" "${PART[*]}" <<'EOF'
import json, sys
O, E, part = sys.argv[1], sys.argv[2], sys.argv[3].split()
cov = json.load(open(f'{O}/_merged/coverage.json'))
keep = None
for s in part:
    ids = set(cov[s]['ids'] or [])
    keep = ids if keep is None else (keep & ids)
for ds in ('realsi_en2zh', 'realsi_zh2en', 'acl6060_en2zh'):
    rows = [l for l in open(f'{E}/work/manifests/{ds}.jsonl', encoding='utf-8')
            if l.strip() and json.loads(l)['id'] in keep]
    open(f'{O}/work_subset/{ds}.jsonl', 'w', encoding='utf-8').writelines(rows)
    print(f'子集 {ds}: {len(rows)} 句')
EOF
    sub_args=("${base_args[@]}")
    for s in ${FULL[@]+"${FULL[@]}"} "${PART[@]}"; do sub_args+=(--sys "$(sys3 "$(label $s)" "$s")"); done
    g=0
    for ds in $DSS; do
      if [ ! -s $O/work_subset/$ds.jsonl ]; then g=$((g + 1)); continue; fi
      (CUDA_VISIBLE_DEVICES=$g PYTHONPATH=$E/code/site_comet:$E/code/custom $PY $C/score_eval.py \
        --manifest $O/work_subset/$ds.jsonl "${sub_args[@]}" \
        --source_utmos "$(j "$O2/_utmos/source.*.jsonl")" \
        --comet_model "$CK" --out $O/_report_subset/$ds.json > $O/logs/score_subset.$ds.log 2>&1) &
      g=$((g + 1))
    done
    wait
    for ds in $DSS; do grep -A20 "^$ds" $O/logs/score_subset.$ds.log 2>/dev/null; done
  fi
  echo SCORE_DONE
  ;;
esac
