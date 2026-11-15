#!/usr/bin/env bash
# v11 P2 (reports/mist_w3po_v1/v11_p2_fluency_pa_prereg.zh-CN.md): the fluency-constrained PA Talker (PA-F), one chain
# per direction, both chains in parallel:
#   1. wait for the rows (fluency_rows --rate-frac 0.5 --require-feasible) -> stop the chain if they are missing;
#   2. training = run_talker_crf.sh MODE=plem verbatim (T1c start, 1200 steps, 4 cards) on 4 cards that have room for
#      the PL-EM peak (non-placeholder memory + PEAK_GIB <= 135 GiB, no other training on them; stacking on decodes is
#      allowed), placeholders stopped by takeover_train.sh; the two chains never pick the same card (claim files);
#      export: zh2en iteration 1200 (= export), en2zh iteration 400 (export_i0400): PA's own choices, no new selection;
#   3. dev = E2's rlplem chain with only the Talker swapped: B's dev text (jdev_b{z,e}_<RT1>_rl), joint_orch (zh2en via
#      joint_orch_plemx.py --plem-x), --talker plem, both TTS seeds, gate_2seed vs B+PA (<N>__rlplem, _tts2);
#      then chunk-protocol perword latency (as e2_baseline_plem.sh lat()), block-start latency (lat_units_dump),
#      pauses (talker_block_pauses) and UTMOS/SIM seed 1 (talker_quality) for PA-F and B+PA.
# Marker logs/mist_v11/p2/P2_DEV_<dir>_DONE. Test runs are NOT here (only after the dev gate; separate launcher).
# Refuses to overwrite run dirs. Checked with bash -n and DRY_RUN=1 only; never test-run.
#   [DIRS="zh2en en2zh"] [DRY_RUN=1] bash experiments/mist_w3po_v1/scripts/v11_p2_plemf.sh
# Run detached: setsid nohup bash ... > logs/mist_v11/p2/plemf.log 2>&1 < /dev/null 9>&- &
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
. experiments/mist_w3po_v1/scripts/joint_lib.sh
DIRS=${DIRS:-"zh2en en2zh"}
P2=logs/mist_v11/p2
RW=eval_outputs/mist_w3po_v1/talker_crf/rows_fluency
B2=$JO/joint_v8h6k_20261113T073658Z/merged_s06000/thinker_merged
RT1=joint_v8h6k_20261113T073658Z_s06000
T1C=eval_outputs/uniss_qwen3_omni_v1/si_t1c_20261021T181440Z/autoext/export_1500/uniss
PEAK_GIB=${PEAK_GIB:-59}
QDIR=$P2/quality
mkdir -p $P2 $QDIR

room_cards() {   # <exclude cards "a b"> -> eligible cards (non-placeholder MiB + peak <= cap, no training), least used first
  local ex=" $1 " tmp=$P2/.others.$BASHPID tmu=$P2/.use.$BASHPID
  others_on 0,1,2,3,4,5,6,7 > $tmp 2>/dev/null
  for c in 0 1 2 3 4 5 6 7; do
    case "$ex" in *" $c "*) continue;; esac
    awk -v c=$c '$1 == c && $4 != "placeholder" {u += $3} END {print u + 0}' $tmp > $tmu
    u=$(cat $tmu)
    istr=0
    for p in $(awk -v c=$c '$1 == c && $4 != "placeholder" {print $2}' $tmp); do
      tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | grep -qE "pretrain_|torch.distributed|train_crf_sft" && istr=1
    done
    [ $istr = 0 ] && [ $((u + PEAK_GIB * GIB)) -le $CAP_MIB ] && echo "$u $c"
  done | sort -n | awk '{print $2}'
}

train() {   # <dir> -> echo export dir on success
  local d=$1 rows=$RW/plemf_$1.pt rid tl cards ex other port
  until [ -e $rows ] && [ -e $rows.json ]; do
    grep -qE "INFEASIBLE|Traceback|Error" $P2/rows_$d.log 2>/dev/null && { log "P2_STOP $d: rows failed/infeasible ($(tail -n 1 $P2/rows_$d.log | cut -c1-300))"; return 1; }
    sleep 30
  done
  log "$d rows: $(head -c 600 $rows.json | tr '\n' ' ')"
  rid=talker_plemf_${d}_$(date -u +%Y%m%dT%H%M%SZ)
  [ $d = zh2en ] && { other=en2zh; port=29831; } || { other=zh2en; port=29832; }
  t0=$(date +%s)
  while :; do
    ex=$(cat $P2/claim_$other 2>/dev/null | tr ',' ' ')
    cards=$(room_cards "$ex" | head -n 4 | sort -n | paste -sd, -)
    if [ $(echo "$cards" | tr ',' '\n' | grep -c .) = 4 ]; then
      ( set -o noclobber; echo "$cards" > $P2/claim_$d ) 2>/dev/null && break
    fi
    [ $(( ($(date +%s) - t0) % 1800 )) -lt 30 ] && log "$d: waiting for 4 cards with room (now: '$cards')"
    sleep 30
  done
  tl=$P2/train_$rid.log
  log "$d: training $rid on cards $cards ($(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | tr '\n' ' '))"
  CARDS=$cards LAUNCH="setsid nohup env RUN_ID=$rid MODE=plem ROWS=$rows CARDS=$cards MASTER_PORT=$port bash $SC/run_talker_crf.sh > $tl 2>&1 < /dev/null 9>&- &" \
    bash $SC/takeover_train.sh >> $P2/takeover_$d.log 2>&1
  until [ -e $tl ] && grep -qE "TALKER_CRF_DONE|TALKER_CRF_FAILED|TALKER_CRF_EXPORT_FAILED" $tl; do sleep 30; done
  command mv -f $P2/claim_$d $P2/claim_$d.released_$(date -u +%Y%m%dT%H%M%SZ)
  grep -q TALKER_CRF_DONE $tl || { log "P2_STOP $d: training: $(grep -E 'TALKER_CRF_' $tl | tail -n 1)"; return 1; }
  local exp=eval_outputs/mist_w3po_v1/talker_crf/$rid
  if [ $d = en2zh ]; then
    CUDA_VISIBLE_DEVICES="" $PY experiments/uniss_qwen3_omni_v1/training/export_e2.py \
      --checkpoint-dir checkpoints/mist_w3po_v1/$rid/iter_0000400 --base $T1C --output $exp/export_i0400 \
      > $P2/export_${d}_i0400.log 2>&1 || { log "P2_STOP $d: export i0400 failed"; return 1; }
    echo $exp/export_i0400 > $P2/EXPORT_$d
  else
    echo $exp/export > $P2/EXPORT_$d
  fi
  log "$d: PA-F export $(cat $P2/EXPORT_$d)"
}

lat_all() {   # <dir> <run name>: perword (E2 lat) + blockstart (lat_units_dump) on the chunk protocol, dev
  local d=$1 n=$2 O=$TE/${2%_tts2} f
  mkdir -p $O $P2/lat
  f=$O/lat_chunk_$n.json
  if [ $d = zh2en ]; then
    [ -e $f ] || CUDA_VISIBLE_DEVICES= $PY -m experiments.mist_w3po_v1.evaluation.speech_latency_lit --run $DEC/$n \
      --talker $SPD/$n/talker --win $WZ_DEVC --out $f > $f.log 2>&1 || log "perword LAAL failed: $n"
    for m in perword blockstart; do
      [ -e $P2/lat/${m}_$n.json ] || LATU_MODE=$m CUDA_VISIBLE_DEVICES= $PY -m experiments.mist_w3po_v1.evaluation.lat_units_dump \
        experiments.mist_w3po_v1.evaluation.speech_latency_lit --run $DEC/$n --talker $SPD/$n/talker --win $WZ_DEVC \
        --out $P2/lat/${m}_$n.json > $P2/lat/${m}_$n.log 2>&1 || log "$m LAAL failed: $n"
    done
  else
    [ -e $f ] || CUDA_VISIBLE_DEVICES= $PY -m experiments.mist_w3po_v1.evaluation.en2zh_lat --ref-win $RE_DEV --work $f.work \
      --run $DEC/$n --talker $SPD/$n/talker --win $WE_DEVC --out $f > $f.log 2>&1 || log "perword LAAL failed: $n"
    for m in perword blockstart; do
      [ -e $P2/lat/${m}_$n.json ] || LATU_MODE=$m CUDA_VISIBLE_DEVICES= $PY -m experiments.mist_w3po_v1.evaluation.lat_units_dump \
        experiments.mist_w3po_v1.evaluation.en2zh_lat --ref-win $RE_DEV --work $P2/lat/${m}_$n.json.work --run $DEC/$n \
        --talker $SPD/$n/talker --win $WE_DEVC --out $P2/lat/${m}_$n.json > $P2/lat/${m}_$n.log 2>&1 || log "$m LAAL failed: $n"
    done
  fi
}

quality() {   # <run name>: UTMOS / SIM on the freest card (small GPU job)
  local n=$1 c
  [ -e $QDIR/$n/quality.json ] && return 0
  c=$(nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv,noheader,nounits | awk -F', ' '{print $2 - $3, $1}' | sort -rn | head -n 1 | awk '{print $2}')
  mkdir -p $QDIR/$n
  CUDA_VISIBLE_DEVICES=$c $PY -m experiments.mist_w3po_v1.talker.talker_quality --talker $SPD/$n/talker \
    --items $SPD/$n/talker_in/items.jsonl --out $QDIR/$n/quality.json > $QDIR/$n/quality.log 2>&1 || log "quality failed: $n"
}

dev() {   # <dir> <export> <port0>
  local d=$1 x=$2 p=$3 N R C
  [ $d = zh2en ] && N=jdev_bz_$RT1 || N=jdev_be_$RT1
  R=$P2/dev_$d
  if [ $d = zh2en ]; then
    C="$PY $SC/joint_orch_plemx.py --plem-x $x --dir zh2en --set dev --base $B2 --run $R --name ${N}__rlplemf --ckpt - --speech-from ${N}_rl --talker plem --window 1 --cap135 --scorer-port0 $p --gate-ref ${N}__rlplem ${N}__rlplem_tts2"
  else
    C="$JORCH --dir en2zh --set dev --base $B2 --run $R --name ${N}__rlplemf --ckpt - --speech-from ${N}_rl --talker plem --plem-export $x --window 1 --cap135 --scorer-port0 $p --gate-ref ${N}__rlplem ${N}__rlplem_tts2"
  fi
  if [ -n "${DRY_RUN:-}" ]; then echo "dev $d: $C"; $C --dry 2>&1 | tail -n 2; return 0; fi
  [ -e $R ] && { log "P2_STOP $d: $R exists (not overwritten)"; return 1; }
  setsid nohup $C < /dev/null > $R.launch.log 2>&1 9>&- &
  log "$d dev speech orchestrator pid $!"
  wait_done $R
  [ -z "$(orch_failed $R)" ] || { log "P2_STOP $d: dev orchestrator failed ($(tail -n 1 $R/orch.log | cut -c1-200))"; return 1; }
  for n in ${N}__rlplemf ${N}__rlplemf_tts2 ${N}__rlplem ${N}__rlplem_tts2; do lat_all $d $n; done
  mkdir -p $P2/pauses
  for n in ${N}__rlplemf ${N}__rlplemf_tts2 ${N}__rlplem ${N}__rlplem_tts2; do
    [ -e $P2/pauses/$n.json ] || CUDA_VISIBLE_DEVICES= $PY -m experiments.mist_w3po_v1.evaluation.talker_block_pauses \
      --talker $SPD/$n/talker > $P2/pauses/$n.json 2> $P2/pauses/$n.log || log "pauses failed: $n"
  done
  quality ${N}__rlplemf
  quality ${N}__rlplem
  ( set -o noclobber; date -u +%FT%TZ > $P2/P2_DEV_${d}_DONE ) || true
  log "P2_DEV_${d}_DONE (gate: $(ls $TE/${N}__rlplemf/gate_*.json 2>/dev/null))"
}

chain() {   # <dir> <port0>
  local d=$1 x
  if [ -n "${DRY_RUN:-}" ]; then
    echo "$d: rows $RW/plemf_$d.pt; room now: $(room_cards '' | tr '\n' ' ')"
    [ $d = zh2en ] && x=eval_outputs/mist_w3po_v1/talker_crf/talker_plem_20261111T001759Z/export \
      || x=eval_outputs/mist_w3po_v1/talker_crf/talker_plem_en2zh_20261114T045019Z/export_i0400
    dev $d $x $2; return 0
  fi
  if [ -s $P2/EXPORT_$d ]; then x=$(cat $P2/EXPORT_$d); log "$d: export exists: $x"
  else train $d || return 1; x=$(cat $P2/EXPORT_$d); fi
  dev $d $x $2
}

if [ -n "${DRY_RUN:-}" ]; then
  k=0; for d in $DIRS; do chain $d $((19960 + 10 * k)); k=$((k + 1)); done; exit 0
fi
log "v11 P2 PA-F: dirs $DIRS, peak $PEAK_GIB GiB"
PIDS=""; k=0
for d in $DIRS; do
  ( chain $d $((19960 + 10 * k)) ) &
  PIDS="$PIDS $!"; k=$((k + 1)); sleep 90
done
for p in $PIDS; do wait $p; done
log "P2_DEV_CHAINS_EXITED"
