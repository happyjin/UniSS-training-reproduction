#!/usr/bin/env bash
# D0 (reports/mist_w3po_v1/robust_wave1_prereg.zh-CN.md): baselines on the OpenBroadcast-SI DEV twin (d0_build.sh), run
# exactly like the test runs, through scripts/voicecond_bench_dev_orch.py (voicecond_bench_orch with the dev roots):
#   B    text orchestrator (joint_v8h6k merged s06000 + identity, window 1, --talker none, --cap135) -> T1c orchestrator
#        (--speech-from obsidev_b_<c>_<d> --talker t1c: --speed-policy fixed --history-ban 8, two TTS seeds) ->
#        bench_dev_eval.  Names obsidev_b_<c>_<d> (text), obsidev_b_<c>_<d>__t1c{,_tts2}
#   I1C  Idea 1 C (denoising-encoder control, eval_outputs/mist_w3po_v1/voicecond_idea1/m2/ctrl/encoder.pt): once B's
#        T1c prep wrote talker_in/items.jsonl (unchanged >= 120 s): HDemucs vocals of B's prompts
#        (idea1_bench_tokens vocals_gpu --items, one small GPU job via d0_gpu_lib.d0_gpu_one; read by idea1_tokens, unused
#        in ctrl mode) -> idea1_tokens --mode ctrl (CPU) -> sidecar json {prompt stem: 32 ids} (sha1 of every prompt
#        re-checked) -> T1c orchestrator with --voice-override <sidecar> --speech-from obsidev_b_<c>_<d> -> bench_dev_eval.
#        Names obsidev_i1c_<c>_<d>__t1c{,_tts2}
# Sets: id2 / clean x zh2en / en2zh, chains in parallel. Every orchestrator gets SI_KV_CACHE_GIB=7.73
# ORCH_TALK_PAIR_GAP_S=0 ORCH_LEDGER_VLLM_GAP_S=10 and --exclude none (stacks on any card with real free memory, <= 135
# GiB per card; placeholders stopped by PID by the orchestrators), and is started under one flock (8, closed for the
# child; 9 closed too), 60 s apart. bench_dev_eval: at most 3 at once (3 slot locks, fd closed for the child) and only
# while the 1-min load is < 130. Refuses existing run dirs; existing outputs are skipped.
# Orchestrator dirs logs/mist_w3po_v1/wave1/d0/runs/<name>; markers logs/mist_w3po_v1/wave1/d0/
# D0_<B|I1C>_<c>_<d>_{DONE,FAILED} and D0_RUNS_DONE at the end.
#   [DRY_RUN=1] [SETS="id2:zh2en ..."] bash experiments/mist_w3po_v1/scripts/d0_runs.sh
# Detached: setsid nohup bash ... > logs/mist_w3po_v1/wave1/d0/runs.log 2>&1 < /dev/null 8>&- 9>&- &
# Checked with bash -n and DRY_RUN=1 only.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
. experiments/mist_w3po_v1/scripts/joint_lib.sh
. experiments/mist_w3po_v1/scripts/d0_gpu_lib.sh
J=/opt/dlami/nvme/jasonleeeli
export PATH=$J/venvs/vllm_omni_030/bin:$PATH HF_HOME=$J/cache/huggingface
export SI_KV_CACHE_GIB=7.73 ORCH_TALK_PAIR_GAP_S=0 ORCH_LEDGER_VLLM_GAP_S=10
VPY=$J/venvs/vllm_omni_030/bin/python; EPY=$J/conda_envs/uniss-eval/bin/python
L=logs/mist_w3po_v1/wave1/d0
R0=$L/runs
O=eval_outputs/mist_w3po_v1/wave1/d0
VOC=data/open_voicecond_v1/bench_dev/idea1_vocals
EV=eval_outputs/mist_w3po_v1/voicecond_bench
CK=eval_outputs/mist_w3po_v1/voicecond_idea1/m2/ctrl/encoder.pt
B2=$JO/joint_v8h6k_20261113T073658Z/merged_s06000/thinker_merged
IDN=$JO/joint_v8h6k_20261113T073658Z/merged_s06000/identity
VO="$PY scripts/voicecond_bench_dev_orch.py"
SETS=${SETS:-"id2:zh2en id2:en2zh clean:zh2en clean:en2zh"}
mkdir -p $R0 $O
orch() {   # <run dir> <orchestrator args...>: start under the launch lock, wait for DONE; rc 0 = no failed job
  local R=$1
  shift
  if [ -n "${DRY_RUN:-}" ]; then
    d0_log "[dry] $(basename $R): $VO $* --run $R"
    $VO "$@" --run $R.dry --dry 2>&1 | tail -2
    return 0
  fi
  if [ ! -e $R ]; then
    (
      flock 8
      [ -e $R ] || { setsid nohup $VO "$@" --run $R < /dev/null > $R.launch.log 2>&1 8>&- 9>&- &
                     echo $! > $R.pid; d0_log "$(basename $R) orchestrator pid $!"; sleep 60; }
    ) 8> $L/launch.lock
  else
    d0_log "$(basename $R) exists: only waited for"
  fi
  until [ -e $R/events/DONE ]; do   # an orchestrator that exits without DONE (e.g. refused its arguments) = failed
    if [ -s $R.pid ] && ! kill -0 $(cat $R.pid) 2>/dev/null; then
      sleep 5; [ -e $R/events/DONE ] && break
      d0_log "$(basename $R) orchestrator exited without DONE: $(tail -2 $R.launch.log | tr '\n' ' ' | cut -c1-300)"
      return 1
    fi
    sleep 30
  done
  [ -z "$(orch_failed $R)" ] || { d0_log "$(basename $R) failed ($(tail -1 $R/orch.log | cut -c1-200))"; return 1; }
}
ev() {   # <cond> <dir> <name>: bench_dev_eval, <= 3 at once, load < 130
  local c=$1 d=$2 n=$3 s fd rc
  [ -e $EV/$n/metrics.json ] && { d0_log "$n metrics.json exists"; return 0; }
  if [ -n "${DRY_RUN:-}" ]; then d0_log "[dry] bench_dev_eval --cond $c --dir $d --name $n"; return 0; fi
  while :; do
    for s in 0 1 2; do
      exec {fd}> $L/eval_slot_$s.lock
      if flock -n $fd; then
        while [ "$(cut -d' ' -f1 /proc/loadavg | cut -d. -f1)" -ge 130 ]; do sleep 60; done
        d0_log "eval $n start (slot $s, load $(cut -d' ' -f1 /proc/loadavg))"
        PYTHONPATH=. $PY -m experiments.mist_w3po_v1.voicecond.bench_dev_eval --cond $c --dir $d --name $n \
          > $L/eval_$n.log 2>&1 {fd}>&- 8>&- 9>&-
        rc=$?
        flock -u $fd; exec {fd}>&-
        d0_log "eval $n rc=$rc: $(tail -1 $L/eval_$n.log | cut -c1-400)"
        return $rc
      fi
      exec {fd}>&-
    done
    sleep 60
  done
}
sidecar() {   # <tokens jsonl> <items jsonl> <out json>: {prompt stem: ids}, every item covered, prompt sha1 re-checked
  $PY - "$@" <<'EOF'
import hashlib, json, sys
from pathlib import Path
tok, items, out = sys.argv[1:4]
R = Path("/opt/dlami/nvme/jasonleeeli/projects/UniSS")
T = {r["key"]: r for r in map(json.loads, open(tok, encoding="utf-8"))}
side = {}
for it in map(json.loads, open(items, encoding="utf-8")):
    p = Path(it["voice_audio"]); p = p if p.is_absolute() else R / p
    r = T[p.stem]
    assert r["mode"] == "ctrl" and hashlib.sha1(p.read_bytes()).hexdigest() == r["sha1"], p
    assert len(r["ids"]) == 32 and all(0 <= t < 4096 for t in r["ids"])
    side[p.stem] = r["ids"]
assert set(side) == set(T), "token rows without an item"
with open(out, "x", encoding="utf-8") as fh:
    json.dump(side, fh)
print(json.dumps({"out": out, "items": len(side)}))
EOF
}
chain_b() {   # <cond> <dir>
  local c=$1 d=$2 n=obsidev_b_$1_$2
  orch $R0/$n --cond $c --dir $d --base $B2 --name $n --ckpt $IDN --talker none --window 1 --cap135 --exclude none \
    || { date > $L/D0_B_${c}_${d}_FAILED; return 1; }
  orch $R0/${n}__t1c --cond $c --dir $d --base $B2 --name ${n}__t1c --ckpt - --speech-from $n --talker t1c --window 1 \
    --cap135 --exclude none || { date > $L/D0_B_${c}_${d}_FAILED; return 1; }
  [ -n "${DRY_RUN:-}" ] && { ev $c $d ${n}__t1c; return 0; }
  ev $c $d ${n}__t1c && date > $L/D0_B_${c}_${d}_DONE || { date > $L/D0_B_${c}_${d}_FAILED; return 1; }
}
chain_c() {   # <cond> <dir>
  local c=$1 d=$2 b=obsidev_b_$1_$2 n=obsidev_i1c_$1_$2__t1c
  local IT=$SPD/${b}__t1c/talker_in/items.jsonl TK=$O/tokens_c_${c}_$d.jsonl SC=$O/sidecar_c_${c}_$d.json
  if [ -n "${DRY_RUN:-}" ]; then
    d0_log "[dry] $n: wait $IT (>= 120 s unchanged) -> vocals_gpu -> $VOC/${c}_$d -> idea1_tokens --mode ctrl --ckpt $CK -> $TK -> $SC"
    [ -e $SC ] && orch $R0/$n --cond $c --dir $d --voice-override $SC --base $B2 --name $n --ckpt - --speech-from $b --talker t1c \
      --window 1 --cap135 --exclude none
    ev $c $d $n
    return 0
  fi
  until { [ -s $IT ] && [ $(( $(date +%s) - $(stat -c %Y $IT) )) -ge 120 ]; } || [ -e $L/D0_B_${c}_${d}_FAILED ]; do sleep 60; done
  [ -s $IT ] || { d0_log "$n: B failed before its talker inputs"; date > $L/D0_I1C_${c}_${d}_FAILED; return 1; }
  d0_log "$n: B talker inputs ready ($(wc -l < $IT) items)"
  NEED_LOG=$L/vocals_${c}_$d.log
  d0_gpu_one 12000 $L/vocals_${c}_$d.DONE env PYTHONPATH=. $EPY -m experiments.mist_w3po_v1.voicecond.idea1_bench_tokens vocals_gpu \
    --items $IT --out $VOC/${c}_$d > $NEED_LOG 2>&1
  grep -q "rc=0" $L/vocals_${c}_$d.DONE || { d0_log "$n vocals failed"; date > $L/D0_I1C_${c}_${d}_FAILED; return 1; }
  [ -e $TK ] || PYTHONPATH=. OMP_NUM_THREADS=16 $VPY -m experiments.mist_w3po_v1.voicecond.idea1_tokens --mode ctrl --ckpt $CK \
    --items $IT --vocals $VOC/${c}_$d --out $TK --device cpu > $L/tokens_c_${c}_$d.log 2>&1 \
    || { d0_log "$n tokens failed"; date > $L/D0_I1C_${c}_${d}_FAILED; return 1; }
  [ -e $SC ] || sidecar $TK $IT $SC > $L/sidecar_c_${c}_$d.log 2>&1 \
    || { d0_log "$n sidecar failed: $(tail -1 $L/sidecar_c_${c}_$d.log)"; date > $L/D0_I1C_${c}_${d}_FAILED; return 1; }
  d0_log "$n: $(tail -1 $L/tokens_c_${c}_$d.log | cut -c1-200); $(tail -1 $L/sidecar_c_${c}_$d.log)"
  orch $R0/$n --cond $c --dir $d --voice-override $SC --base $B2 --name $n --ckpt - --speech-from $b --talker t1c --window 1 \
    --cap135 --exclude none || { date > $L/D0_I1C_${c}_${d}_FAILED; return 1; }
  ev $c $d $n && date > $L/D0_I1C_${c}_${d}_DONE || { date > $L/D0_I1C_${c}_${d}_FAILED; return 1; }
}
[ -n "${DRY_RUN:-}" ] || [ -e $L/D0_BUILD_DONE ] || { d0_log "D0_BUILD_DONE missing"; exit 1; }
[ -e $CK ] || { d0_log "missing $CK"; exit 1; }
for x in $SETS; do
  c=${x%%:*}; d=${x##*:}
  for f in data/open_voicecond_v1/bench_dev/$c/$d/refs_windows.jsonl $DD/windows_obsidev_${c}_${d}_c2g06_v1/windows_test_online_$d.jsonl \
           $DD/windows_obsidev_${c}_${d}_c2g06_chunk_v1/windows_dev_online.jsonl $B2/sd_proj.pt $IDN; do
    [ -e $f ] || { d0_log "missing $f"; [ -n "${DRY_RUN:-}" ] || exit 1; }
  done
done
if [ -n "${DRY_RUN:-}" ]; then
  for x in $SETS; do chain_b ${x%%:*} ${x##*:}; chain_c ${x%%:*} ${x##*:}; done
  exit 0
fi
d0_log "D0 runs start: $SETS"
P=""
for x in $SETS; do
  c=${x%%:*}; d=${x##*:}
  chain_b $c $d & P="$P $!"
  chain_c $c $d & P="$P $!"
  sleep 5
done
for p in $P; do wait $p || true; done
ok=1
for x in $SETS; do for a in B I1C; do [ -e $L/D0_${a}_${x%%:*}_${x##*:}_DONE ] || ok=0; done; done
[ $ok = 1 ] && date > $L/D0_RUNS_DONE || date > $L/D0_RUNS_FAILED
d0_log "D0 runs exit (ok=$ok)"
