#!/usr/bin/env bash
# Paper v6 (MIST), v2 of run_v6_followups.sh (stage B now follows the dec4 queue): everything that follows the decodes and trainings, run automatically and idempotently; every stage
# writes new files only and runs once (a marker under logs/multispk/v6/fu/). Checks every 5 minutes; exits when all done.
#   A  now: AliMeeting reference-window speech (M07), single-stream Talker rate, batch invariance (T13) -> queue sp1
#   B  dec4 queue and the remaining dec3x jobs done: after_dec.sh; then the whole-meeting speech of the online runs (M03) -> queue sp2
#   C  a v6 model merged: its dec2 decodes -> queue dec2_<what>
#   D  online speech done: listener-side latency (one channel / one per speaker)
#   E  dec2 / speech outputs: merge and score (after_dec2.sh)
#     setsid nohup bash reports/gstar_si_v6/run_v6_followups2.sh > /dev/null 2>&1 < /dev/null &
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
L=logs/multispk/v6; F=$L/fu; mkdir -p $F
exec >> $L/followups.log 2>&1
log() { echo "[$(date -u +%F' '%T) UTC] $*"; }
EV=eval_outputs/uniss_qwen3_omni_v1; V6=$EV/gstar_v6_20261031; SP=$EV/gstar_v6_speech_20261031
ON=data/online_v6_20261031; J=reports/gstar_si_v6/jobs
PYE=/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval/bin/python
PYV=/opt/dlami/nvme/jasonleeeli/venvs/vllm_omni_030/bin/python
T1C=$EV/si_t1c_20261021T181440Z/autoext/export_1500/uniss
queue() {   # $1 jobs file, $2 log dir
  CARDS=2,3,4,5,6,7 setsid nohup bash reports/gstar_si_v6/run_v6_queue.sh $1 $2 > /dev/null 2>&1 < /dev/null &
  log "queue $1 started"
}
qdone() { grep -q QUEUE_DONE $1/queue.log 2>/dev/null; }

stage_A() {
  [ -e $F/A ] && return
  cat > $J/sp1_systems.json <<EOF
[{"name": "ali_O2", "set": "ali", "hyp": "$EV/gstar_v3_20261029/ali_O2", "windows": "data/alimeeting_si_v1/windows_test_ref.jsonl", "mode": "offline"},
 {"name": "ali_C-ft2", "set": "ali", "hyp": "$EV/gstar_v3_20261029/ali_C-ft2", "windows": "data/alimeeting_si_v1/windows_test_ref.jsonl", "mode": "offline"},
 {"name": "ali_S2dl_d1", "set": "ali", "hyp": "$EV/gstar_v5_20261030/ali_S2dl3k_d1", "windows": "data/alimeeting_si_v1/windows_test_ref.jsonl", "mode": "stream"},
 {"name": "ali_O2_LA2_d1", "set": "ali", "hyp": "$EV/gstar_v3_20261029/ali_O2_LA2_d1", "windows": "data/alimeeting_si_v1/windows_test_ref.jsonl", "mode": "stream"}]
EOF
  python3 reports/gstar_si_v6/make_v6_speech_jobs.py $J/sp1_systems.json > $J/sp1.json.part || return
  python3 - $J/sp1.json.part <<'PY' > $J/sp1.json
import json, sys
jobs = json.load(open(sys.argv[1]))
EV = "eval_outputs/uniss_qwen3_omni_v1"
R = f"{EV}/gstar_v6_20261031/talker_rate"
for dr in ("en2zh", "zh2en"):   # single-stream Talker rate: one long stream alone in the engine
    jobs.append({"id": f"talker_rate_{dr}", "gpus": 1, "vllm": True, "deps": [], "mem": 56000,
                 "cmd": f"CUDA_VISIBLE_DEVICES=$GPU0 PATH=/opt/dlami/nvme/jasonleeeli/venvs/vllm_omni_030/bin:$PATH "
                        f"VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_CACHE_ROOT=/opt/dlami/nvme/jasonleeeli/cache/vllm_omni_030 "
                        f"PYTHONPATH=. OMP_NUM_THREADS=16 /opt/dlami/nvme/jasonleeeli/venvs/vllm_omni_030/bin/python -m "
                        f"experiments.uniss_qwen3_omni_v1.evaluation.si_infer --model {EV}/si_t1c_20261021T181440Z/autoext/export_1500/uniss "
                        f"--direction {dr} --window 1 --output {R}/{dr} --text-policy file --blocks-file {R}/{dr}_blocks.jsonl "
                        f"--items-file {R}/{dr}_items.jsonl --speed-policy fixed --history-ban 8 --max-model-len 32768 "
                        f"--gpu-mem 0.3 --shard 0 --num-shards 1 > {R}/{dr}.log 2>&1"})
print(json.dumps(jobs, indent=1))
PY
  python3 reports/gstar_si_v6/make_v6_jobs.py dec2:t13 > $J/dec2_t13.json || return
  queue $J/sp1.json $L/q_sp1
  queue $J/dec2_t13.json $L/q_dec2_t13
  touch $F/A
}

stage_B() {
  [ -e $F/B ] && return
  qdone $L/q_dec4 || return
  ps -eo args | grep -q "[g]star_multi_v6 --thinker .*jobs/dec3x" && return   # the dec3x jobs still running
  bash reports/gstar_si_v6/after_dec.sh >> $L/after.log 2>&1
  for x in en2zh zh2en ali; do for s in S2dl O2_LA2; do [ -e $V6/on_${x}_$s/hyp.jsonl ] || { log "B: on_${x}_$s not merged yet"; return; }; done; done
  python3 - > $J/sp2_systems.json <<PY
import json
out = []
for st in ("en2zh", "zh2en", "ali"):
    for s in ("S2dl", "O2_LA2"):
        out.append({"name": f"on_{st}_{s}", "set": st, "hyp": "$V6/on_" + st + "_" + s,
                    "windows": "$ON/windows_test_online_" + st + "_sub.jsonl", "mode": "stream"})
print(json.dumps(out, indent=1))
PY
  python3 reports/gstar_si_v6/make_v6_speech_jobs.py $J/sp2_systems.json > $J/sp2.json || return
  queue $J/sp2.json $L/q_sp2
  touch $F/B
}

stage_C() {
  for w in zhchar:thinker_gstar_s2dlchar_zh2en_3k_20261031 seeds:thinker_gstar_s2dl_seed20261101 \
           dlabc:thinker_gstar_s2dlabc_20261031 cturn:thinker_gstar_cturn_20261031; do
    what=${w%%:*}; rid=${w##*:}
    [ -e $F/C_$what ] && continue
    [ -e $EV/$rid/thinker_merged/sd_proj.pt ] || continue
    python3 reports/gstar_si_v6/make_v6_jobs.py dec2:$what > $J/dec2_$what.json || continue
    queue $J/dec2_$what.json $L/q_dec2_$what
    touch $F/C_$what
  done
}

stage_D() {
  [ -e $F/D ] && return
  qdone $L/q_sp2 || return
  qdone $L/q_sp1 || return
  # measured computation: median read of the single-stream timing runs, Talker codec tokens per second
  python3 - > $F/rates.env <<'PY'
import json, re, statistics, glob
V6 = "eval_outputs/uniss_qwen3_omni_v1/gstar_v6_20261031"
def read_ms(name):
    xs = [x for f in glob.glob(f"{V6}/{name}/hyp_s*.jsonl") for l in open(f) for x in json.loads(l).get("read_ms", [])]
    return statistics.median(xs) if xs else 0
def rate(dr):
    codes = sum(b["codes"] for l in open(f"{V6}/talker_rate/{dr}/part_0.jsonl") for b in json.loads(l)["blocks"])
    t = float(re.search(r"talker ([0-9.]+)s", open(f"{V6}/talker_rate/{dr}.log").read()).group(1))
    return codes / t
print(f"RS_EN={read_ms('time_en2zh_S2dl')}\nRS_ZH={read_ms('time_zh2en_S2dl')}\nRLA={read_ms('time_en2zh_O2_LA2')}")
print(f"TOK_EN={rate('en2zh'):.1f}\nTOK_ZH={rate('zh2en'):.1f}")
PY
  . $F/rates.env
  mkdir -p $V6/latency
  for st in en2zh zh2en ali; do
    lang=$([ $st = en2zh ] && echo cmn || echo eng); tok=$([ $st = en2zh ] && echo $TOK_EN || echo $TOK_ZH)
    for s in S2dl O2_LA2; do
      rms=$([ $s = O2_LA2 ] && echo $RLA || ([ $st = en2zh ] && echo $RS_EN || echo $RS_ZH))
      for lay in single per-speaker; do
        out=$V6/latency/${st}_${s}$([ $lay = per-speaker ] && echo _ps).json
        [ -e $out ] || $PYE reports/gstar_si_v6/listener_latency.py --windows $ON/windows_test_online_${st}_sub.jsonl \
          --hyp $V6/on_${st}_$s/hyp.jsonl --talker $SP/on_${st}_$s --read-ms $rms --tok-rate $tok --lang $lang --layout $lay \
          --out $out && log "D: $out"
      done
    done
  done
  touch $F/D
}

stage_E() {
  bash reports/gstar_si_v6/after_dec2.sh >> $L/after.log 2>&1
}

while :; do
  stage_A; stage_B; stage_C; stage_D; stage_E
  if [ -e $F/A ] && [ -e $F/B ] && [ -e $F/D ] && [ -e $F/C_zhchar ] && [ -e $F/C_seeds ] && [ -e $F/C_dlabc ] && [ -e $F/C_cturn ] \
     && [ -e $F/E_all ]; then log "FOLLOWUPS_DONE"; break; fi
  sleep 300
done
