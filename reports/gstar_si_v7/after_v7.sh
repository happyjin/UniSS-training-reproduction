#!/usr/bin/env bash
# Paper v7 (prereg_v7): merge and score every v7 decode whose shards are complete, queue the speech of the merged
# whole-meeting outputs, and score speech and listener-side lag once it exists. Idempotent: a merged hyp.jsonl, a
# score or a queued job file is never rewritten; incomplete systems are reported and left alone. CPU only.
#   whole meetings   on_<set>_<sys>_rest/hyp.jsonl (v7 decodes) + v6's subset hyp.jsonl -> on_<set>_<sys>_full/hyp.jsonl
#                    -> scores/on_<set>_<sys>_full.json (meeting_score --online); speech of the rest windows (queue
#                    file add/sp_<set>_<sys>.json), merged with v6's subset speech -> speech tcpBLEU and listener lag
#   N-check          nchk_on_en2zh_S2dl_sub -> scores/nchk.json (same scorer as v6's subset score)
#   cascades         <dir>_<OC|CC>_seg -> gstar_cascade_v6 assemble-v6; <set>_Cturn_seg -> gstar_cascade assemble; gstar_score
#   seeds            <set>_<model>_d1 (reference windows) -> gstar_score
#     bash reports/gstar_si_v7/after_v7.sh >> logs/multispk/v7/after.log 2>&1
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
PY=/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval/bin/python
PYV=/opt/dlami/nvme/jasonleeeli/venvs/vllm_omni_030/bin/python
M=experiments.uniss_qwen3_omni_v1.evaluation
EV=eval_outputs/uniss_qwen3_omni_v1; V6=$EV/gstar_v6_20261031; SP6=$EV/gstar_v6_speech_20261031
V7=$EV/gstar_v7_20261101; SP7=$EV/gstar_v7_speech_20261101; SC=$V7/scores; LAT=$V7/latency
D=data/ami_multispk_si_v1; A=data/alimeeting_si_v1; ON=data/online_v6_20261031
Q=logs/multispk/v7/queue2/add; F=logs/multispk/v7/fu
mkdir -p $SC $LAT $SP7 $F
export PYTHONPATH=.
log() { echo "[$(date -u +%F' '%T) UTC] $*"; }
eval "$(sed -n '/^merge() {/,/^}/p' reports/gstar_si_v6/after_dec.sh)"
score() { [ -e $1/score.json ] && return 0; $PY -m $M.gstar_score --windows $2 --hyp $1/hyp.jsonl --post-key $3 ${4:-} --out $1/score.json > $1/score.log 2>&1 && log "$1: $(grep -m1 cpBLEU_session $1/score.json)"; }
mscore() { [ -e $5 ] && return 0; $PY reports/gstar_si_v6/meeting_score.py --dir $1 --windows $2 --hyp $3 --ref-windows $4 $6 --out $5 > $5.log 2>&1 && log "$5: $(tail -n 1 $5.log | cut -c1-200)"; }
REF_EN=$D/e2e_gstar_v1/windows_test_ref.jsonl; REF_ENS=$D/e2e_gstar_stream_v1/windows_test_ref.jsonl
REF_ZH=$D/e2e_gstar_zh2en_v1/windows_test_ref.jsonl; REF_ALI=$A/windows_test_ref.jsonl
ref_of() { [ $1 = en2zh ] && echo $REF_EN || ([ $1 = zh2en ] && echo $REF_ZH || echo $REF_ALI); }
ok=1

# ---- whole meetings: merge, text score, speech jobs, speech score, listener lag
for st in en2zh zh2en ali; do for s in S2dl O2_LA2; do
  R=$V7/on_${st}_${s}_rest; FULL=$V7/on_${st}_${s}_full
  merge $R $R/todo.windows.jsonl || { ok=0; continue; }
  if [ ! -e $FULL/hyp.jsonl ]; then
    mkdir -p $FULL
    python3 - $V6/on_${st}_$s/hyp.jsonl $R/hyp.jsonl $ON/windows_test_online_${st}.jsonl $FULL/hyp.jsonl <<'PY'
import json, sys
a, b, wf, out = sys.argv[1:5]
rec = {}
for f in (a, b):
    for l in open(f, encoding="utf-8"):
        rec[json.loads(l)["id"]] = l if l.endswith("\n") else l + "\n"
ids = [json.loads(l)["id"] for l in open(wf, encoding="utf-8")]
assert all(i in rec for i in ids), sum(i in rec for i in ids)
with open(out, "x", encoding="utf-8") as fh:
    fh.writelines(rec[i] for i in ids)
print("full", out, len(ids))
PY
  fi
  mscore $st $ON/windows_test_online_${st}.jsonl $FULL/hyp.jsonl $(ref_of $st) $SC/on_${st}_${s}_full.json --online
  # speech of the rest windows: queued once
  if [ ! -e $Q/sp_${st}_${s}.json ]; then
    echo "[{\"name\": \"on_${st}_${s}_rest\", \"set\": \"$st\", \"hyp\": \"$R\", \"windows\": \"$R/todo.windows.jsonl\", \"mode\": \"stream\"}]" > $F/sp_${st}_${s}_systems.json
    $PY reports/gstar_si_v7/make_v7_speech_jobs.py $F/sp_${st}_${s}_systems.json > $F/sp_${st}_${s}.json.part \
      && command mv -f $F/sp_${st}_${s}.json.part $F/sp_${st}_${s}.json && cp -n $F/sp_${st}_${s}.json reports/gstar_si_v7/jobs/sp_${st}_${s}.json \
      && cp -n $F/sp_${st}_${s}.json $Q/sp_${st}_${s}.json && log "queued speech of on_${st}_${s}_rest"
  fi
  SR=$SP7/on_${st}_${s}_rest; SF=$SP7/on_${st}_${s}_full
  [ -e $SR/talker/asr_speech.jsonl ] || { ok=0; continue; }
  if [ ! -e $SF/talker/asr_speech.jsonl ]; then
    mkdir -p $SF/talker $SF/talker_in
    python3 - $SP6/on_${st}_$s $SR $SF <<'PY'
import json, os, sys
old, new, out = sys.argv[1:4]
def recs(paths):
    r = {}
    for p in paths:
        if os.path.exists(p):
            for l in open(p, encoding="utf-8"):
                r.setdefault(json.loads(l)["id"], l if l.endswith("\n") else l + "\n")
    return r
import glob
res = recs([f"{old}/talker/results.jsonl"] + sorted(glob.glob(f"{old}/talker_r*/part_*.jsonl")) + [f"{new}/talker/results.jsonl"])
blk = recs([f"{old}/talker_in/blocks.jsonl", f"{new}/talker_in/blocks.jsonl"])
asr = recs([f"{old}/talker/asr_speech.jsonl"] + sorted(glob.glob(f"{old}/talker_r*/asr_speech.jsonl")) + [f"{new}/talker/asr_speech.jsonl"])
for name, r, d in (("results.jsonl", res, "talker"), ("blocks.jsonl", blk, "talker_in"), ("asr_speech.jsonl", asr, "talker")):
    with open(f"{out}/{d}/{name}.part", "x", encoding="utf-8") as fh:
        fh.writelines(r.values())
for name, d in (("results.jsonl", "talker"), ("blocks.jsonl", "talker_in"), ("asr_speech.jsonl", "talker")):
    os.replace(f"{out}/{d}/{name}.part", f"{out}/{d}/{name}")
drop = []
for p in glob.glob(f"{old}/talker_in*/DROPPED.txt") + glob.glob(f"{new}/talker_in*/DROPPED.txt"):
    drop += [x.split()[0].rstrip(":") for x in open(p) if x.strip() and not x.startswith("#")]
open(f"{out}/DROPPED_ids.txt", "w").write(",".join(sorted(set(drop))))
print("speech full", out, len(res), len(blk), len(asr), "dropped", drop)
PY
  fi
  if [ ! -e $FULL/hyp_asr.jsonl ]; then
    python3 - $FULL/hyp.jsonl $SF/talker/asr_speech.jsonl $FULL/hyp_asr.jsonl <<'PY'
import json, sys
from collections import defaultdict
hyp, asr, out = sys.argv[1:4]
A = {r["id"]: r.get("asr_text", "") for r in map(json.loads, open(asr, encoding="utf-8"))}
with open(out, "x", encoding="utf-8") as fh:
    for h in map(json.loads, open(hyp, encoding="utf-8")):
        by = defaultdict(list)
        for s in h["segments"]:
            by[s["spk"]].append(s)
        segs = []
        for k, ss in by.items():
            t = A.get(f"{h['id']}__S{k}", "").strip()
            if t:
                segs.append({"spk": k, "start_s": min(s["start_s"] for s in ss), "end_s": max(s["end_s"] for s in ss),
                             "text": t, "write_ms": max(s.get("write_ms", 0) for s in ss)})
        fh.write(json.dumps({"id": h["id"], "segments": segs}, ensure_ascii=False) + "\n")
PY
  fi
  mscore $st $ON/windows_test_online_${st}.jsonl $FULL/hyp_asr.jsonl $(ref_of $st) $SC/sp_on_${st}_${s}_full.json --online
  lang=$([ $st = en2zh ] && echo cmn || echo eng)
  v6l=$V6/latency/${st}_${s}.json
  rms=$(python3 -c "import json;print(json.load(open('$v6l'))['read_ms'])"); tok=$(python3 -c "import json;print(json.load(open('$v6l'))['tok_rate'])")
  for lay in single per-speaker; do
    for ex in none dropped; do
      out=$LAT/${st}_${s}_full$([ $lay = per-speaker ] && echo _ps)$([ $ex = dropped ] && echo _ex).json
      exl=$( [ $ex = dropped ] && cat $SP7/on_${st}_S2dl_full/DROPPED_ids.txt $SP7/on_${st}_O2_LA2_full/DROPPED_ids.txt 2>/dev/null | tr '\n' ',' )
      [ $ex = dropped ] && [ ! -e $SP7/on_${st}_S2dl_full/DROPPED_ids.txt -o ! -e $SP7/on_${st}_O2_LA2_full/DROPPED_ids.txt ] && continue
      [ -e $out ] || $PY reports/gstar_si_v6/listener_latency.py --windows $ON/windows_test_online_${st}.jsonl \
        --hyp $FULL/hyp.jsonl --talker $SF --read-ms $rms --tok-rate $tok --lang $lang --layout $lay ${exl:+--exclude $exl} \
        --out $out > $out.log 2>&1 && log "latency $out"
    done
  done
done; done

# ---- N-check: En->Zh MIST-Simul's v6 subset decoded again with the v7 decoder
N=$V7/nchk_on_en2zh_S2dl_sub
if merge $N $ON/windows_test_online_en2zh_sub.jsonl; then
  mscore en2zh $ON/windows_test_online_en2zh_sub.jsonl $N/hyp.jsonl $REF_EN $SC/nchk.json --online
else ok=0; fi

# ---- cascades on reference windows
for dr in en2zh zh2en; do for c in OC CC; do
  seg=$V6/${dr}_${c}-segments/segments.jsonl; out=$V7/${dr}_${c}
  merge $V7/${dr}_${c}_seg $seg || { ok=0; continue; }
  if [ ! -e $out/hyp.jsonl ]; then
    mkdir -p $out; extra=""; [ $c = CC ] && extra="--core"
    $PY -m $M.gstar_cascade_v6 assemble-v6 --seg $seg --hyp $V7/${dr}_${c}_seg/hyp.jsonl --windows $(ref_of $dr) --dir $dr $extra \
      --out $out/hyp.jsonl > $out/assemble.log 2>&1 && log "$out assembled"
  fi
  [ -e $out/hyp.jsonl ] && score $out $(ref_of $dr) offline
done; done
for st in en2zh zh2en ali; do
  seg=$EV/gstar_v3_20261029/segments_$st/segments.jsonl; out=$V7/${st}_C-turn
  merge $V7/${st}_Cturn_seg $seg || { ok=0; continue; }
  if [ ! -e $out/hyp.jsonl ]; then
    mkdir -p $out
    $PY -m $M.gstar_cascade assemble --seg $seg --hyp $V7/${st}_Cturn_seg/hyp.jsonl --windows $(ref_of $st) --out $out/hyp.jsonl \
      > $out/assemble.log 2>&1 && log "$out assembled"
  fi
  [ -e $out/hyp.jsonl ] && score $out $(ref_of $st) offline
done

# ---- reference-window decodes of v7 (seeds, LA2 on the prefix-trained models, fixed lag, streaming cascade)
for x in $(cat reports/gstar_si_v7/refwin_systems.txt 2>/dev/null); do
  n=${x%%:*}; w=${x#*:}; w=${w%%:*}; pk=${x##*:}
  if merge $V7/$n $w; then score $V7/$n $w $pk; else ok=0; fi
done
[ $ok = 1 ] && touch $F/ALL && log "after_v7: everything present"
