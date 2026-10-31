#!/usr/bin/env bash
# Paper v6: merge and score the dec2 decodes (v6-trained models), the batch-invariance pair and the v6 speech outputs.
# Idempotent; touches logs/multispk/v6/fu/E_all when everything is there. CPU only.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
PY=/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval/bin/python
M=experiments.uniss_qwen3_omni_v1.evaluation
EV=eval_outputs/uniss_qwen3_omni_v1; V6=$EV/gstar_v6_20261031; SP=$EV/gstar_v6_speech_20261031; SC=$V6/scores
D=data/ami_multispk_si_v1; A=data/alimeeting_si_v1; ON=data/online_v6_20261031
export PYTHONPATH=.
log() { echo "[$(date -u +%F' '%T) UTC] $*"; }
# merge(): as after_dec.sh
eval "$(sed -n '/^merge() {/,/^}/p' reports/gstar_si_v6/after_dec.sh)"
score() { [ -e $1/score.json ] && return 0; $PY -m $M.gstar_score --windows $2 --hyp $1/hyp.jsonl --post-key $3 $4 --out $1/score.json > $1/score.log 2>&1 && log "$1: $(grep -m1 cpBLEU_session $1/score.json)"; }
REF_EN=$D/e2e_gstar_v1/windows_test_ref.jsonl; REF_ENS=$D/e2e_gstar_stream_v1/windows_test_ref.jsonl
REF_ZH=$D/e2e_gstar_zh2en_v1/windows_test_ref.jsonl; REF_ALI=$A/windows_test_ref.jsonl; DEV=data/v6_sets/zh2en_dev800.jsonl
ok=1
chk() { "$@" || ok=0; }
# E-zh: dev gate and tests
for x in dev_zh2en_S2dl3k:$DEV dev_zh2en_S2dlchar:$DEV zh2en_S2dlchar_d1:$REF_ZH ali_S2dlchar_d1:$REF_ALI \
         en2zh_S2dl_seed1_d1:$REF_ENS en2zh_S2dlabc_d1:$REF_ENS zh2en_S2dlabc_d1:$REF_ZH \
         t13_zh2en_b16:$REF_ZH t13_zh2en_b4:$REF_ZH; do
  s=${x%%:*}; w=${x##*:}
  if [[ $s == t13_* ]]; then   # 96 windows only
    [ -e $V6/$s/hyp.jsonl ] || { ls $V6/$s/hyp_s*.jsonl >/dev/null 2>&1 && [ "$(cat $V6/$s/hyp_s*.jsonl | wc -l)" -ge 96 ] \
      && cat $V6/$s/hyp_s*.jsonl > $V6/$s/hyp.jsonl.part && command mv -f $V6/$s/hyp.jsonl.part $V6/$s/hyp.jsonl; } || ok=0
    continue
  fi
  if merge $V6/$s $w; then score $V6/$s $w lowlat ""; else ok=0; fi
done
# turn-trained cascade: assemble like C-ft2 (gstar_cascade assemble), then score
for dr in en2zh zh2en ali; do
  seg=$EV/gstar_v3_20261029/segments_$dr/segments.jsonl
  REF=$([ $dr = en2zh ] && echo $REF_EN || ([ $dr = zh2en ] && echo $REF_ZH || echo $REF_ALI))
  if merge $V6/${dr}_C-turn_seg $seg; then
    if [ ! -e $V6/${dr}_C-turn/hyp.jsonl ]; then
      mkdir -p $V6/${dr}_C-turn
      $PY -m $M.gstar_cascade assemble --seg $seg --hyp $V6/${dr}_C-turn_seg/hyp.jsonl --windows $REF --out $V6/${dr}_C-turn/hyp.jsonl \
        > $V6/${dr}_C-turn/assemble.log 2>&1
    fi
    score $V6/${dr}_C-turn $REF offline ""
  else ok=0; fi
done
# speech: AliMeeting reference windows (ASR-cpBLEU)
for x in ali_O2:$EV/gstar_v3_20261029/ali_O2:offline ali_C-ft2:$EV/gstar_v3_20261029/ali_C-ft2:offline \
         ali_S2dl_d1:$EV/gstar_v5_20261030/ali_S2dl3k_d1:lowlat ali_O2_LA2_d1:$EV/gstar_v3_20261029/ali_O2_LA2_d1:lowlat; do
  IFS=: read -r n h pk <<< "$x"
  if [ -e $SP/$n/talker/asr_speech.jsonl ]; then
    [ -e $SP/$n/score.json ] || $PY -m $M.gstar_score --windows $REF_ALI --hyp $h/hyp.jsonl --asr $SP/$n/talker/asr_speech.jsonl \
      --post-key $pk --out $SP/$n/score.json > $SP/$n/score.log 2>&1 && log "$SP/$n speech scored"
  else ok=0; fi
done
# speech of the online runs: whole-meeting tcpBLEU of what is spoken (each output speaker's ASR as one segment)
for st in en2zh zh2en ali; do for s in S2dl O2_LA2; do
  n=on_${st}_$s
  if [ -e $SP/$n/talker/asr_speech.jsonl ] && [ -e $V6/$n/hyp.jsonl ]; then
    if [ ! -e $V6/$n/hyp_asr.jsonl ]; then
      python3 - $V6/$n/hyp.jsonl $SP/$n/talker/asr_speech.jsonl $V6/$n/hyp_asr.jsonl <<'PY'
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
    REF=$([ $st = en2zh ] && echo $D/e2e_gstar_v1/windows_test_ref.jsonl || ([ $st = zh2en ] && echo $REF_ZH || echo $REF_ALI))
    [ -e $SC/sp_$n.json ] || $PY reports/gstar_si_v6/meeting_score.py --dir $st --windows $ON/windows_test_online_${st}_sub.jsonl \
      --hyp $V6/$n/hyp_asr.jsonl --ref-windows $REF --online --out $SC/sp_$n.json > $SC/sp_$n.json.log 2>&1 && log "speech tcp $n"
  else ok=0; fi
done; done
[ $ok = 1 ] && touch logs/multispk/v6/fu/E_all && log "after_dec2: all present"
