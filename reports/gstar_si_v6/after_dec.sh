#!/usr/bin/env bash
# Paper v6: merge and score every v6 decode whose shards are all complete (idempotent: a merged hyp.jsonl or an existing
# score is never rewritten; incomplete systems are reported and left alone). CPU only.
#   ref-window systems   -> <sys>/hyp.jsonl, <sys>/score.json (gstar_score)
#   cascades             -> <dir>_<OC|CC|SC>/hyp.jsonl via gstar_cascade_v6 assemble-v6, then gstar_score
#   online systems       -> on_<set>_<sys>/hyp.jsonl, scores/on_<set>_<sys>.json (meeting_score --online)
#   window references    -> scores/win_<set>_<sys>.json (meeting_score on the reference-window output of the same system)
#     bash reports/gstar_si_v6/after_dec.sh >> logs/multispk/v6/after.log 2>&1
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
PY=/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval/bin/python
M=experiments.uniss_qwen3_omni_v1.evaluation
EV=eval_outputs/uniss_qwen3_omni_v1; V6=$EV/gstar_v6_20261031; SC=$V6/scores
D=data/ami_multispk_si_v1; A=data/alimeeting_si_v1; ON=data/online_v6_20261031
mkdir -p $SC
export PYTHONPATH=.
log() { echo "[$(date -u +%F' '%T) UTC] $*"; }

merge() {   # $1 dir, $2 windows: when every window has a valid record in some hyp_s*.jsonl (shards and rest runs),
            # write hyp.jsonl in window order (a line cut by a stopped worker is ignored); else report the count
  [ -e $1/hyp.jsonl ] && return 0
  ls $1/hyp_s*.jsonl > /dev/null 2>&1 || { log "$1: no output yet"; return 1; }
  python3 - "$1" "$2" <<'PY' || return 1
import json, sys, os
d, wf = sys.argv[1:3]
rec = {}
for f in sorted(os.listdir(d)):
    if f.startswith("hyp_s") and f.endswith(".jsonl"):
        for line in open(os.path.join(d, f), encoding="utf-8"):
            try:
                h = json.loads(line)
            except ValueError:
                continue
            rec.setdefault(h["id"], line if line.endswith("\n") else line + "\n")
ids = [json.loads(l)["id"] for l in open(wf, encoding="utf-8")]
if not all(i in rec for i in ids):
    print(f"[merge] {d}: {sum(i in rec for i in ids)} / {len(ids)}", flush=True)
    sys.exit(1)
with open(os.path.join(d, "hyp.jsonl.part"), "x", encoding="utf-8") as fh:
    fh.writelines(rec[i] for i in ids)
os.replace(os.path.join(d, "hyp.jsonl.part"), os.path.join(d, "hyp.jsonl"))
print(f"[merge] {d}: merged {len(ids)}", flush=True)
PY
}
score() {   # $1 dir, $2 windows, $3 post key
  [ -e $1/score.json ] && return 0
  $PY -m $M.gstar_score --windows $2 --hyp $1/hyp.jsonl --post-key $3 --out $1/score.json > $1/score.log 2>&1 \
    && log "$1: $(grep -m1 cpBLEU_session $1/score.json) $(grep -m1 LAAL_s $1/score.json)"
}
mscore() {   # $1 set, $2 windows, $3 hyp, $4 ref windows, $5 out, $6 extra
  [ -e $5 ] && return 0
  $PY reports/gstar_si_v6/meeting_score.py --dir $1 --windows $2 --hyp $3 --ref-windows $4 $6 --out $5 > $5.log 2>&1 \
    && log "$5: $(tail -n 1 $5.log | cut -c1-200)"
}
REF_EN=$D/e2e_gstar_v1/windows_test_ref.jsonl; REF_ENS=$D/e2e_gstar_stream_v1/windows_test_ref.jsonl
REF_ZH=$D/e2e_gstar_zh2en_v1/windows_test_ref.jsonl; REF_ALI=$A/windows_test_ref.jsonl

# reference-window systems
# cascades
for dr in en2zh zh2en; do
  REF=$([ $dr = en2zh ] && echo $REF_EN || echo $REF_ZH)
  for c in OC CC; do
    seg=$V6/${dr}_${c}-segments/segments.jsonl
    merge $V6/${dr}_${c}_seg $seg || continue
    out=$V6/${dr}_${c}
    if [ ! -e $out/hyp.jsonl ]; then
      mkdir -p $out
      extra=""; [ $c = CC ] && extra="--core"; [ $c = SC ] && extra="--stream"
      $PY -m $M.gstar_cascade_v6 assemble-v6 --seg $seg --hyp $V6/${dr}_${c}_seg/hyp.jsonl --windows $REF --dir $dr $extra \
        --out $out/hyp.jsonl > $out/assemble.log 2>&1 && log "$out assembled: $(cat $out/assemble.log)"
    fi
    [ -e $out/hyp.jsonl ] && score $out $REF $([ $c = SC ] && echo lowlat || echo offline)
  done
done
# online whole-meeting systems
for x in en2zh:S2dl en2zh:O2_LA2 en2zh:S2dlm zh2en:S2dl zh2en:O2_LA2 ali:S2dl ali:O2_LA2; do
  set_=${x%%:*}; s=${x##*:}
  W=$ON/windows_test_online_${set_}_sub.jsonl   # prereg_v6 amendment 2: meeting subset
  REF=$([ $set_ = en2zh ] && echo $REF_EN || ([ $set_ = zh2en ] && echo $REF_ZH || echo $REF_ALI))
  merge $V6/on_${set_}_$s $W && mscore $set_ $W $V6/on_${set_}_$s/hyp.jsonl $REF $SC/on_${set_}_$s.json --online
done
# the same scorer on the reference-window outputs (window evaluation vs whole meeting, same scored utterances)
mscore en2zh $REF_ENS $EV/gstar_v3_20261029/en2zh_S2dl_d1/hyp.jsonl $REF_EN $SC/win_en2zh_S2dl.json ""
mscore en2zh $REF_ENS $EV/gstar_v3_20261029/en2zh_O2_LA2_d1/hyp.jsonl $REF_EN $SC/win_en2zh_O2_LA2.json ""
mscore en2zh $REF_ENS $EV/gstar_v5_20261030/en2zh_S2dlm_d1/hyp.jsonl $REF_EN $SC/win_en2zh_S2dlm.json ""
mscore en2zh $REF_EN $EV/gstar_v3_20261029/en2zh_O2/hyp.jsonl $REF_EN $SC/win_en2zh_O2.json ""
mscore en2zh $REF_EN $EV/gstar_v3_20261029/en2zh_C-ft2/hyp.jsonl $REF_EN $SC/win_en2zh_C-ft2.json ""
mscore zh2en $REF_ZH $EV/gstar_v5_20261030/zh2en_S2dl3k_d1/hyp.jsonl $REF_ZH $SC/win_zh2en_S2dl.json ""
mscore zh2en $REF_ZH $EV/gstar_v3_20261029/zh2en_O2_LA2_d1/hyp.jsonl $REF_ZH $SC/win_zh2en_O2_LA2.json ""
mscore zh2en $REF_ZH $EV/gstar_v3_20261029/zh2en_O2/hyp.jsonl $REF_ZH $SC/win_zh2en_O2.json ""
mscore zh2en $REF_ZH $EV/gstar_v3_20261029/zh2en_C-ft2/hyp.jsonl $REF_ZH $SC/win_zh2en_C-ft2.json ""
mscore ali $REF_ALI $EV/gstar_v5_20261030/ali_S2dl3k_d1/hyp.jsonl $REF_ALI $SC/win_ali_S2dl.json ""
mscore ali $REF_ALI $EV/gstar_v3_20261029/ali_O2_LA2_d1/hyp.jsonl $REF_ALI $SC/win_ali_O2_LA2.json ""
mscore ali $REF_ALI $EV/gstar_v3_20261029/ali_O2/hyp.jsonl $REF_ALI $SC/win_ali_O2.json ""
mscore ali $REF_ALI $EV/gstar_v3_20261029/ali_C-ft2/hyp.jsonl $REF_ALI $SC/win_ali_C-ft2.json ""
# window evaluation on the same meeting subset (the comparison online vs reference windows of prereg_v6 §1)
SUBE=$ON/windows_test_ref_en2zh_offline_sub.jsonl; SUBES=$ON/windows_test_ref_en2zh_stream_sub.jsonl
SUBZ=$ON/windows_test_ref_zh2en_sub.jsonl; SUBA=$ON/windows_test_ref_ali_sub.jsonl
mscore en2zh $SUBES $EV/gstar_v3_20261029/en2zh_S2dl_d1/hyp.jsonl $SUBE $SC/winsub_en2zh_S2dl.json ""
mscore en2zh $SUBES $EV/gstar_v3_20261029/en2zh_O2_LA2_d1/hyp.jsonl $SUBE $SC/winsub_en2zh_O2_LA2.json ""
mscore zh2en $SUBZ $EV/gstar_v5_20261030/zh2en_S2dl3k_d1/hyp.jsonl $SUBZ $SC/winsub_zh2en_S2dl.json ""
mscore zh2en $SUBZ $EV/gstar_v3_20261029/zh2en_O2_LA2_d1/hyp.jsonl $SUBZ $SC/winsub_zh2en_O2_LA2.json ""
mscore ali $SUBA $EV/gstar_v5_20261030/ali_S2dl3k_d1/hyp.jsonl $SUBA $SC/winsub_ali_S2dl.json ""
mscore ali $SUBA $EV/gstar_v3_20261029/ali_O2_LA2_d1/hyp.jsonl $SUBA $SC/winsub_ali_O2_LA2.json ""
# LA2 on the prefix-trained model: the 8-meeting subset (prereg_v6 amendment 3)
merge $V6/en2zh_S2dl_LA2_d1 $SUBES && { [ -e $V6/en2zh_S2dl_LA2_d1/score.json ] || $PY -m $M.gstar_score --windows $SUBES \
  --hyp $V6/en2zh_S2dl_LA2_d1/hyp.jsonl --post-key lowlat --out $V6/en2zh_S2dl_LA2_d1/score.json > $V6/en2zh_S2dl_LA2_d1/score.log 2>&1; }
log "after_dec pass done"
