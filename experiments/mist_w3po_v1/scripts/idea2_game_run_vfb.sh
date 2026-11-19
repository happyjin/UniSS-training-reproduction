#!/usr/bin/env bash
# Idea 2 prereg v2.2: the shrink+Vfb variant on the 10 esports demo excerpts. Same as idea2_game_run.sh (Idea 1's
# voice-override wrapper, baseline B text and flags, both TTS seeds, cards 4-7 only) with the token tables
# tokens/game_<dir>_shrinkvfb.jsonl; run names game_b_<dir>__t1c_shrinkvfb{,_tts2}. Marker GAME_VFB_TALKER_DONE, then CPU
# metrics (measure, blind BLEU, ONE --pair with all four comparisons: idea1_speech_metrics --pair is nargs="+"), the
# identity check of non-fallback outputs against the shrink runs (sha1 of the generated wavs), TitaNet speaker
# similarity (idea2_game_spksim.py sim, card 7, stacked), and idea2_game_summary --tag shrinkvfb. Marker GAME_VFB_DONE.
#   [DRY_RUN=1] bash experiments/mist_w3po_v1/scripts/idea2_game_run_vfb.sh
# Detached: setsid nohup bash ... > logs/mist_w3po_v1/voicecond/idea2/game_run_vfb.log 2>&1 < /dev/null 9>&- &
# Checked with bash -n and DRY_RUN=1 only.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
. experiments/mist_w3po_v1/scripts/joint_lib.sh
export PATH=/opt/dlami/nvme/jasonleeeli/venvs/vllm_omni_030/bin:$PATH
GL=logs/mist_w3po_v1/voicecond/idea2/game
TOK=eval_outputs/mist_w3po_v1/voicecond_idea2/tokens
MET=eval_outputs/mist_w3po_v1/voicecond_idea2/metrics
B1=eval_outputs/mist_w3po_v1/voicecond_idea1/metrics
EPY=/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval/bin/python
B2=$JO/joint_v8h6k_20261113T073658Z/merged_s06000/thinker_merged
MARK=logs/mist_w3po_v1/voicecond/idea2
mkdir -p $GL $MET
run_orch() {   # <run dir> <command...>
  local R=$1; shift
  if [ -n "${DRY_RUN:-}" ]; then echo "$(basename $R): $*"; "$@" --dry | tail -3; return 0; fi
  [ -e $R ] && { log "$R exists (not overwritten)"; return 1; }
  setsid nohup "$@" < /dev/null > $R.launch.log 2>&1 9>&- &
  log "$(basename $R) orchestrator pid $!"
  wait_done $R; [ -z "$(orch_failed $R)" ] || { log "$(basename $R) failed ($(tail -1 $R/orch.log | cut -c1-200))"; return 1; }
}
chain() {   # <dir> <set>
  local d=$1 s=$2
  [ -s $TOK/game_${d}_shrinkvfb.jsonl ] || { log "no token table for $d"; return 1; }
  run_orch $GL/game_${d}_shrinkvfb $PY experiments/mist_w3po_v1/voicecond/idea1_game_orch.py \
    --voice-override $TOK/game_${d}_shrinkvfb.jsonl --dir $d --set $s --base $B2 --run $GL/game_${d}_shrinkvfb \
    --name game_b_${d}__t1c_shrinkvfb --ckpt - --speech-from game_b_${d} --talker t1c --window 1 --cap135 --exclude 0 1 2 3
}
if [ -n "${DRY_RUN:-}" ]; then chain zh2en ali; chain en2zh test; exit 0; fi
rc=0
chain zh2en ali & p1=$!
sleep 60
chain en2zh test & p2=$!
wait $p1 || rc=1
wait $p2 || rc=1
echo "rc=$rc $(date -u +%FT%TZ)" >> $MARK/GAME_VFB_TALKER_DONE
log "GAME_VFB_TALKER_DONE rc=$rc"
[ $rc = 0 ] || exit 1
for d in zh2en en2zh; do
  for sfx in "" _tts2; do
    n=game_b_${d}__t1c_shrinkvfb$sfx
    [ -e $MET/$n.jsonl ] || $EPY -m experiments.mist_w3po_v1.voicecond.idea1_speech_metrics measure \
      --talker eval_outputs/mist_w3po_v1/speech/$n/talker --out $MET/$n.jsonl --procs 48 || rc=1
    [ -e $MET/blind_bleu_$n.json ] || $EPY -m experiments.mist_w3po_v1.evaluation.game_blind_bleu --dir $d \
      --hyp eval_outputs/mist_w3po_v1/decode/$n/hyp_asr.jsonl --out $MET/blind_bleu_$n.json || rc=1
  done
done
[ -e $MET/compare_game_shrinkvfb_vs_B.json ] || $EPY -m experiments.mist_w3po_v1.voicecond.idea1_speech_metrics compare \
  --pair $B1/game_b_zh2en__t1c.jsonl:$MET/game_b_zh2en__t1c_shrinkvfb.jsonl:zh2en_s1 \
         $B1/game_b_zh2en__t1c_tts2.jsonl:$MET/game_b_zh2en__t1c_shrinkvfb_tts2.jsonl:zh2en_s2 \
         $B1/game_b_en2zh__t1c.jsonl:$MET/game_b_en2zh__t1c_shrinkvfb.jsonl:en2zh_s1 \
         $B1/game_b_en2zh__t1c_tts2.jsonl:$MET/game_b_en2zh__t1c_shrinkvfb_tts2.jsonl:en2zh_s2 \
  --primary zh2en_s1 zh2en_s2 en2zh_s1 en2zh_s2 --out $MET/compare_game_shrinkvfb_vs_B.json || rc=1
[ -e $MET/vfb_identity_check.json ] || $EPY - <<'PYEOF' || rc=1
import hashlib, json
from pathlib import Path
S = Path("eval_outputs/mist_w3po_v1/speech"); T = Path("eval_outputs/mist_w3po_v1/voicecond_idea2/tokens")
res = {}
for d in ("zh2en", "en2zh"):
    fb = {r["id"] for r in map(json.loads, open(T / f"game_{d}_shrinkvfb.jsonl")) if r["fallback"]}
    for sfx in ("", "_tts2"):
        a = {r["id"]: r.get("speech_path") for r in map(json.loads, open(S / f"game_b_{d}__t1c_shrink{sfx}/talker/results.jsonl"))}
        b = {r["id"]: r.get("speech_path") for r in map(json.loads, open(S / f"game_b_{d}__t1c_shrinkvfb{sfx}/talker/results.jsonl"))}
        h = lambda p: hashlib.sha1(Path(p).read_bytes()).hexdigest() if p and Path(p).exists() else None
        ids = sorted(i for i in a if i in b and i not in fb)
        same = sum(h(a[i]) == h(b[i]) and h(a[i]) is not None for i in ids)
        res[f"{d}{sfx or '_s1'}"] = {"non_fallback": len(ids), "identical_wav": same}
Path("eval_outputs/mist_w3po_v1/voicecond_idea2/metrics/vfb_identity_check.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res))
PYEOF
R=""; for d in zh2en en2zh; do for sfx in "" _tts2; do R="$R game_b_${d}__t1c_shrinkvfb$sfx"; done; done
CUDA_VISIBLE_DEVICES=7 /opt/dlami/nvme/jasonleeeli/venvs/nemo_diar/bin/python experiments/mist_w3po_v1/voicecond/idea2_game_spksim.py \
  sim --runs $R --device cuda:0 || rc=1
[ $rc = 0 ] && { $EPY -m experiments.mist_w3po_v1.voicecond.idea2_game_summary --tag shrinkvfb > $MET/summary_shrinkvfb.print 2>&1 || rc=1; }
[ $rc = 0 ] && { date -u +%FT%TZ >> $MARK/GAME_VFB_DONE; log GAME_VFB_DONE; } || { date -u +%FT%TZ >> $MARK/GAME_VFB_FAILED; log "vfb metrics failed"; }
