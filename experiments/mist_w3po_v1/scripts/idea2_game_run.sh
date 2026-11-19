#!/usr/bin/env bash
# Idea 2 M1b (prereg v2 §G): baseline B's T1c speech chain on the 10 esports demo excerpts with the Talker's 32 voice
# tokens overridden by the contraction-model tokens (idea2_game_tokens.py tables), via Idea 1's wrapper
# voicecond/idea1_game_orch.py (only the talk jobs change: si_infer -> idea1_si_infer_vo, sha1-checked against the
# baseline prompt files). Same text (--speech-from game_b_<dir>), same flags as game_run.sh, both TTS seeds, cards 4-7
# only (--exclude 0 1 2 3). Run names game_b_<dir>__t1c_shrink{,_tts2}. After both directions: marker GAME_TALKER_DONE
# (always written; it releases idea2_full_chain_v2.sh's GPU part), then CPU metrics: idea1_speech_metrics measure /
# compare vs baseline B (Idea 1's measured baseline files), game_blind_bleu. Marker GAME_METRICS_DONE.
#   [DRY_RUN=1] bash experiments/mist_w3po_v1/scripts/idea2_game_run.sh
# Detached: setsid nohup bash ... > logs/mist_w3po_v1/voicecond/idea2/game_run.log 2>&1 < /dev/null 9>&- &
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
  [ -s $TOK/game_${d}_shrink.jsonl ] || { log "no token table for $d"; return 1; }
  run_orch $GL/game_${d}_shrink $PY experiments/mist_w3po_v1/voicecond/idea1_game_orch.py \
    --voice-override $TOK/game_${d}_shrink.jsonl --dir $d --set $s --base $B2 --run $GL/game_${d}_shrink \
    --name game_b_${d}__t1c_shrink --ckpt - --speech-from game_b_${d} --talker t1c --window 1 --cap135 --exclude 0 1 2 3
}
if [ -n "${DRY_RUN:-}" ]; then chain zh2en ali; chain en2zh test; exit 0; fi
rc=0
chain zh2en ali & p1=$!
sleep 60
chain en2zh test & p2=$!
wait $p1 || rc=1
wait $p2 || rc=1
echo "rc=$rc $(date -u +%FT%TZ)" >> $MARK/GAME_TALKER_DONE
log "GAME_TALKER_DONE rc=$rc"
[ $rc = 0 ] || exit 1
for d in zh2en en2zh; do
  for sfx in "" _tts2; do
    n=game_b_${d}__t1c_shrink$sfx
    [ -e $MET/$n.jsonl ] || $EPY -m experiments.mist_w3po_v1.voicecond.idea1_speech_metrics measure \
      --talker eval_outputs/mist_w3po_v1/speech/$n/talker --out $MET/$n.jsonl --procs 48 || rc=1
    [ -e $MET/blind_bleu_$n.json ] || $EPY -m experiments.mist_w3po_v1.evaluation.game_blind_bleu --dir $d \
      --hyp eval_outputs/mist_w3po_v1/decode/$n/hyp_asr.jsonl --out $MET/blind_bleu_$n.json || rc=1
  done
done
[ -e $MET/compare_game_shrink_vs_B.json ] || $EPY -m experiments.mist_w3po_v1.voicecond.idea1_speech_metrics compare \
  --pair $B1/game_b_zh2en__t1c.jsonl:$MET/game_b_zh2en__t1c_shrink.jsonl:zh2en_s1 \
  --pair $B1/game_b_zh2en__t1c_tts2.jsonl:$MET/game_b_zh2en__t1c_shrink_tts2.jsonl:zh2en_s2 \
  --pair $B1/game_b_en2zh__t1c.jsonl:$MET/game_b_en2zh__t1c_shrink.jsonl:en2zh_s1 \
  --pair $B1/game_b_en2zh__t1c_tts2.jsonl:$MET/game_b_en2zh__t1c_shrink_tts2.jsonl:en2zh_s2 \
  --primary zh2en_s1 zh2en_s2 en2zh_s1 en2zh_s2 --out $MET/compare_game_shrink_vs_B.json || rc=1
[ $rc = 0 ] && { date -u +%FT%TZ >> $MARK/GAME_METRICS_DONE; log GAME_METRICS_DONE; } || log "metrics failed"
