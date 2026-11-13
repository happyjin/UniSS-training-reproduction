#!/usr/bin/env bash
# Final test, Ali group 2 (VE1rl / D1vrl + PL-EM) late post-processing. The Ali driver wrote FT_ALL_DONE before the two
# continued read-level chains (final_test_after_merge.sh) finished, so gates2 ran on nothing (empty gate / utmos_paired
# outputs from 00:41 UTC are left untouched). This script waits for both FT_AFTER_MERGE_DONE lines, then writes NEW files
# only (suffix _v2 / new dirs): gate_2seed for (d) vs (c) and (e) vs (c) on the full set, the sensitivity (windows lost by
# any of the three merges removed from every system, text from the source decode), and the paired UTMOS differences.
# CPU only. Detach: setsid nohup bash <this> > logs/mist_w3po_v1/final_test/ali_g2_late.log 2>&1 < /dev/null &
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
export PYTHONPATH=. CUDA_VISIBLE_DEVICES= TMPDIR=/opt/dlami/nvme/jasonleeeli/work/mist_w3po_v1/tmp
PY=/opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python
E=logs/mist_w3po_v1/final_test
O=eval_outputs/mist_w3po_v1/crf/t_eval/final_test_v1/ali
DEC=eval_outputs/mist_w3po_v1/decode
Q=eval_outputs/mist_w3po_v1/talker_crf/quality
log() { echo "$(date -u +%FT%TZ) $*"; }
until grep -q FT_AFTER_MERGE_DONE $E/after_merge_ali_d1vrlplem.log 2>/dev/null && grep -q FT_AFTER_MERGE_DONE $E/after_merge_ali_ve1rlplem.log 2>/dev/null; do sleep 60; done
log "both Ali group-2 chains done"
C=ft_ali_aprime_s3000__rlplem; V=ft_ali_ve1_s3000__rlplem; D=ft_ali_d1v_s3000__rlplem
gate() { $PY -m experiments.mist_w3po_v1.evaluation.gate_2seed --cand $1/$2 $1/${2}_tts2 --ref $1/$3 $1/${3}_tts2 --out $4 > ${4%.json}.log 2>&1; log "gate $4 rc=$?"; }
gate $DEC $V $C $O/gate2_v2_${V}_vs_$C.json
gate $DEC $D $C $O/gate2_v2_${D}_vs_$C.json
for p in "$V ve1_vs_c" "$D d1v_vs_c"; do set -- $p
  $PY -m experiments.mist_w3po_v1.evaluation.listen_paired_meeting --cand $Q/$1 $Q/${1}_tts2 --ref $Q/$C $Q/${C}_tts2 > $O/utmos_paired_v2_$2.json 2> $O/utmos_paired_v2_$2.log
  log "paired UTMOS $2 rc=$?"; done
L=$O/merge_pairlost_ft_ali_aprime_s3000.json; for t in ve1 d1v; do L="$L $O/merge_pairlost_ft_ali_${t}_s3000.json"; done
NAMES="ft_ali_aprime_s3000 ft_ali_ve1_s3000 ft_ali_d1v_s3000"; for n in $C $V $D; do NAMES="$NAMES $n ${n}_tts2"; done
$PY -m experiments.mist_w3po_v1.evaluation.final_test_excl --set ali --lost $L --out $O/excl_g2_v1 --names $NAMES > $O/excl_g2_v1.log 2>&1
log "excl rc=$?"
X=$O/excl_g2_v1; mkdir -p $X/gate_in
for d in $X/ft_*; do n=$(basename $d); [ -e $d/score_asr.json ] || continue
  s=${n%%__*}; s=${s%_tts2}; mkdir -p $X/gate_in/$n
  ln -s $(realpath $d/score_asr.json) $X/gate_in/$n/score_asr.json; ln -s $(realpath $X/$s/score_text.json) $X/gate_in/$n/score_text.json; done
gate $X/gate_in $V $C $X/gate2_${V}_vs_$C.json
gate $X/gate_in $D $C $X/gate2_${D}_vs_$C.json
log FT_ALI_G2_LATE_DONE
