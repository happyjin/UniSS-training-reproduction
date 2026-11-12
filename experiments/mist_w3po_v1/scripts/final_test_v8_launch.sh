#!/usr/bin/env bash
# Descriptive baseline (prereg amendment 1): MIST-Simul official v8 merged Thinker (no CRF ckpt), finalpost close_detect,
# the SAME C2 gap-0.6 test windows / refs / scorer as A' (final_test_orch.py --plain-thinker), text only, AMI zh->en and
# AliMeeting. Waits until En->Zh (a)'s Thinker training has exited (CRF_RUN_EXIT in its log), then starts both decode
# orchestrators on cards 0-3 only (--exclude 4 5 6 7 = the joint (b) training cards; --cap135 stacks within 135 GiB by
# real nvidia-smi usage; the ami decode starts after every ali shard has started). Refuses to overwrite.
#   DRY_RUN=1 prints the plan.  setsid nohup bash $0 > logs/mist_w3po_v1/final_test/v8_launch.log 2>&1 < /dev/null 9>&- &
# Markers: FT_V8_WAITING / FT_V8_LAUNCHED <set> / FT_V8_DECODE_DONE <set> / FT_V8_DONE / FT_V8_FAILED ...
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
export PYTHONPATH=. TMPDIR=/opt/dlami/nvme/jasonleeeli/work/mist_w3po_v1/tmp
PY=/opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python
E=logs/mist_w3po_v1/final_test
DEC=eval_outputs/mist_w3po_v1/decode
EN=logs/mist_w3po_v1/en2zh
TRL=logs/mist_w3po_v1/$(cat $EN/RUN_ID).log
WD=eval_outputs/mist_w3po_v1/data/windows_test_c2g06_v1
log() { echo "$(date -u +%FT%TZ) $*"; }
cmd() {  # <set> <win file> <port0> <extra>
  echo "$PY experiments/mist_w3po_v1/scripts/final_test_orch.py --set $1 --win $WD/$2 --window 1 --cap135 --run $E/$1_v8plain --name ft_$1_v8plain --ckpt - --plain-thinker --talker none --scorer-port0 $3 --exclude 4 5 6 7 $4"
}
CA=$(cmd ali windows_test_online_ali.jsonl 20310 "")
CM=$(cmd ami windows_test_online_zh2en.jsonl 20300 "--dec-after $E/ali_v8plain ft_ali_v8plain")
EXIST=""
for s in ami ali; do for d in $E/${s}_v8plain $DEC/ft_${s}_v8plain eval_outputs/mist_w3po_v1/crf/t_eval/ft_${s}_v8plain; do [ -e $d ] && EXIST="$EXIST $d"; done; done
if [ -n "${DRY_RUN:-}" ]; then echo "train log $TRL; existing (must be none):${EXIST:- none}"; echo "$CA"; echo "$CM"; exit 0; fi
[ -z "$EXIST" ] || { log "FT_V8_FAILED refuse to overwrite:$EXIST"; exit 1; }
log "FT_V8_WAITING for CRF_RUN_EXIT in $TRL (En->Zh (a) Thinker training; cards 0-3 exclusive until then)"
until [ -e $TRL ] && grep -q CRF_RUN_EXIT $TRL; do sleep 60; done
log "FT_V8_TRAINING_EXITED: $(grep CRF_RUN_EXIT $TRL | tail -1 | cut -c1-200)"
mkdir -p $E
launch() { setsid nohup $2 < /dev/null > $E/$1_v8plain.launch.log 2>&1 9>&- & log "FT_V8_LAUNCHED $1 pid $!"; sleep 30
  [ -e $E/$1_v8plain/orch.log ] || { log "FT_V8_FAILED launch $1 (see $E/$1_v8plain.launch.log)"; exit 1; }; }
launch ali "$CA"
launch ami "$CM"
fail=0
for s in ali ami; do
  until [ -e $E/${s}_v8plain/events/DONE ]; do sleep 30; done
  if grep -q 'DONE; failed/skipped \[\]' $E/${s}_v8plain/orch.log; then log "FT_V8_DECODE_DONE $s: $(tail -n 1 $DEC/ft_${s}_v8plain/score.log 2>/dev/null | cut -c1-200)"
  else log "FT_V8_FAILED $s: $(grep 'DONE;' $E/${s}_v8plain/orch.log)"; fail=1; fi
done
[ $fail = 0 ] && log "FT_V8_DONE" || log "FT_V8_FAILED (see above)"
