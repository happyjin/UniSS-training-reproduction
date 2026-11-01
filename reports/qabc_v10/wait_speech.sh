#!/usr/bin/env bash
# Wait for the v10 speech runs, returning early on anything that needs a decision. Prints one reason and the status:
#   ALL_DONE      every task of the task files has asr_speech.jsonl (in its folder or a -r<n> re-run)
#   WORKERS_GONE  no talker worker is left but tasks are not complete (failed attempts to re-run)
#   HOT           a card over its limit in mem_hot.log since the start
#   MIST_MSG      a message from MIST(v6) to Q(v10) in events.log since the start
#   FAILED        a new FAILED line in a worker log
#   FALLBACK      60 minutes without any of the above (one status line, then the wait is started again)
#   bash reports/qabc_v10/wait_speech.sh <tasks.jsonl> [<tasks.jsonl> ...]
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
start=$(date -u '+%F %T')
t0=$(date +%s)
f0=$(cat logs/qabc_v10/talker_worker_c*.log logs/qabc_v10/asr_worker_c*.log 2>/dev/null | grep -c FAILED)
left() {
  python3 - "$@" <<'EOF'
import json, sys, glob, os, re
n = 0
for f in sys.argv[1:]:
    for t in (x for x in map(json.loads, open(f)) if not x["key"].startswith("dsp_recalpl")):
        o = t["out"]
        cands = [o] + (glob.glob(re.sub(r"(speech/system_[^/]+?)(_m[\d.]+)", r"\1-r*\2", o)) if "/speech/system_" in o else glob.glob(o + "-r*"))
        if not any(os.path.exists(os.path.join(c, "asr_speech.jsonl")) for c in cands):
            n += 1
print(n)
EOF
}
while true; do
  n=$(left "$@")
  if [ "$n" -eq 0 ]; then echo "ALL_DONE"; break; fi
  if [ "$(ps -eo args | grep -c '[t]alker_worker.py talker')" -eq 0 ]; then echo "WORKERS_GONE ($n left)"; break; fi
  if grep -E "HOT card [01]:" logs/qabc_v10/mem_hot.log | awk -v s="[$start" '$0 >= s' | grep -q .; then echo "HOT ($n left)"; break; fi
  if awk -v s="[$start" '$0 >= s' logs/gpu_coord/events.log | grep -q "MIST(v6) -> Q"; then echo "MIST_MSG ($n left)"; break; fi
  f1=$(cat logs/qabc_v10/talker_worker_c*.log logs/qabc_v10/asr_worker_c*.log 2>/dev/null | grep -c FAILED)
  if [ "$f1" -gt "$f0" ]; then echo "FAILED ($n left)"; break; fi
  if [ $(( $(date +%s) - t0 )) -ge 3600 ]; then echo "FALLBACK ($n left)"; break; fi
  sleep 120
done
bash reports/qabc_v10/status.sh
