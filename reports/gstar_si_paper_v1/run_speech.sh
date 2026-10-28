#!/usr/bin/env bash
# Full-test-set speech of five G-STAR-SI systems (make_speech_jobs.py): voices for the streaming preset on every test
# meeting, talker inputs per system, then the holders on all eight cards stop, the gpuq6 queue runs the talker shards
# and the ASR / stitch / score jobs, and the holders come back. Launch detached (setsid nohup); check with bash -n.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
PYE=/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval/bin/python
SCR=/tmp/claude-0/-opt-dlami-nvme-jasonleeeli-projects-UniSS/d490b1cc-23c2-464f-a450-9958e1eda107/scratchpad
OLD=/tmp/claude-0/-opt-dlami-nvme-jasonleeeli-projects-UniSS/88583ca0-585a-4fad-ada7-4b193b743b50/scratchpad
EV=eval_outputs/uniss_qwen3_omni_v1
OUTROOT=${EV}/gstar_speech_full_20261028
JOBS=reports/gstar_si_paper_v1/jobs/speech_jobs.json
log() { echo "[$(TZ=Europe/Prague date +%H:%M:%S)] $*"; }

[ -e "${OUTROOT}" ] && { log "拒绝覆盖已有输出: ${OUTROOT}"; exit 1; }
mkdir -p "${OUTROOT}"
log "voices (lowlat, every test meeting; existing files kept)"
PYTHONPATH=. ${PYE} -m experiments.uniss_qwen3_omni_v1.evaluation.gstar_talker voices \
  --windows data/ami_multispk_si_v1/e2e_gstar_stream_v1/windows_test_ref.jsonl --preset lowlat \
  --out data/ami_multispk_si_v1/e2e_gstar_v1/voices_lowlat || { log "voices failed"; exit 1; }
prep() {   # name hyp_dir windows preset
  PYTHONPATH=. ${PYE} -m experiments.uniss_qwen3_omni_v1.evaluation.gstar_talker prep --hyp "$2/hyp.jsonl" \
    --windows "$3" --voices "data/ami_multispk_si_v1/e2e_gstar_v1/voices_$4" --out "${OUTROOT}/$1/talker_in" \
    || { log "prep $1 failed"; exit 1; }
}
mkdir -p ${OUTROOT}/{O-G1,B-off,S-T_d1,S-LA2_d1,B-LA2_d1}
prep O-G1 ${EV}/gstar_offline_eval_o1_20261026T195403Z/O-G1 data/ami_multispk_si_v1/e2e_gstar_v1/windows_test_ref.jsonl offline
prep B-off ${EV}/gstar_offline_eval_o1_20261026T195403Z/B-off data/ami_multispk_si_v1/e2e_gstar_v1/windows_test_ref.jsonl offline
for s in S-T_d1 S-LA2_d1 B-LA2_d1; do
  prep $s ${EV}/gstar_stream_eval_o1_20261026T195403Z/$s data/ami_multispk_si_v1/e2e_gstar_stream_v1/windows_test_ref.jsonl lowlat
done
log "holders off"
bash ${OLD}/holder_off.sh all
sleep 20
GPUQ_CARDS=0,1,2,3,4,5,6,7 ${PYE} -u ${SCR}/gpuq6.py ${JOBS} logs/gstar_speech_full
log "QUEUE_DONE"
sleep 15
bash ${OLD}/holder_on.sh 0,1,3,4
bash ${OLD}/holder_on.sh 2,5,6,7
log "SPEECH_FULL_DONE"
