#!/usr/bin/env bash
# Idea 2 e2e prereg §4 ablation constlam (idea2_e2e_abl_constlam_v1) on cards 4-7 (coordinator / user 2026-11-19 ~09:50
# Prague: start now, stacked with the e2e test evaluation). Same setup as main_v2 (same trainer and fixes, data, seed,
# 3000 steps, 4-GPU DDP); NO slot claim (slot B would put 4-7 into exclude_cards and block the test orchestrators); no
# waiting for evals; only the guard placeholders on cards 4-7 are stopped, by PID (cmdline checked), right before launch.
# Writes <out>/LAUNCHER_RC and a line in logs/mist_w3po_v1/voicecond/idea2/E2E_ABLATION_CHAIN_DONE. Refuses an existing <out>.
#   [DRY_RUN=1] bash experiments/mist_w3po_v1/scripts/idea2_e2e_constlam_run.sh
# Detached: setsid nohup bash ... > logs/mist_w3po_v1/voicecond/idea2/e2e_idea2_e2e_abl_constlam_v1.launcher.log 2>&1 < /dev/null 9>&- &
# Checked with bash -n and DRY_RUN=1 only.
set -uo pipefail
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
J=/opt/dlami/nvme/jasonleeeli
VB=$J/venvs/vllm_omni_030/bin
CARDS=4,5,6,7
CL="4 5 6 7"
RUN=idea2_e2e_abl_constlam_v1
LOGD=logs/mist_w3po_v1/voicecond/idea2
OUT=eval_outputs/mist_w3po_v1/voicecond_idea2/e2e/$RUN
MARK=$LOGD/E2E_ABLATION_CHAIN_DONE
PORT=29662
log() { echo "$(TZ=Europe/Prague date '+%F %T %Z') $*"; }
[ -e $OUT ] && { log "refuse: $OUT exists"; exit 1; }
free_card() {
  local k=$1 sup kids s a
  case " $CL " in *" $k "*) ;; *) log "refuse to free card $k"; return;; esac
  sup=$(ps -eo pid,ppid,args | awk -v k="$k" '$2 == 1 && $0 ~ /train_mist_workload\.py/ && $0 ~ ("--devices " k " ") {print $1}')
  for s in $sup; do
    a=$(tr '\0' ' ' < /proc/$s/cmdline 2>/dev/null)
    case "$a" in *train_mist_workload.py*"--devices $k "*) ;; *) continue;; esac
    kids=$(ps -eo pid,ppid | awk -v s="$s" '$2 == s {print $1}' | tr '\n' ' ')
    log "card $k: SIGTERM placeholder supervisor $s children: $kids"
    [ -n "${DRY_RUN:-}" ] || kill -TERM $kids $s 2>/dev/null
  done
}
for k in $CL; do free_card $k; done
[ -n "${DRY_RUN:-}" ] || sleep 10
CMD="env TZ=Europe/Prague TMPDIR=$J/work/mist_w3po_v1/tmp HF_HOME=$J/cache/huggingface TORCH_HOME=$J/cache/torch \
XDG_CACHE_HOME=$J/cache PATH=$VB:$PATH OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$CARDS $VB/torchrun --nproc_per_node 4 \
--master_port $PORT -m experiments.mist_w3po_v1.voicecond.idea2_e2e_train --out $OUT --ablation constlam"
log "launch: ${CMD##*torchrun}"
[ -n "${DRY_RUN:-}" ] && exit 0
bash -c "$CMD" > $LOGD/e2e_$RUN.log 2>&1
rc=$?
st=FAILED; [ -e $OUT/DONE ] && st=DONE
[ -d $OUT ] && echo "$rc $(date -u +%FT%TZ)" >> $OUT/LAUNCHER_RC
echo "$RUN $rc $st" >> $MARK
log "$RUN: torchrun exit $rc, $st"
