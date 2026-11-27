#!/usr/bin/env bash
# portfolio 2026-11-27: card-pool job wrapper (started by lib_pool.sh launch under the per-card flock locks, which this process and
# its children hold until they exit). Stops the placeholders of its cards by PID (lib.sh free_card), then execs the job.
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
CARDS=$1; shift
source experiments/sst_hsmm_v1/lib.sh
for g in ${CARDS//,/ }; do free_card "$g"; done
log "pool_run: cards $CARDS" >&2
exec "$@"
