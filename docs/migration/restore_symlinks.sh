#!/usr/bin/env bash
# Recreate the symlinks listed in symlinks_jasonleeeli.tsv (path<TAB>target, exported 2026-10-30 from the old server).
# The S3 backup ran with --no-follow-symlinks, so restored conda envs, venvs and HF caches miss them.
# Only creates links whose path does not exist yet; never replaces or removes anything.
#
#   bash docs/migration/restore_symlinks.sh [prefix-filter, e.g. /opt/dlami/nvme/jasonleeeli/conda_envs]
set -u
TSV="$(dirname "$0")/symlinks_jasonleeeli.tsv"
FILTER="${1:-/opt/dlami/nvme/jasonleeeli/}"
made=0; kept=0
while IFS=$'\t' read -r path target; do
    case "$path" in "$FILTER"*) ;; *) continue ;; esac
    if [ -e "$path" ] || [ -L "$path" ]; then kept=$((kept + 1)); continue; fi
    mkdir -p "$(dirname "$path")" && ln -s "$target" "$path" && made=$((made + 1))
done < "$TSV"
echo "created $made symlinks, $kept already present"
