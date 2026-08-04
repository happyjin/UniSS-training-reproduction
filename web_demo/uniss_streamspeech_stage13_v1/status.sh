#!/usr/bin/env bash
set -euo pipefail
DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SESSION=${UNISS_STAGE13_SESSION:-uniss_streamspeech_stage13_public_v1}
tmux has-session -t "$SESSION" 2>/dev/null && echo "SESSION=running" || echo "SESSION=stopped"
test -f "$DIR/public_url.txt" && echo "PUBLIC_URL=$(cat "$DIR/public_url.txt")"
test -f "$DIR/access_info.json" && cat "$DIR/access_info.json"
