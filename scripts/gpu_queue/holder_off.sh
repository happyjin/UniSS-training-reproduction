# Stop holder sessions covering any of the given cards (or all with "all"); by tmux session name, never by pattern-kill.
want="${1:?cards or all}"
for s in $(tmux ls -F '#S' 2>/dev/null | grep -E '^(gpu_hold_[0-9]+|uniss_gpu_load_60)$'); do
  if [ "$want" = all ] || [ "$s" = uniss_gpu_load_60 ] || [[ "${s#gpu_hold_}" == *[${want//,/}]* ]]; then tmux kill-session -t "$s" && echo "stopped $s"; fi
done
