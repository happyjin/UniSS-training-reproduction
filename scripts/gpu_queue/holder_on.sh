# Start the GPU holder on the given cards only (e.g. "4,5,6,7"), one tmux session per call; refuses cards in use.
cards="${1:?cards, e.g. 4,5,6,7}"; tag="${cards//,/}"
cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
for c in ${cards//,/ }; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $c)
  [ "$used" -lt 1000 ] || { echo "card $c in use (${used} MiB); holder not started"; exit 3; }
done
tmux has-session -t gpu_hold_$tag 2>/dev/null && { echo "gpu_hold_$tag already running"; exit 0; }
tmux new-session -d -s gpu_hold_$tag "exec /opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval/bin/python -u scripts/gpu_load/target_gpu_util.py --devices $cards --target-util 60 --target-memory-percent 90 --cycle-seconds 1 --matrix-size 16384 --dtype bfloat16 --sync-every 1 --log-interval 30 >> logs/gpu_hold_$tag.log 2>&1"
echo "holder on cards $cards (tmux gpu_hold_$tag)"
