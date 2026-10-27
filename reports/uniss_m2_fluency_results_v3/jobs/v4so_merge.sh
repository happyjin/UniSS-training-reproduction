cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
for it in 200 400; do
  until [ -f checkpoints/uniss_qwen3_omni_v1/thinker_op_v4so_20261027T122245Z/iter_0000$it/metadata.json ]; do sleep 20; done
  sleep 90
  echo "merge i$it $(date -u +%H:%M)"
  CUDA_VISIBLE_DEVICES='' /opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python experiments/uniss_qwen3_omni_v1/training/merge_thinker_sd.py --checkpoint-dir checkpoints/uniss_qwen3_omni_v1/thinker_op_v4so_20261027T122245Z/iter_0000$it --rank 32 --alpha 64 --out eval_outputs/uniss_qwen3_omni_v1/thinker_op_v4so_20261027T122245Z/thinker_merged_i$it > logs/thinker_op_v4so_20261027T122245Z_merge_i$it.log 2>&1 &
done
wait
echo MERGE_DONE $(date -u +%H:%M)
