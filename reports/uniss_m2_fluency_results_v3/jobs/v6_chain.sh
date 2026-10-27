cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
S=/tmp/claude-0/-opt-dlami-nvme-jasonleeeli-projects-UniSS/88583ca0-585a-4fad-ada7-4b193b743b50/scratchpad
until grep -q V5_CHAIN_DONE logs/thinker_op_v5so_20261027T130139Z_chain.log; do sleep 20; done
sleep 30
echo "gen $(date -u +%H:%M)"
(cd $S && python3 gpuq5.py flu16a2_jobs.json flu16a_logs > flu16a_queue.log 2>&1)
sleep 30
cat data/si_align_oracle_v1/full/part_*.jsonl > data/si_align_oracle_v1/full/align.jsonl
python3 -m experiments.uniss_qwen3_omni_v1.data_clean.build_align_rows --align data/si_align_oracle_v1/full/align.jsonl --delta 1.0 --out data/si_thinker_oracle_prefix_v6_align || exit 1
bash $S/holder_off.sh all; sleep 15
echo "train $(date -u +%H:%M)"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 RUN_ID=thinker_op_v6al_20261027T140752Z ROWS=data/si_thinker_oracle_prefix_v6_align/rows_train.jsonl NPROC=8 MBS=32 GBS=256 NUM_WORKERS=12 TRAIN_ITERS=800 SAVE_INTERVAL=200 EXIT_INTERVAL=200 WARMUP_ITERS=30 MASTER_PORT=29836 MERGED=$PWD/eval_outputs/uniss_qwen3_omni_v1/thinker_op_v6al_20261027T140752Z/thinker_merged_i200 bash experiments/uniss_qwen3_omni_v1/scripts/run_thinker_sd.sh > logs/thinker_op_v6al_20261027T140752Z.train.log 2>&1
echo "trained+merged $(date -u +%H:%M) $(tail -1 logs/thinker_op_v6al_20261027T140752Z.train.log)"
(cd $S && python3 gpuq5.py flu16b2_jobs.json flu16b_logs > flu16b_queue.log 2>&1)
echo "V6_CHAIN_DONE $(date -u +%H:%M)"
