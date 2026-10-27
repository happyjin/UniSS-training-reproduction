cd /opt/dlami/nvme/jasonleeeli/projects/UniSS
S=/tmp/claude-0/-opt-dlami-nvme-jasonleeeli-projects-UniSS/88583ca0-585a-4fad-ada7-4b193b743b50/scratchpad
until grep -q QUEUE_DONE $S/flu13_queue.log; do sleep 20; done
while ps -eo pid,args | awk '$2=="python3" && $3=="gpuq5.py" && $4=="flu13c_jobs.json"{f=1} END{exit !f}'; do sleep 5; done
sleep 15
bash $S/holder_off.sh all; sleep 15
echo "train $(date -u +%H:%M)"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 RUN_ID=thinker_op_v5so_20261027T130139Z ROWS=data/si_thinker_oracle_prefix_v5_startoracle_lb4/rows_train.jsonl NPROC=8 MBS=32 GBS=256 NUM_WORKERS=12 TRAIN_ITERS=800 SAVE_INTERVAL=200 EXIT_INTERVAL=200 WARMUP_ITERS=30 MASTER_PORT=29835 MERGED=$PWD/eval_outputs/uniss_qwen3_omni_v1/thinker_op_v5so_20261027T130139Z/thinker_merged_i200 bash experiments/uniss_qwen3_omni_v1/scripts/run_thinker_sd.sh > logs/thinker_op_v5so_20261027T130139Z.train.log 2>&1
echo "trained+merged $(date -u +%H:%M) $(tail -1 logs/thinker_op_v5so_20261027T130139Z.train.log)"
cd $S && python3 gpuq5.py flu14_jobs.json flu14_logs > flu14_queue.log 2>&1
echo "V5_CHAIN_DONE $(date -u +%H:%M)"
