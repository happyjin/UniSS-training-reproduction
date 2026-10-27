Plan 4b rows (see data_clean/build_thinker_oracle_prefix.py).

{
 "reads_all": 569844,
 "finals_all": 80000,
 "offline_all": 40000,
 "train": 235309,
 "val": 4691,
 "wait_share": 0.731,
 "empty_wait_share": 0.225,
 "offline_share": 0.167,
 "en2zh": 117341,
 "zh2en": 122659,
 "start_reads_with_start_oracle": 50027,
 "start_targets_lengthened": 8113,
 "start_reads_with_rewrite": 0,
 "their_agreement": 0.0,
 "args": {
  "dumps": "data/si_bayes_lr_v1/dump_*/dumps/*.jsonl,data/si_belief_v1/shard_*/dumps/*.jsonl",
  "items": "data/unist_simul_thinker_v1/items_{d}.jsonl",
  "offline_share": 0.2,
  "max_rows": 240000,
  "val_every": 50,
  "seed": 20261023,
  "oracle_rewrites": "",
  "rewrite_offline_share": 0.5,
  "clean_targets": false,
  "start_oracle": "data/si_start_oracle_v1/start_oracle.strict2.jsonl",
  "start_boost": 2
 }
}
