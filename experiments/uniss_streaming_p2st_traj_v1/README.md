# uniss_streaming_p2st_traj_v1

Trajectory-supervision improvements ported from SimulS2ST-Omni (arXiv 2607.19810,
`data/external/simuls2st_omni_demo/paper.txt`; repo clone at
`data/external/simuls2st_omni_demo/repo/SimulS2ST-Omni/`).

Parent: `uniss_streaming_p2st_pure_ce_v1` (C) at `iter_0004236`.  **Nothing in C
or in any earlier experiment is modified.**  Every module here is a new sibling
of an audited one, and C's `evaluation/` chain is reused unchanged for scoring
so the numbers stay comparable with everything already reported.

Steps, each its own training run and its own evaluation gate:

1. **NIR monotonicity filtering + difficulty/length stratification** (§A.3, §4.5).
   The paper's ablation shows m1 collapsing to 4.59/3.56 BLEU without it.
2. **Fixed chunk grid with explicit IDLE, and END_SEMANTIC on word-block
   boundaries** (§3.2 Step 2).
3. **Latency conditioning** (`src/train/prompt_formats.py::build_system_prompt`).

Step 4 (two-stream factorisation) is deliberately out of scope for this
experiment; see `reports/uniss_streaming_p2st_realsi_v1/PAPER_READING_2607.19810.zh-CN.md`.
