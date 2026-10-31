"""Task file for talker_worker.py: the v10 speech runs, in priority order (dev speech of the decision rules first, then the
RealSI runs of the adapter seeds and of the binary gate, three talker seeds and LA2's 15 windows each). Names and folders
are those of make_v10_jobs.py, so the loaders of the paper scripts find them.

    python reports/qabc_v10/make_talker_tasks.py <out.jsonl>
"""
import json
import sys

O = "eval_outputs/uniss_qwen3_omni_v1"
O10 = f"{O}/qabc_v10"
W15 = ["0.5", "0.6", "0.75", "0.9", "1", "1.1", "1.25", "1.5", "1.75", "2", "2.5", "3", "4", "5", "6"]
DEVW = ["0.5", "0.75", "1", "1.25", "1.5", "2", "2.5", "3"]
RATE = {"en2zh": "0.191", "zh2en": "0.339"}
tasks = []
for rule in ("p5", "seqb", "recalpl"):
    for d in ("en2zh", "zh2en"):
        for w in DEVW:
            tasks.append({"key": f"dsp_{rule}_{d}_m{w}", "blocks": f"{O10}/dev_rules/blocks_{d}_w{w}_qr{rule}s0b1q{RATE[d]}.jsonl",
                          "direction": d, "window": float(w), "seed": 0, "out": f"{O10}/devspeech/{rule}_{d}_m{w}",
                          "items": f"data/si_thinker_oracle_prefix_v1/dev_items_{d}.jsonl"})
systems = [("v10s1-q-b1", f"{O}/si_v4/realsi_grid_s1", "_sws1b1q{r}"), ("v10s3-q-b1", f"{O10}/realsi_grid_s3", "_sws3b1q{r}"),
           ("v10qb-q-b1", f"{O}/si_v4/realsi_grid_s0qb", "_sws0b1qb{r}"), ("v10s2-q-b1", f"{O10}/realsi_grid_s2", "_sws2b1q{r}"),
           ("v10s4-q-b1", f"{O10}/realsi_grid_s4", "_sws4b1q{r}")]
for seed in (0, 1, 2):
    for name, bdir, suf in systems:
        for d in ("en2zh", "zh2en"):
            for w in W15:
                nm = (name if d == "en2zh" else f"{name}-zh") + (f"-seed{seed}" if seed else "")
                tasks.append({"key": f"sp_{nm}_m{w}_{d}", "blocks": f"{bdir}/blocks_{d}_w{w}{suf.format(r=RATE[d])}.jsonl",
                              "direction": d, "window": float(w), "seed": seed,
                              "out": f"{O10}/speech/system_{nm}_m{w}/_{d}/realsi_{d}"})
with open(sys.argv[1], "x", encoding="utf-8") as fh:
    for t in tasks:
        fh.write(json.dumps(t) + "\n")
print(len(tasks), "tasks")
