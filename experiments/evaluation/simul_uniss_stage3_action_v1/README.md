# Simul-UniSS Stage3 action evaluation v1

This directory is isolated from `experiments/evaluation/uniss_full198_phase2_phase3`.
It evaluates the full198 Stage3 iteration 4753 WAIT/WRITE checkpoint without
changing or overwriting any training data, checkpoint, historical evaluation,
or offline evaluation script.

The first full run uses:

- UniST dev on GPU 0–3;
- UniST test, called `eval` in the run layout, on GPU 4–7;
- bf16 and FlashAttention 2;
- a measured 262,144-token / 512-sample batch budget and 256-event LM-head batch;
- unpacked samples with independent attention masks;
- no sample truncation;
- full-vocabulary action CE/top1 plus WAIT-vs-WRITE classification metrics;
- GPU utilization, power, and memory monitoring.

The batch budget was selected on 7,000 identical dev records.  It processed
about 206.7k real tokens/s with 90.6% padding efficiency.  A larger 524,288
token batch increased peak power but reduced useful throughput to about
181.6k tokens/s and padding efficiency to 84.0%, so it is not the default.
Both settings produced identical discrete predictions on all 53,169 action
events (maximum target-CE difference: `1.91e-6`).  The evaluator never repeats
records or adds synthetic work merely to increase GPU utilization.

Run validation first:

```bash
experiments/evaluation/simul_uniss_stage3_action_v1/run_smoke.sh
```

Run the full 4+4 GPU evaluation:

```bash
experiments/evaluation/simul_uniss_stage3_action_v1/run_full_4plus4.sh
```

Every invocation creates a new timestamped directory under:

```text
eval_outputs/simul_uniss_stage3_action_v1/
```

The Stage3 report is deliberately limited to teacher-forced action metrics.
It does not label pseudo-schedule timing as paper-comparable AL/LAAL/ATD and
does not report ASR-BLEU, which requires Stage4/6 free-running waveform output.
