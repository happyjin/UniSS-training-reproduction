# Stage08: Emformer + Phase3 joint repair

This isolated experiment keeps the proven causal Emformer baseline and repairs
the disconnected Stage03b/Stage06 optimization path in two steps.

## Step 1: frozen Phase3

One shared Emformer forward is supervised by:

```text
4 * ASR CTC
+ 4 * NAR-S2TT CTC
+ 8 * AR-S2TT CE
+ 0.5 * frozen Phase3 target NLL
+ 1e-4 * B1 residual MSE
```

The Stage03b model initializes the Emformer, CTC heads and AR decoder. Stage04
initializes the frozen B2 bridge, while Stage06 iteration 600 initializes the B1
residual. Only the last four Emformer layers, output norm, endpoint heads, AR
decoder and B1 residual are trainable. Qwen and BiCodec remain immutable.

Step 2 normally adds Qwen LoRA and offline Phase3 replay only after Step 1
passes its fixed checkpoint gate. Step1-R did not pass that gate. A separately
named `step2_qwen_lora_replay_v1` path is therefore allowed only as an explicit
research-only pipeline validation at the user's request; it freezes Step1-R,
requires a command-line override, and cannot be cited as a formally unlocked
result. Both steps use new checkpoint, log, TensorBoard and report directories
and never overwrite Stage03--07 artifacts.

The Step 1 Megatron launch keeps the project's proven single-node layout:

```text
8-way data parallel, TP=PP=1
micro batch 1, global batch 128 (16 gradient-accumulation micro-batches)
1000 iterations, 1e-5 -> 1e-6 cosine learning rate, 200-step warmup
gradient clipping 0.5, validation/checkpoint every 100 iterations
```

Run a new isolated job with:

```bash
bash experiments/uniss_streamspeech_ctc_v1/stage08_emformer_phase3_joint/step1_frozen_qwen/run_megatron_8gpu.sh
```

`RUN_NAME`, `TRAIN_ITERS`, interval variables and `MASTER_PORT` may be
overridden for smoke tests. The launcher refuses to reuse an existing
checkpoint, TensorBoard or log path.

After the formal run finishes, evaluate every 100-step checkpoint on the same
fixed direction-balanced 32 validation rows used by Stage04 and Stage07:

```bash
bash experiments/uniss_streamspeech_ctc_v1/stage08_emformer_phase3_joint/step1_frozen_qwen/run_step1_gate_8gpu.sh
```

The gate selects the highest mean bidirectional BLEU checkpoint and requires
EN→ZH > 22.95 and ZH→EN > 22.46 before Step2 may enable Qwen LoRA and offline
Phase3 replay.
