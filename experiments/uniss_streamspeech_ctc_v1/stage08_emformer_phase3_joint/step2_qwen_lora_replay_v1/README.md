# Stage08 Step2: Qwen LoRA + offline replay (research-only)

Step1-R iteration 350 improved the fixed bidirectional mean BLEU to 20.6985,
but did not pass the formal Step2 hard gate. At the user's request, this
isolated Step2 continues only to validate the downstream pipeline and measure
whether limited Qwen adaptation is promising. It must not be reported as a
formally unlocked or quality-qualified result.

The selected Step1-R streaming encoder and bridge are frozen. Qwen's original
weights are also frozen; rank-8 LoRA adapters are added only to all `q_proj`
and `v_proj` layers. Each balanced-direction training example uses the same
Phase3 performance target twice:

```text
0.70 * NLL(predicted streaming Emformer/B1 embeddings)
+ 0.30 * NLL(original offline source_glm tokens)
```

The second term is real Phase3 replay from the immutable Stage-A source
manifest, not a duplicated streaming input. The launcher requires an explicit
`--step2-research-only-override`, uses new checkpoint/log/TensorBoard paths,
and refuses to reuse an existing result directory.

Run the 8-GPU job:

```bash
bash experiments/uniss_streamspeech_ctc_v1/stage08_emformer_phase3_joint/step2_qwen_lora_replay_v1/run_megatron_8gpu.sh
```

For a two-iteration smoke, override `RUN_NAME`, `TRAIN_ITERS`, interval and
warmup variables. After training, the fixed probe launcher evaluates equal
EN→ZH and ZH→EN partitions and compares them with Step1-R iteration 350:

```bash
bash experiments/uniss_streamspeech_ctc_v1/stage08_emformer_phase3_joint/step2_qwen_lora_replay_v1/run_probe_8gpu.sh
```

Only a later rerun whose prerequisite Step1 checkpoint passes both formal BLEU
thresholds may drop the research-only designation.
