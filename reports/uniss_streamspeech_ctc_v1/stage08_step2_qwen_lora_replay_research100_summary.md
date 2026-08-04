# Stage08 Step2 Qwen-LoRA + offline replay research result

Date: 2026-08-04 UTC

## Status

The downstream Step2 pipeline is fully runnable, but the experiment remains
**research-only** because its Step1-R input did not pass the pre-registered
hard gate. The run cannot be presented as a formally qualified streaming
model.

Best research checkpoint:

```text
checkpoints/uniss_streamspeech_ctc_v1/stage08_step2_qwen_lora_replay_r8_replay30_research_v1/iter_0000100
```

TensorBoard events:

```text
runs/uniss_streamspeech_ctc_v1/stage08_step2_qwen_lora_replay_r8_replay30_research_v1
```

TensorBoard service on the current server:

```text
http://10.1.6.203:6028
tmux session: uniss_stage08_step2_tensorboard_6028
```

## Experiment

- input: Step1-R balanced checkpoint iteration 350;
- data: the same UniST 15-shard training line and immutable source manifest;
- sampling: exactly 50:50 EN→ZH / ZH→EN;
- streaming encoder, B1 residual and original Qwen weights: frozen;
- trainable parameters: dependency-free rank-8 LoRA on all Qwen `q_proj` and
  `v_proj` layers;
- objective: `0.70 * streaming Phase3 NLL + 0.30 * offline Phase3 replay NLL`;
- offline replay source: original `source_glm`, not predicted streaming tokens;
- Megatron: eight-way data parallel, micro batch 1, global batch 128;
- schedule: 100 iterations, `5e-5 -> 5e-6` cosine, 10-iteration warmup;
- checkpoint and validation interval: 25 iterations.

The run completed with zero skipped iterations and zero NaN iterations.

## NLL health

| Iteration | Validation streaming NLL | Validation replay NLL | LoRA B RMS |
|---:|---:|---:|---:|
| 25 | 4.0962 | 4.0824 | 3.1092e-4 |
| 50 | 4.1171 | 4.1001 | 4.9590e-4 |
| 75 | 4.1001 | 4.0850 | 5.6503e-4 |
| 100 | 4.1142 | 4.0985 | 5.8854e-4 |
| 100 final validation pass | 4.1047 | 4.0898 | 5.8854e-4 |

The adapters updated smoothly and replay remained finite, but NLL did not show
a monotonic improvement. Checkpoint selection therefore uses the fixed
generation probe rather than training loss.

## Fixed 32-row bidirectional probe

| Model | EN→ZH BLEU | ZH→EN BLEU | Mean BLEU | Δ Mean vs Step1-R | Compute RTF/source |
|---|---:|---:|---:|---:|---:|
| Step1-R iter350 | 21.2031 | 20.1939 | 20.6985 | — | 0.0901 |
| Step2 iter25 | 21.6803 | 19.5990 | 20.6396 | -0.0588 | 0.1068 |
| Step2 iter50 | 20.6664 | 19.9107 | 20.2885 | -0.4099 | 0.1064 |
| Step2 iter75 | 20.7785 | 19.8763 | 20.3274 | -0.3711 | 0.1085 |
| **Step2 iter100** | **21.8508** | **20.0660** | **20.9584** | **+0.2599** | **0.1048** |

Iteration 100 produced 32/32 non-empty outputs and a 100% `END_CONTENT` rate.
Its compute RTF/source p95 was 0.1503 and mean first-token wall time was 0.1669
seconds. The LoRA wrapper increased mean compute RTF by about 16.4% relative to
Step1-R, while remaining well below real time on this short probe.

## Formal gate gap

| Direction | Required | Step2 iter100 | Remaining gap |
|---|---:|---:|---:|
| EN→ZH | >22.9500 | 21.8508 | 1.0992 below |
| ZH→EN | >22.4600 | 20.0660 | 2.3940 below |

The result is a small positive hypothesis signal, not a gate pass. LoRA helped
EN→ZH but did not repair the weaker ZH→EN direction. The dominant unresolved
problem is still source/bridge alignment and direction-specific capacity, not
training instability.

## What happens after a real gate pass

1. Run the same Step2 as a formal 200--400 iteration experiment, select on the
   fixed bidirectional probe, and require both directions to improve rather
   than only the mean.
2. Run offline Phase3 regression on the unchanged offline dev/test protocol,
   plus ASR CTC, NAR-S2TT CTC and AR-S2TT probes, to prove that LoRA did not
   trade away offline quality or auxiliary tasks.
3. Attach the calibrated source/target CTC count policy and evaluate true
   chunk-by-chunk READ/WRITE behavior: AL, LAAL, DAL, ATD, StartOffset,
   first-useful-audio, compute RTF and revision/premature-WRITE rates.
4. Add the NAR BiCodec semantic head as a separate stage to reduce target audio
   generation RTF while retaining the original BiCodec global speaker tokens.
5. Only after the 15-shard quality/latency Pareto passes, reproduce the selected
   configuration on full198 and then run full dev/test/CVSS-T evaluation.

## What may still be validated before the gate passes

It is safe to continue in explicitly marked research-only directories:

- run a Step2-R directional adapter experiment with a modest ZH→EN objective
  weight or direction-specific LoRA, using Step2 iter100 only as a hypothesis
  baseline;
- connect iter100 to the CTC READ/WRITE policy to verify state handling,
  chunked inference, latency accounting and end-to-end audio plumbing;
- smoke the NAR semantic interface and codec continuity checks on a small
  subset.

These runs can reveal engineering failures and whether the downstream ideas
have any effect. They must not unlock full198 training, formal quality claims,
or overwrite the Step1/Step2 baselines. The preferred scientific fix remains
an alignment repair that raises both directions before relying on Qwen LoRA.

Detailed machine-readable comparison:

```text
reports/uniss_streamspeech_ctc_v1/stage08_step2_qwen_lora_replay_probe32_research_v1/comparison.json
```
