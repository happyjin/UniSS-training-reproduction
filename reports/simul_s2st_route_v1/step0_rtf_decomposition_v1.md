# Step 0 — streaming S2ST wall-clock decomposition

> Run `step0_rtf_decomposition_v1` · 2026-08-09T03:06:38+0000 · research only.
> Baseline pass is unpatched; the instrumented pass adds CUDA synchronisation at every
> span boundary, so treat the baseline for RTF and the instrumented pass for shares.

## 1. Headline

| Pass | Samples | Source s | Wall s | Compute RTF (pooled) | First audio NCA | First audio CA |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 16 | 97.7 | 615.1 | 6.30 | 4584 ms | 33538 ms |
| instrumented | 16 | 97.7 | 618.0 | 6.33 | 4584 ms | 33706 ms |

Instrumentation overhead on total wall clock: **0.5%**.

## 2. Where the wall clock goes

| Bucket | Seconds | Share of streaming wall | RTF contribution | What it is |
|---|---:|---:|---:|---|
| `qwen_ar_decode` | 324.55 | 52.5% | 3.322 | Qwen autoregressive WRITE generation (text + 50 Hz BiCodec semantic) |
| `offline_fallback` | 214.61 | 34.7% | 2.197 | Final-only offline safety generation when no WRITE was accepted |
| `qwen_prefill_source` | 49.39 | 8.0% | 0.505 | Qwen KV-cache append of START_GLM + source embeddings + END_GLM |
| `source_runtime` | 14.81 | 2.4% | 0.152 | Stage09 chunk-causal source frontend (mel + Emformer + CTC + B1 + policy) |
| `qwen_prefill_wait` | 13.79 | 2.2% | 0.141 | Qwen KV-cache append of a single WAIT token |
| `codec_stream_push` | 0.52 | 0.1% | 0.005 | Streaming BiCodec wrapper (window selection, holdback, crossfade) |
| `session_setup` | 0.43 | 0.1% | 0.004 | Session construction incl. Qwen streaming prompt prefill |
| `result_io` | 0.22 | 0.0% | 0.002 | Final WAV/JSON writing and stereo alignment |
| `session_push_other` | 0.12 | 0.0% | 0.001 |  |

## 3. Full call tree

| Path | Calls | Inclusive s | Exclusive s | ms/call | Share (inclusive) |
|---|---:|---:|---:|---:|---:|
| `session_push` | 619 | 618.01 | 0.12 | 998.40 | 100.0% |
| &nbsp;&nbsp;&nbsp;&nbsp;`codec_stream_push` | 39 | 0.52 | 0.01 | 13.31 | 0.1% |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`codec_vocoder` | 39 | 0.51 | 0.51 | 13.17 | 0.1% |
| &nbsp;&nbsp;&nbsp;&nbsp;`offline_fallback` | 9 | 214.61 | 214.61 | 23846.04 | 34.7% |
| &nbsp;&nbsp;&nbsp;&nbsp;`qwen_ar_decode` | 71 | 324.55 | 1.44 | 4571.13 | 52.5% |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`logits_block_collapse` | 10681 | 0.53 | 0.53 | 0.05 | 0.1% |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`logits_repetition_penalty` | 10681 | 49.15 | 49.15 | 4.60 | 8.0% |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`parse_write_tokens` | 71 | 0.01 | 0.01 | 0.14 | 0.0% |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`qwen_forward_ids` | 10752 | 273.41 | 273.41 | 25.43 | 44.2% |
| &nbsp;&nbsp;&nbsp;&nbsp;`qwen_prefill_source` | 618 | 49.39 | 0.04 | 79.91 | 8.0% |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`qwen_forward_embeds` | 618 | 18.15 | 18.15 | 29.37 | 2.9% |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`qwen_forward_ids` | 1236 | 31.19 | 31.19 | 25.24 | 5.0% |
| &nbsp;&nbsp;&nbsp;&nbsp;`qwen_prefill_wait` | 547 | 13.79 | 0.01 | 25.21 | 2.2% |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`qwen_forward_ids` | 547 | 13.78 | 13.78 | 25.19 | 2.2% |
| &nbsp;&nbsp;&nbsp;&nbsp;`result_io` | 16 | 0.22 | 0.22 | 13.93 | 0.0% |
| &nbsp;&nbsp;&nbsp;&nbsp;`source_runtime` | 619 | 14.81 | 0.51 | 23.92 | 2.4% |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`src_b1_bridge` | 618 | 0.70 | 0.70 | 1.13 | 0.1% |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`src_ctc_head` | 1236 | 0.10 | 0.10 | 0.08 | 0.0% |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`src_encoder_infer` | 618 | 13.11 | 13.11 | 21.21 | 2.1% |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`src_feature_projection` | 619 | 0.34 | 0.18 | 0.55 | 0.1% |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`src_mel_spectrogram` | 619 | 0.17 | 0.17 | 0.27 | 0.0% |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`src_output_norm` | 618 | 0.02 | 0.02 | 0.04 | 0.0% |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`src_policy_update` | 618 | 0.03 | 0.03 | 0.04 | 0.0% |
| `session_setup` | 16 | 0.43 | 0.00 | 27.09 | 0.1% |
| &nbsp;&nbsp;&nbsp;&nbsp;`qwen_forward_ids` | 16 | 0.43 | 0.43 | 26.86 | 0.1% |

## 3b. Qwen forward cost isolated from the pipeline

One-token forward against a growing KV cache — this is what a lambda-shaped cache shrinks.

| KV cache length | ms per forward |
|---:|---:|
| 0 | 24.36 |
| 128 | 24.60 |
| 512 | 24.28 |
| 1024 | 24.91 |
| 2048 | 24.90 |
| 4096 | 24.61 |

New positions per forward at a fixed cache length of 1024 — this is what a non-autoregressive head exploits.

| New tokens in one forward | ms per forward | ms per token | Speed-up vs 1-token steps |
|---:|---:|---:|---:|
| 1 | 24.60 | 24.605 | 1.0x |
| 4 | 27.92 | 6.980 | 3.5x |
| 16 | 27.86 | 1.741 | 14.1x |
| 64 | 28.00 | 0.437 | 56.2x |
| 256 | 29.12 | 0.114 | 216.3x |

## 4. Per-sample (baseline pass)

| Sample | Direction | Source s | Wall s | RTF | NCA | CA | Writes ok/rej | Fallback |
|---|---|---:|---:|---:|---:|---:|---:|:---:|
| `HQ-Conversations_0000000233` | cmn->eng | 10.64 | 79.48 | 7.47 | 10640 ms | 79449 ms | 0/3 | yes |
| `emilia_zh_0003916242` | cmn->eng | 6.04 | 17.36 | 2.87 | 6040 ms | 17352 ms | 0/3 | yes |
| `emilia_zh_0003959991` | cmn->eng | 8.42 | 53.59 | 6.36 | 8420 ms | 53576 ms | 0/8 | yes |
| `emilia_zh_0004004800` | cmn->eng | 4.46 | 14.62 | 3.28 | 4460 ms | 14614 ms | 0/2 | yes |
| `emilia_zh_0004056116` | cmn->eng | 10.14 | 144.16 | 14.22 | 10140 ms | 144129 ms | 0/9 | yes |
| `emilia_zh_0004108036` | cmn->eng | 4.86 | 83.25 | 17.13 | 4860 ms | 83230 ms | 0/4 | yes |
| `emilia_zh_0004151415` | cmn->eng | 5.52 | 17.74 | 3.21 | 5520 ms | 17728 ms | 0/4 | yes |
| `emilia_zh_0004188063` | cmn->eng | 3.42 | 36.84 | 10.77 | 3420 ms | 36832 ms | 0/3 | yes |
| `NCSSD_R_EN_0000000315` | eng->cmn | 6.00 | 12.29 | 2.05 | 880 ms | 4326 ms | 2/8 | no |
| `CommonVoice_EN_0000032073` | eng->cmn | 5.18 | 11.66 | 2.25 | 1520 ms | 4580 ms | 2/0 | no |
| `CommonVoice_EN_0000061461` | eng->cmn | 9.20 | 58.57 | 6.37 | 9200 ms | 58558 ms | 0/3 | yes |
| `CommonVoice_EN_0000157609` | eng->cmn | 4.38 | 10.25 | 2.34 | 1040 ms | 4802 ms | 5/0 | no |
| `CommonVoice_EN_0000190398` | eng->cmn | 4.20 | 10.52 | 2.50 | 1520 ms | 5231 ms | 2/0 | no |
| `CommonVoice_EN_0000223245` | eng->cmn | 6.14 | 13.92 | 2.27 | 1520 ms | 2206 ms | 6/0 | no |
| `CommonVoice_EN_0000251737` | eng->cmn | 3.32 | 14.91 | 4.49 | 1040 ms | 2103 ms | 4/0 | no |
| `CommonVoice_EN_0000287413` | eng->cmn | 5.78 | 35.94 | 6.22 | 3120 ms | 7897 ms | 2/1 | no |

## 5. Text quality on the same samples

| Direction | Samples | Text-BLEU | chrF |
|---|---:|---:|---:|
| eng->cmn | 8 | 11.45 | 10.59 |
| cmn->eng | 8 | 28.78 | 52.49 |

## 6. Configuration

```json
{
  "device": "cuda:0",
  "samples_per_direction": 8,
  "min_source_seconds": 3.0,
  "max_source_seconds": 12.0,
  "ingress_ms": 160,
  "max_write_tokens": 384,
  "stage09": {
    "dataset_index": "/opt/dlami/nvme/jasonleeeli/projects/UniSS/data/processed/uniss_streamspeech_ctc_v1/stage02_ctc_probe/dataset_index.json",
    "source_manifest": "/opt/dlami/nvme/jasonleeeli/projects/UniSS/data/processed/simul_uniss_subsecond_v1/pilot_15shard/stage_a_source/stage_a_source_manifest.jsonl",
    "tokenizer_dir": "/opt/dlami/nvme/jasonleeeli/projects/UniSS/data/processed/uniss_streamspeech_ctc_v1/stage01_data/tokenizers",
    "stage03b_checkpoint": "/opt/dlami/nvme/jasonleeeli/projects/UniSS/checkpoints/uniss_streamspeech_ctc_v1/stage03b_ar_s2tt_b16_v3/best.pt",
    "historical_stage_b_checkpoint": "/opt/dlami/nvme/jasonleeeli/projects/UniSS/checkpoints/simul_uniss_subsecond_v3/stage_b_v3_balanced_hidden_15shard_v1/candidates/step_008000.pt",
    "stage04_checkpoint": "/opt/dlami/nvme/jasonleeeli/projects/UniSS/checkpoints/uniss_streamspeech_ctc_v1/stage04_b2_phase3_endpoint_v1/best.pt",
    "stage06_checkpoint": "/opt/dlami/nvme/jasonleeeli/projects/UniSS/checkpoints/uniss_streamspeech_ctc_v1/stage06_b1_megatron_v2/iter_0000600",
    "step1_checkpoint": "/opt/dlami/nvme/jasonleeeli/projects/UniSS/checkpoints/uniss_streamspeech_ctc_v1/stage08_step1_repair_balanced_p3w2_zhen1p25_v1/iter_0000350",
    "step2_checkpoint": "/opt/dlami/nvme/jasonleeeli/projects/UniSS/checkpoints/uniss_streamspeech_ctc_v1/stage08_step2_qwen_lora_replay_r8_replay30_research_v1/iter_0000100",
    "codebook_model": "/opt/dlami/nvme/jasonleeeli/projects/UniSS/pretrained_models/UniSS/glm4_tokenizer",
    "phase3_model": "/opt/dlami/nvme/jasonleeeli/projects/UniSS/checkpoints/exported_hf/qwen0p5b_phase3_unist198_iter_0009075_hf",
    "device": "cuda:0",
    "confirmations": "2",
    "lagging_k": "0"
  },
  "stage11": {
    "speech_tokenizer_path": "/opt/dlami/nvme/jasonleeeli/projects/UniSS/pretrained_models/UniSS",
    "output_root": "/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_streamspeech_ctc_v1/stage11_streaming_audio_v1",
    "max_write_tokens": "384",
    "codec_left_context_tokens": "50",
    "codec_holdback_tokens": "5",
    "codec_overlap_ms": "80.0",
    "semantic_unique_ratio_min": "0.1",
    "semantic_max_run": "16"
  }
}
```
