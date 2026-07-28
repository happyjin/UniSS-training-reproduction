# UniSS Phase3 CVSS-T zh/en evaluation v1

This directory is the isolated implementation of the CVSS-T evaluation plan in:

```text
docs/uniss_training_reproduction/simuls2st_omni_cvss_t_data_preparation_and_evaluation_plan.md
```

It reproduces the objective CVSS-T protocol from UniSS Table 1 without changing
historical UniST, Phase2, Phase3, or simultaneous-streaming experiment scripts.

## Frozen evaluation target

```text
Model: checkpoints/exported_hf/qwen0p5b_phase3_unist198_iter_0009075_hf
Pairs: 4,897
Directions: EN->ZH and ZH->EN
Modes: performance (P) and quality (Q)
```

UniSS decoding parameters:

```text
temperature=0.7
top_k=-1
top_p=0.8
repetition_penalty=1.1
```

Objective metrics:

```text
Speech-BLEU
Text-BLEU
AutoPCP
SLC-0.2
SLC-0.4
UTMOS
```

The paper's subjective MOS protocol requires six bilingual human raters and is
tracked separately; it is not represented as an automatic objective score.

## Non-overwriting data roots

```text
Raw CVSS-T:
/opt/dlami/nvme/jasonleeeli/CVSS/extracted/cvss_t_zh_en_v1.0

Raw Common Voice v4 test subset:
/opt/dlami/nvme/jasonleeeli/CVSS/source/common_voice_v4_zh-CN

Canonical audio:
/opt/dlami/nvme/jasonleeeli/CVSS/canonical_16k/cvss_t_zh_en_test

Tokenized parquet:
/opt/dlami/nvme/jasonleeeli/CVSS/tokenized/cvss_t_zh_en_v1

Generated output:
eval_outputs/cvss_t_zh_en_phase3_full198_iter_0009075_v1
```

Every command refuses to overwrite an existing completed output unless its
explicit resume mode is enabled and its input identity matches.

## Current preparation state (2026-07-28)

Completed CPU stages:

```text
canonical pairs       4,897 / 4,897
source ZH duration    8.2432067 h (decoded canonical duration)
target EN duration    6.2916875 h
audio format          16 kHz / mono / PCM16
CVSS ID leakage       0 exact ID matches
normalized text       1,705 matching UniST training records
```

The text matches are dominated by CommonVoice, WenetSpeech and GigaSpeech
records. They do not prove byte-identical audio leakage, but they must be
disclosed with the final benchmark. Full audit:

```text
/opt/dlami/nvme/jasonleeeli/CVSS/audits/cvss_t_zh_en_vs_unist198_text_leakage.json
```

Tokenization is the first GPU stage. Its outputs are created outside Git and
are never mixed with historical UniST parquets.

## End-to-end commands

### 1. Rebuild canonical audio only when starting from raw data

```bash
bash experiments/evaluation/cvss_t_zh_en_phase3_v1/prepare_canonical_audio.sh
```

The current canonical data is already complete. The command is non-overwriting
and should not be rerun against the same output directory without deliberate
cleanup of that isolated cache.

### 2. Tokenize 4,897 pairs on 8 GPUs

```bash
EVAL_GPU_LIST=0,1,2,3,4,5,6,7 \
  bash experiments/evaluation/cvss_t_zh_en_phase3_v1/tokenize_8gpu.sh
```

This encodes both waveforms once and writes two directional parquets. Each
direction uses its own source waveform's first 32 BiCodec global tokens for
speaker conditioning.

Expected manifests:

```text
/opt/dlami/nvme/jasonleeeli/CVSS/tokenized/cvss_t_zh_en_v1/manifests/zh_en/unist_test_all.jsonl
/opt/dlami/nvme/jasonleeeli/CVSS/tokenized/cvss_t_zh_en_v1/manifests/en_zh/unist_test_all.jsonl
```

### 3. Run the mandatory 10-pair smoke evaluation

After tokenization and while a GPU is free:

```bash
EVAL_GPU_LIST=0 \
  bash experiments/evaluation/cvss_t_zh_en_phase3_v1/run_smoke.sh
```

The smoke run executes the complete chain for both directions and both Q/P
modes: vLLM generation, generated-audio decoding, integrity validation,
Text-BLEU, Speech-BLEU, SLC, UTMOS, AutoPCP, and report generation. It is a
functional check only; its 10-pair scores must not be quoted as Table 1.

### 4. Run the formal UniSS Table 1 evaluation

```bash
EVAL_GPU_LIST=0,1,2,3,4,5,6,7 \
  bash experiments/evaluation/cvss_t_zh_en_phase3_v1/run_full_evaluation.sh
```

The two directions run sequentially, with all selected GPUs used as independent
data-parallel workers for each stage. Every per-sample GPU metric is resumable.
The formal output is:

```text
eval_outputs/cvss_t_zh_en_phase3_full198_iter_0009075_v1/
├── cvss_t_phase3_full_cmn_to_eng/
├── cvss_t_phase3_full_eng_to_cmn/
└── report/
    ├── cvss_t_phase3_table1_report.json
    └── cvss_t_phase3_table1_report.md
```

The report is only marked formal-complete when all 24 metric cells exist and
each contains 4,897 samples.

## Individual commands and resume behavior

Generation for one direction:

```bash
RESUME=1 EVAL_GPU_LIST=0,1,2,3,4,5,6,7 \
  bash experiments/evaluation/cvss_t_zh_en_phase3_v1/run_vllm_eval.sh \
  /opt/dlami/nvme/jasonleeeli/CVSS/tokenized/cvss_t_zh_en_v1/manifests/zh_en/unist_test_all.jsonl \
  eval_outputs/cvss_t_zh_en_phase3_full198_iter_0009075_v1/cvss_t_phase3_full_cmn_to_eng
```

Objective metrics for that direction:

```bash
EVAL_GPU_LIST=0,1,2,3,4,5,6,7 EXPECTED_PAIRS=4897 \
  bash experiments/evaluation/cvss_t_zh_en_phase3_v1/run_objective_metrics.sh \
  eval_outputs/cvss_t_zh_en_phase3_full198_iter_0009075_v1/cvss_t_phase3_full_cmn_to_eng \
  'cmn->eng'
```

The integrity gate runs before metrics and rejects duplicate/missing Q/P rows,
wrong direction or synthetic flags, unreadable official WAVs, generated audio
that aliases an official source/reference WAV, or incomplete formal counts.

## Paper comparison boundary

- Main report: UniSS arXiv:2509.21144 Table 1 sampling protocol.
- SimulS2ST-Omni greedy unified re-score is a separate protocol and must use a
  separate output/report rather than being merged into this table.
- ZH->EN uses real Common Voice input and synthetic CVSS-T reference speech.
- EN->ZH is the reversed synthetic-source benchmark; it is retained to match
  the paper but is not evidence of real-English-input generalization.
- Subjective MOS is not automatic. Reproducing it requires six bilingual
  raters and a separately randomized webMUSHRA evaluation.
