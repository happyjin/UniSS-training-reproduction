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
