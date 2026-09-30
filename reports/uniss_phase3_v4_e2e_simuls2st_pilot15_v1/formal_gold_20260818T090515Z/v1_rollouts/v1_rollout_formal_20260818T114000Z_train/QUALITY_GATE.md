# V1 rollout strict quality gate and strata

- Status: **passed**
- Records: **1,325,243**
- Clean: **385,051** (29.06%)
- Noisy content retained: **616,445** (46.52%)
- Quarantined protocol errors: **323,747** (24.43%)
- Rollout-dependent supervision retained: **1,001,496** (75.57%)
- Final EOS sample rate: **0.9999**

## Policy

- clean English WER <= 0.30
- clean Chinese CER <= 0.20
- noisy_content keeps structurally valid free-running errors for robustness training
- quarantine means malformed WRITE, early EOS, or missing final EOS
- quarantine remains eligible only for gold-source incremental MT and Phase3 replay

## Hard checks

- all_records_classified_once: **PASS**
- quarantine_rate_within_limit: **PASS**
- accepted_rollout_rate_meets_minimum: **PASS**
- final_eos_rate_meets_minimum: **PASS**
- every_language_retains_rollout_supervision: **PASS**

| source language | samples | clean | noisy | quarantine | weighted WER/CER |
|---|---:|---:|---:|---:|---:|
| cmn | 565,268 | 210,317 | 218,604 | 136,347 | 0.2515 |
| eng | 759,975 | 174,734 | 397,841 | 187,400 | 0.3914 |
