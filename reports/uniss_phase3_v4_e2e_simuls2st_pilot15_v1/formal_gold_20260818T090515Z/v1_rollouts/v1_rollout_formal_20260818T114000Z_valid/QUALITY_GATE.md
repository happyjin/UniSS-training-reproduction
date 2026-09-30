# V1 rollout strict quality gate and strata

- Status: **passed**
- Records: **13,469**
- Clean: **3,878** (28.79%)
- Noisy content retained: **6,288** (46.68%)
- Quarantined protocol errors: **3,303** (24.52%)
- Rollout-dependent supervision retained: **10,166** (75.48%)
- Final EOS sample rate: **1.0000**

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
| cmn | 5,708 | 2,118 | 2,216 | 1,374 | 0.2502 |
| eng | 7,761 | 1,760 | 4,072 | 1,929 | 0.3898 |
