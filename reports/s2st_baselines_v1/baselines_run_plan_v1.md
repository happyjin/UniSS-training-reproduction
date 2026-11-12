# External baselines for the single-speaker QABC paper: registration note v1

Written 2026-11-12 15:25 (Prague), before any baseline output was seen. Paper copy:
`paper/bayes_simul_s2st_v12_baselines/`. The original `bayes_simul_s2st_v11_gpt/` is not changed. This note is the
registration. A tracked copy is committed at `reports/s2st_baselines_v1/baselines_run_plan_v1.md`, because `paper/`
is untracked.

Rules:
- Every setting below is fixed now.
- Test sets (CVSS-T test, RealSI) are only scored. Nothing is tuned on them.
- Nothing in our own system changes because of these results.

## 1. Who runs what

| System | Benchmark | Runner | Grid / settings file |
|---|---|---|---|
| (a) SeamlessStreaming | RealSI sentence level, En→Zh and Zh→En | R1 | `paper/baselines_run_plan_seamless_v1.md` |
| (b) Hy-SimulMT-30B-A3B + our Talker (Qwen3-TTS 0.6B optional) | RealSI sentence level | R2 | `paper/baselines_run_plan_hysimulmt_v1.md` |
| (c) Qwen3-Omni-30B-A3B-Instruct native (its own Thinker, Talker and code2wav) | CVSS-T test, offline, both directions | this agent | §3 below |

For (a) and (b), the tier rule is the paper's rule. Within each SimulS2ST-Omni tier (budgets 1.18 / 1.98 / 2.46 / 2.98 s
for En→Zh, 1.18 / 1.92 / 2.40 / 2.98 s for Zh→En), report the operating point with the highest ASR-BLEU whose segment
LAAL (D1) fits the budget, and "--" where none fits.

## 2. Protocol: the paper's own code

R = `/opt/dlami/nvme/jasonleeeli/projects/UniSS`. PYE = `conda_envs/uniss-eval/bin/python`. PYV = `venvs/vllm_omni_030/bin/python`.

### Data
- RealSI: `R/data/raw/realsi/RealSI_data/data/{en2zh,zh2en}`.
- Sentences come from `si_infer.realsi_items()`:
  - id = `<talk>_<k:03d>`;
  - audio = the talk wav cut at the segment's start/end, 16 kHz;
  - reference = `trg_text`.
- Sizes: En→Zh 346, Zh→En 431.

### Talker, for systems that output text (b)
1. Write a blocks file with one row per sentence:
   `{"id","translation","blocks":[{"write_ms","text","final"}]}`.
   - `write_ms` = source audio read when the text was committed (computation not counted, as in the paper).
   - The last block has `final: true` and `write_ms` = the sentence duration.
   - English blocks after the first start with a space.
   - Every id must be present.
2. Run `si_infer.py` in PYV:
   ```
   --model R/eval_outputs/uniss_qwen3_omni_v1/si_t1c_20261021T181440Z/autoext/export_1500/uniss --text-policy file --min-words 2 --blocks-file <f> --speed-policy block --history-ban 8 --gpu-mem 0.15 --num-shards 2
   ```
   - Use talker seeds default (20261021), 1 and 2.
   - The equivalent in-process caller is `reports/qabc_v10/talker_worker.py`.
3. A block plays from max(its write time, the end of the previous block).

### ASR-BLEU (speech only)
1. Run `PYE reports/uniss_si_v4/speech_only.py <run dir>`.
2. Run `PYE -m evaluation.asr_transcribe --input results_speechonly.jsonl --output asr_speech.jsonl --batch-size 4`.
   - English is transcribed with Whisper-large-v3 and Chinese with Paraformer-large.
   - Set `HF_HOME=/opt/dlami/nvme/jasonleeeli/cache/huggingface` and `MODELSCOPE_CACHE=/opt/dlami/nvme/jasonleeeli/cache/modelscope`.
3. Score with `evaluation/text_metrics.corpus_bleu`:
   - normalise first: English NFKC, lowercase, punctuation removed; Chinese NFKC, t2s, punctuation removed;
   - sacreBLEU tokenizer `zh` for Chinese, `13a` for English;
   - a missing ASR row counts as an empty hypothesis, and ASR coverage must be at least 99%.

Text BLEU = `translated_text` against `translation_ref`, with the same function.

### Latency (D1, D0w)
- **D1** = the paper's segment LAAL, `collect_results.realsi_points` / `si_metrics.laal`.
  - A unit (Chinese character plus Latin words, or English word) is timed at the playback start (`start_ms`) of its block.
  - LAAL is averaged over sentences.
- **D0w** = SimulEval's word-level LAAL: the same units timed at their block's emission (`write_ms`), without the queue.
- Both are computed by `eval_outputs/mist_w3po_v1/speech_laal_v1/analysis_20261112/laal_defs.py` `sentence_defs()` on `results.jsonl` rows.
- A RealSI point is the mean over the three talker seeds. A system with its own speech, such as (a), has one run per point.

## 3. (c) Qwen3-Omni native on CVSS-T test: one point, fixed now

- **Model:** `R/pretrained_models/Qwen3-Omni-30B-A3B-Instruct`, served by vLLM-Omni 0.30.0 in vllm_omni_030 (no installs).
  - Its own three stages: thinker → talker (with code predictor) → code2wav.
  - Script: `experiments/s2st_baselines_v1/qwen3omni_native_cvss.py`, which is P1-a (`p1_vllm_omni.py`) on the paper's CVSS-T items.
- **Items:** `R/data/cvss_t/si_items_v1/items_{en2zh,zh2en}.jsonl`, 4,897 per direction. These are the same items and references as Table 1's own rows.
- **Prompt:** user turn = [source audio, the instruction] with no system prompt, built by `Qwen3OmniMoeProcessor.apply_chat_template`. The instruction is the repo's fixed one (`phase_d_e2e.INSTRUCTION`):
  - En→Zh: "Listen to the English speech and translate it into Chinese. Reply with the Chinese translation only."
  - Zh→En: "Listen to the Chinese speech and translate it into English. Reply with the English translation only."
- **Decoding:**
  - Thinker: greedy, max 256 tokens, stops at `<|im_end|>`.
  - Talker: the official sampling (temperature 0.9, top-k 50, repetition penalty 1.05, codec EOS 2150).
  - code2wav: greedy.
  - Seed 20261020. Preset voice "Ethan": Qwen3-Omni cannot clone the source voice.
- **Scoring** (our Table 1 protocol, as for our own rows):
  - output speech 24 kHz → 16 kHz;
  - ASR-BLEU with `speech_only.py` + `asr_transcribe` (`--batch-size 16`) + `corpus_bleu`;
  - Text-BLEU = the thinker's text;
  - A.PCP and SIM-O with `reports/uniss_si_v4/cvss_table1_metrics.py`: AutoPCP multilingual v2, and wavlm-base-plus-sv against the paired real target speech, the same scale as our bracketed SIM-O.
- **Report:** one row in Table 1, marked "ours run, our protocol". No other setting is tried. Utterances without speech count as empty ASR hypotheses.
- **Compute:** shards over both directions (`items[k::4]`), with one engine per pair of non-training cards (thinker on one card, talker and code2wav on the other). Engines start at least 60 s apart, and each card stays at or below 135 GiB.

## 4. Literature numbers placed in the paper (no runs)

- **UniSS Table 1** (arXiv 2509.21144v1, the only version):
  - GPT-4o 31.64 / 19.27;
  - Qwen2.5-Omni 7.10 / 22.66;
  - 3-stage cascade 25.02 / 16.62;
  - 2-stage cascade 26.94 / 20.86;
  - SeamlessM4T-medium 14.53 / 14.36 (ASR-BLEU En→Zh / Zh→En; Text-BLEU and A.PCP as in its table).
- **LiveInterpret 2.0** (arXiv 2507.17527 v1–v3 identical), Table 2, sentence-level speech-to-speech LAAL:
  - En→Zh 3.12 s, Zh→En 3.79 s.
  - These are paired with SimulS2ST-Omni's RealSI ASR-BLEU, 29.09 / 22.19, as SimulS2ST-Omni v2 does itself (Fig. 3, footnote 1).
  - The LAAL is its own computation on its own sentence-level test sets, so the pairing is marked approximate.
