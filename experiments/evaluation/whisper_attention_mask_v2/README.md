# Whisper attention-mask v2 correction

This isolated evaluation reruns only English ASR and ZH-to-EN Speech-BLEU
from existing generated WAV files.  It does not regenerate model outputs,
decode audio, overwrite legacy metrics, or rerun Chinese Paraformer ASR.

The correction fixes a Transformers Whisper batching failure where padded
short utterances were decoded without an explicit attention mask.  Affected
transcripts repeated phrases until the decoder limit and made ZH-to-EN
Speech-BLEU invalid.

Each run writes:

```text
<existing-run>/metrics_whisper_attention_mask_v2/
├── asr_results_eng.jsonl
├── asr_results_eng.summary.json
├── speech_bleu_eng.json
├── verification.json
└── COMPLETE
```

Launch the full correction while leaving GPU 0 for the public demo:

```bash
experiments/evaluation/whisper_attention_mask_v2/launch_all_tmux.sh
```

The launcher uses two independent workers on each of GPUs 1--7.  Workers are
assigned disjoint manifest slots, and each GPU remains well below H200 memory
capacity with batch size 8.  Override `WHISPER_V2_GPU_LIST` if a different
allocation is required.

The run manifest intentionally contains only full evaluation runs whose
published reports used the affected batched Whisper protocol.

The default Whisper batch size is 8.  Any transcript that violates the
duration-aware length guard is automatically retried with batch size 1.  A
result that is still implausible after that retry is recorded as an explicit
empty, unintelligible hypothesis and remains in corpus BLEU instead of being
skipped or contaminating the metric with repeated text.  Retry and rejection
counts are recorded in `verification.json`.

After all runs finish, build a checked legacy/corrected comparison table with:

```bash
python experiments/evaluation/whisper_attention_mask_v2/summarize.py \
  --manifest experiments/evaluation/whisper_attention_mask_v2/runs.tsv \
  --repo-root "$PWD" \
  --output-json /tmp/whisper_attention_mask_v2_summary.json \
  --output-markdown /tmp/whisper_attention_mask_v2_table.md
```
