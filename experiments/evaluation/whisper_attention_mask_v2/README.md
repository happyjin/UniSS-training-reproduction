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

The run manifest intentionally contains only full evaluation runs whose
published reports used the affected batched Whisper protocol.
