# Validation record

Validated on 2026-07-27 with:

```text
model = Phase3 full198 iter_0009075
mode = Quality
GPU = NVIDIA H200
gradio = 5.49.1
torch = 2.6.0+cu124
transformers = 4.53.1
```

## Direct engine validation

Chinese to English:

```text
sample = magicdata_0000000001
reference transcription = 我想用百度搜索短信
demo transcription = 我想用百度搜索短信
demo translation = I want to use Baidu to search for text messages.
generated audio = 6.76 seconds
warnings = none
```

English to Chinese:

```text
sample = gigaspeech_podcast_0000000003
reference transcription = Leila Green is a trauma surgeon in Houston.
demo transcription = Liligreeen is a trauma surgeon in Houston.
demo translation = 李利格林是休斯顿的一名创伤外科医生。
generated audio = 4.34 seconds
warnings = none
```

Both runs returned non-empty Phase3-owned transcription, translation, and a
playable 16 kHz WAV. Runtime files remain below the ignored
`runtime_outputs/` directory.

## Public Gradio validation

- Password-protected Gradio share tunnel created successfully.
- The public page returned HTTP 200 through the external tunnel.
- `gradio_client` without credentials was rejected.
- Authenticated `/translate_phase3_quality` returned six outputs:
  transcription, translation, generated WAV, result JSON, status, and chat
  history.
- The authenticated public request reproduced the Chinese sample result above.

The live URL and credentials are intentionally not committed. They are written
with mode 0600 to `access_info.json`; the temporary URL is also written to
`public_url.txt`.
