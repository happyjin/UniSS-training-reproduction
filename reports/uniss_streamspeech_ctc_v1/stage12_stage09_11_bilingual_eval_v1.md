# Stage12 Stage09–11 simultaneous S2ST evaluation

> Research-only. The pipeline is runnable and demoable; upstream quality gates remain unmet.

| Direction | BLEU | chrF | First WRITE | First audio NCA | First audio CA | Valid/rejected | Fallback | Compute RTF |
|---|---:|---:|---:|---:|---:|---:|:---:|---:|
| eng->cmn | 2.26 | 3.88 | 560 ms | 880 ms | 5164 ms | 2/8 | no | 2.23 |
| cmn->eng | 14.35 | 51.65 | 2160 ms | 10640 ms | 79628 ms | 0/3 | yes | 7.49 |

## Verdict

- Stage09 true chunking, Stage10 KV-cache and Stage11 BiCodec audio are all connected.
- EN→ZH demonstrates first accepted audio at 880 ms NCA, but computation-aware latency is 5.16 s and text is repetitive.
- ZH→EN has no accepted online semantic WRITE; its playable audio is a clearly labeled final offline fallback.
- Therefore a public research demo is appropriate, but neither bilingual quality nor true subsecond wall-clock performance passes.
- The demo must expose fallback status, rejected WRITEs and NCA/CA separately.

## Listening artifacts

### eng->cmn

- continuous target: `/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_streamspeech_ctc_v1/stage11_streaming_audio_v1/stage11_streaming_audio_smoke_eng_cmn_v2/translation.wav`
- WAIT timeline: `/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_streamspeech_ctc_v1/stage11_streaming_audio_v1/stage11_streaming_audio_smoke_eng_cmn_v2/translation_timeline.wav`
- stereo left-source/right-translation: `/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_streamspeech_ctc_v1/stage11_streaming_audio_v1/stage11_streaming_audio_smoke_eng_cmn_v2/aligned_stereo.wav`

### cmn->eng

- continuous target: `/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_streamspeech_ctc_v1/stage11_streaming_audio_v1/stage11_streaming_audio_smoke_cmn_eng_v2/translation.wav`
- WAIT timeline: `/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_streamspeech_ctc_v1/stage11_streaming_audio_v1/stage11_streaming_audio_smoke_cmn_eng_v2/translation_timeline.wav`
- stereo left-source/right-translation: `/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_streamspeech_ctc_v1/stage11_streaming_audio_v1/stage11_streaming_audio_smoke_cmn_eng_v2/aligned_stereo.wav`
