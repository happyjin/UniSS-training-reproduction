# Stage11: incremental Phase3 semantic + Streaming BiCodec

Stage11 converts accepted Stage10 Micro-WRITEs into playable target audio. It
uses the existing UniSS BiCodec and preserves the 32 global speaker tokens.
Semantic spans are decoded only when they are structurally complete and pass
anti-collapse gates. The decoder retains left semantic context, holds back the
unstable tail and crossfades chunk boundaries.

Outputs are isolated under:

```text
eval_outputs/uniss_streamspeech_ctc_v1/stage11_streaming_audio_v1/
```

Each completed session writes continuous target audio, a WAIT-aligned target
timeline, a stereo file (`left=source`, `right=translation`) and full event
JSON. This is the first stage where the current Stage08 line can be listened to
as simultaneous speech-to-speech.

## Executed EN→ZH smoke

- first CTC WRITE: 560 ms;
- first accepted translated audio: 880 ms NCA / 5164 ms computation-aware;
- accepted/rejected audio WRITEs: 2 / 8;
- source/target audio: 6.00 / 2.26 seconds;
- every generated WAV is finite, non-empty PCM.

The result proves the audio plumbing but not acceptable translation quality.
Most early Qwen WRITEs are structurally incomplete, and the accepted text is
repetitive. Stage12 reports this explicitly instead of hiding rejected events.
