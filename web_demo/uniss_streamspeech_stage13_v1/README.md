# Stage13 public UniSS-Stream research demo

This no-login Gradio page exposes the exact Stage09--12 research pipeline in a
new directory. It supports upload replay and microphone chunk streaming,
displays causal CTC ASR, CTC target text, Qwen text, every WAIT/WRITE and every
semantic rejection, then provides continuous target audio, a WAIT timeline and
left-source/right-translation stereo audio.

The page is intentionally explicit that the model has not passed quality or
subsecond computation-aware gates. ZH→EN may use a final offline safety
fallback, which is shown in the status rather than hidden.

Launch on one idle GPU:

```bash
bash web_demo/uniss_streamspeech_stage13_v1/launch_tmux.sh
```
