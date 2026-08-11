# Dense-aligned pilot15 validation-best true-streaming demo

This isolated package reuses the verified causal runtime from
`web_demo/true_subsecond_pilot15_streaming_v1` while loading the dense-aligned
Megatron checkpoint `iter_0000500`.  It does not modify or overwrite any older
demo, experiment, checkpoint, or output directory.

The selected checkpoint is the minimum validation objective when every term is
scored with the final curriculum weights.  The runtime contract is:

- PCM arrives in source-clock order;
- WhisperVQ emits append-only codes from 160 ms blocks with 80 ms bounded right
  context;
- Qwen retains append-only source KV state and branches per WRITE decision;
- BiCodec decodes append-only semantic microblocks;
- Gradio upload simulates real-time source arrival, but ordinary Gradio audio is
  not browser WebRTC packet streaming.

Run train/validation audio regression:

```bash
/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-offline-demo/bin/python \
  -m web_demo.dense_aligned_pilot15_streaming_v1.validate_samples \
  --train-count 2 --valid-count 2 --chunk-ms 320
```

Launch public Gradio:

```bash
web_demo/dense_aligned_pilot15_streaming_v1/launch_public_tmux.sh
web_demo/dense_aligned_pilot15_streaming_v1/status.sh
```

