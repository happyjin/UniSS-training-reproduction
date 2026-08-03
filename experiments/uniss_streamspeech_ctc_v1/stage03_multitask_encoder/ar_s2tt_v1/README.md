# Stage03b: StreamSpeech-style AR-S2TT joint supervision

The CTC-only causal encoder improves ASR strongly, but NAR-S2TT unigram recall
plateaus far below the 40% feasibility gate.  This is an expected ablation: the
StreamSpeech paper uses an AR-S2TT cross-entropy objective with weight `8.0` in
the same joint model, not two isolated CTC objectives.

This sub-stage adds:

- a shared four-layer causal Transformer translation decoder;
- language-specific English/Chinese target embeddings and output projections;
- teacher-forced AR-S2TT CE weight `8.0`;
- the existing ASR CTC and NAR-S2TT CTC weights `4.0 + 4.0`.

It initializes the causal encoder and CTC heads from the completed Stage03 best
checkpoint.  Qwen and the historical Phase3 checkpoint remain untouched; this
decoder is an endpoint-training auxiliary and can later be replaced by the B2
Phase3 bridge.

