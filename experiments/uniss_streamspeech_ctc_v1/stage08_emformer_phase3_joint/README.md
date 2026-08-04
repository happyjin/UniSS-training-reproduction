# Stage08: Emformer + Phase3 joint repair

This isolated experiment keeps the proven causal Emformer baseline and repairs
the disconnected Stage03b/Stage06 optimization path in two steps.

## Step 1: frozen Phase3

One shared Emformer forward is supervised by:

```text
4 * ASR CTC
+ 4 * NAR-S2TT CTC
+ 8 * AR-S2TT CE
+ 0.5 * frozen Phase3 target NLL
+ 1e-4 * B1 residual MSE
```

The Stage03b model initializes the Emformer, CTC heads and AR decoder. Stage04
initializes the frozen B2 bridge, while Stage06 iteration 600 initializes the B1
residual. Only the last four Emformer layers, output norm, endpoint heads, AR
decoder and B1 residual are trainable. Qwen and BiCodec remain immutable.

Step 2 adds Qwen LoRA and offline Phase3 replay only after Step 1 passes its
fixed checkpoint gate. Both steps use new checkpoint, log, TensorBoard and
report directories and never overwrite Stage03--07 artifacts.
