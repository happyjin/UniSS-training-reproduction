# Qwen-OmniSpeak (`training_pipelines/custom/qwen_omnispeak/`)

Custom-model package for the **Qwen-OmniSpeak** composite model described in
`QWEN_OMNISPEAK_MODEL_DESIGN.md`. Built on top of the **ms-swift** custom
plugin mechanism plus a self-contained, `trust_remote_code`-friendly
HuggingFace model.

## Architecture

```
Qwen Omni Thinker  ──>  DualRoute Bridge  ──>  Qwen3-TTS Talker (sliced)
(Qwen3-Omni or Qwen2.5-Omni)              └─> Code Predictor (5 layers)
                                           └─> Codec LM head
```

Supported Thinker choices:

| thinker | `thinker_family` | bridge hidden input |
|---------|------------------|---------------------|
| `Qwen/Qwen3-Omni-30B-A3B-Instruct` / compatible Qwen3-Omni | `qwen3_omni_moe` | hidden state index `24` (legacy default) |
| `Qwen/Qwen2.5-Omni-7B` | `qwen2_5_omni` | final/last hidden state |
| `Qwen/Qwen2.5-Omni-3B` | `qwen2_5_omni` | final/last hidden state |

Ten officially supported variants are available for every Thinker choice:

| variant | bridge type            | talker layers (out of 28) |
|---------|------------------------|---------------------------|
| **A**   | DualRoute MLP Bridge   | full-28                   |
| **B**   | DualRoute MLP Bridge   | top-16 (default)          |
| **C**   | DualRoute MLP Bridge   | top-12                    |
| **D**   | DualRoute MLP Bridge   | top-8                     |
| **E**   | DualRoute MLP Bridge   | top-4                     |
| **F**   | DualRoute Q-Former     | full-28                   |
| **G**   | DualRoute Q-Former     | top-16                    |
| **H**   | DualRoute Q-Former     | top-12                    |
| **I**   | DualRoute Q-Former     | top-8                     |
| **J**   | DualRoute Q-Former     | top-4                     |

## File layout

```
qwen_omnispeak/
├── README.md                            # this file
├── configuration_qwen_omnispeak.py     # all configs (single-file)
├── modeling_qwen_omnispeak.py          # all modeling code (single-file)
├── register_qwen_omnispeak.py          # ms-swift plugin (10 variants)
├── export_qwen_omnispeak.py            # build & export 10 variant directories
└── test_qwen_omnispeak.py              # smoke tests (forward + save/load)
```

The `modeling_qwen_omnispeak.py` and `configuration_qwen_omnispeak.py`
files are **kept as single top-level modules**. This is required because
`transformers`'s `trust_remote_code` loader resolves `auto_map` entries via a
strict `"<module_file>.<ClassName>"` two-segment shorthand — packages with
nested submodules cannot be referenced. The two files use *relative imports*
(`from .configuration_qwen_omnispeak import ...`) inside a `try` block so
they remain `python`-runnable while still being valid trust-remote-code
modules after export.

## End-to-end usage

### 1. Smoke test (no real weights required)

```bash
python training_pipelines/custom/qwen_omnispeak/test_qwen_omnispeak.py
```

Should print all ten variants passing, plus a `save/load` round-trip with
`max |Δ main_codec_logits| ≈ 0`.

### 2. Export the ten variants

```bash
python training_pipelines/custom/qwen_omnispeak/export_qwen_omnispeak.py \
    --thinker_model_path  Qwen/Qwen3-Omni-30B-A3B-Instruct \
    --tts_model_path      Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
    --output_root         ./exported/qwen_omnispeak \
    --variants            A B C D E F G H I J \
    --dtype               bfloat16
```

For Qwen2.5-Omni Thinker exports, use the same A-J variants; the exporter
infers `thinker_family=qwen2_5_omni` and defaults `hidden_state_selection=last`:

```bash
python training_pipelines/custom/qwen_omnispeak/export_qwen_omnispeak.py \
    --thinker_model_path  Qwen/Qwen2.5-Omni-7B \
    --tts_model_path      Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
    --output_root         ./exported/qwen_omnispeak_qwen25_7b \
    --variants            A B C D E F G H I J \
    --dtype               bfloat16
```

`Qwen/Qwen2.5-Omni-3B` is supported identically.

Output layout:

```
exported/qwen_omnispeak/
├── qwen_omnispeak_a/      # variant A directory (full-28 + MLP Bridge)
│   ├── config.json                 # contains auto_map for trust_remote_code
│   ├── model-*.safetensors         # full composite weights
│   ├── modeling_qwen_omnispeak.py # bundled code
│   ├── configuration_qwen_omnispeak.py
│   ├── tokenizer.json / processor / ...   # selected Qwen Omni processor
│   └── ...
├── qwen_omnispeak_b/      # top-16 + MLP Bridge
├── qwen_omnispeak_c/      # top-12 + MLP Bridge
├── qwen_omnispeak_d/      # top-8 + MLP Bridge
├── qwen_omnispeak_e/      # top-4 + MLP Bridge
├── qwen_omnispeak_f/      # full-28 + Q-Former Bridge
├── qwen_omnispeak_g/      # top-16 + Q-Former Bridge
├── qwen_omnispeak_h/      # top-12 + Q-Former Bridge
├── qwen_omnispeak_i/      # top-8 + Q-Former Bridge
└── qwen_omnispeak_j/      # top-4 + Q-Former Bridge
```

Each export is **self-contained**: subsequent training/inference does not
need access to either the selected Qwen Omni Thinker or Qwen3-TTS source weights.
The exported `config.json` records `thinker_family`, `thinker_model_id`,
`talker_model_id`, and `hidden_state_selection`.

### 3. Train via ms-swift

Pass the registration plugin via `--external_plugins`, then point ms-swift at
the exported directory. The `model_type` selects the variant.

```bash
swift sft \
    --external_plugins training_pipelines/custom/qwen_omnispeak/register_qwen_omnispeak.py \
    --model_type qwen_omnispeak_b \
    --model     ./exported/qwen_omnispeak/qwen_omnispeak_b \
    --template  qwen_omnispeak \
    --train_type full \
    --freeze_llm true \
    --freeze_vit true \
    --freeze_aligner false \
    ...
```

Stage-1 training (Bridge-only) is achieved by setting:

| flag                 | value     | effect                                              |
|----------------------|-----------|-----------------------------------------------------|
| `--freeze_llm`       | `true`    | freezes Thinker text backbone                       |
| `--freeze_vit`       | `true`    | freezes Thinker audio_tower / visual                |
| `--freeze_aligner`   | `false`   | trains audio proj / vision merger / **DualRoute Bridge** |

The talker side is in the `generator` group of the registered model_arch and
is therefore frozen by `freeze_llm/vit/aligner` alone. The bridge_alignment
pipeline additionally re-activates the residual code predictor and the bottom
talker layers via `--trainable_parameters model.bridge model.talker.code_predictor`
and `--trainable_parameters_regex 'model\.talker\.model\.layers\.[0-3]\.'`
(see `pipelines/bridge_alignment/common_args.sh`).

### 4. Inference

```bash
swift infer \
    --external_plugins training_pipelines/custom/qwen_omnispeak/register_qwen_omnispeak.py \
    --model_type qwen_omnispeak_b \
    --model     ./exported/qwen_omnispeak/qwen_omnispeak_b
```

## Forward signature

The exported model exposes a HF-compatible `forward()`:

```python
out = model(
    # one of the two thinker entry paths:
    input_ids=...,                      # raw tokens, run thinker on the fly  (set run_thinker=True)
    inputs_embeds=...,                  # ditto
    # ... or pre-computed thinker outputs:
    thinker_text_embeds=...,            # [B, T_text, 2048]
    thinker_hidden_states=...,          # [B, T_sem,  2048]
    thinker_attention_mask=...,         # [B, T_sem]
    semantic_route_mask=...,            # [B, T] in {0=text, 1=hidden}, optional
    # codec route:
    codec_input_ids=...,                # [B, T_codec]    REQUIRED
    codec_attention_mask=...,           # [B, T_codec]
    codec_labels=...,                   # [B, T_codec]   (main code group)
    codec_residual_labels=...,          # [B, T_codec, num_code_groups]
)

out.main_codec_logits           # [B, T_codec, codec_vocab_size]   = 3072
out.residual_codec_logits       # [B, T_codec, num_code_groups, code_predictor_vocab_size]
out.talker_hidden_states        # [B, T_sem + T_codec, hidden]
out.semantic_block              # [B, T_sem, hidden]
out.loss                        # scalar (main + residual CE; reduction='mean')
out.logits                      # codec-only mode -> main_codec_logits (for token-acc);
                                #   text-supervised mode -> thinker text logits
```

The **residual code predictor** runs an autoregressive roll-out along the
*codebook-group* dimension, mirroring upstream Qwen3-TTS
(`forward_sub_talker_finetune` + `code_predictor.forward_finetune`). For every
codec frame it builds a length `num_code_groups + 1` group sequence
`[talker_hidden, E_main(main_code), E_res[0](res_0), ...]`, runs a **causal**
transformer over the group dimension (teacher-forced with the ground-truth
lower groups), and lets `lm_head[i]` predict residual group `i`. It is therefore
only evaluated for the codec loss when **both** `codec_labels` and
`codec_residual_labels` are provided.

## Notes on the design

* The **bridge starts at random init** and is the principal trainable module
  in Stage 1. Loading from the exported directory does NOT skip it; you can
  train it directly without re-loading Qwen3-Omni or Qwen3-TTS weights.
* The talker uses `Qwen3Model` (dense, transformers ≥ 5.0) as its backbone.
  Layer slicing (`top_16` / `top_12` / `top_8` / `top_4`) is implemented at export time:
  the exported directory only contains the kept layers, so subsequent
  training has no overhead from the dropped layers.
* The **residual Code Predictor** is trained in Stage-1 bridge alignment
  (residual codebooks, group-dimension autoregressive teacher forcing),
  together with the bridge and the bottom talker layers. Only the Code2Wav
  decoder is **not** trained and is **not** part of this package — supply the
  upstream Qwen3-TTS-Tokenizer-12Hz decoder externally for
  evaluation/inference.
* `codec_pad_id` defaults to **2148** (the upstream Qwen3-TTS codec pad id).
  Do **not** use `0`, which is a valid codebook value and would collide with
  real main-code frames.
* **Control-token prefix.** The codec sequence is prefixed with the talker's
  pretraining "mode-switch" tokens (`think/nothink`, `think_bos/eos`, optional
  `language_id`, `codec_pad`, `codec_bos`), so a frozen / partially-unfrozen
  talker sees its familiar conditioning state. These positions are never
  supervised (label = -100). The ids and the `codec_language_id` map live on
  the model config (defaults mirror Qwen3-TTS-12Hz-1.7B-Base).
* **Streaming extension point.** `build_block_causal_mask` takes an optional
  `codec_to_sem_visible` (`[codec_len]` or `[B, codec_len]`): per codec frame,
  how many leading semantic tokens it may attend to. `None` (default) keeps the
  offline "codec sees all semantics" behaviour used by Stage-1; a monotonic
  schedule turns it into a chunk-causal (read-write) mask for simultaneous
  interpretation — without touching any other code path.
* Full Stage 0 / Stage 1 / Stage 2 / Stage 3 training scripts are **not
  included** here; this package focuses on the model definition and the
  ms-swift registration glue, in line with the original task scope (model
  implementation + 8 variants + registration + export script).

## Known caveats

* `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` is not natively supported by
  `transformers ≥ 5.x` (no `Qwen3TTS*` classes). The export script therefore
  loads the raw state-dict (safetensors / pytorch shards) and remaps keys to
  this package's `Qwen3TtsTalkerModel`. If your local copy uses a different
  state_dict prefix layout, extend `_TALKER_KEY_PREFIXES` in
  `export_qwen_omnispeak.py`.
* The DualRoute Q-Former bridge (variants F/G/H/I/J) initialises learnable query
  tokens at `~N(0, 0.02)`. Stage-1 training is recommended before any
  evaluation.
* `_no_split_modules` is set for HF auto-device-mapping; if you use
  `accelerate`'s `infer_auto_device_map` directly, you may need to extend
  this list with the sub-module names of your specific Qwen3-Omni
  checkpoint.
