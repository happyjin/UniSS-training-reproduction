# UniSS runtime prompt/KV parity v2

This isolated module fixes one specific runtime/training mismatch without
changing any historical demo.

For every 160/320/480/640 ms decision tick it commits the following sequence to
the **same persistent Qwen KV cache**:

```text
START_GLM + newly-visible causal GLM tokens + END_GLM + WAIT
```

or:

```text
START_GLM + newly-visible causal GLM tokens + END_GLM + WRITE
+ language + speed + START_CONTENT + text delta + END_CONTENT
+ START_SEMANTIC + BiCodec semantic delta + END_SEMANTIC
```

The action head observes the hidden state at `END_GLM`, exactly the position
that predicts the action token in dense trajectory training.  WAIT/WRITE, text,
semantic audio, and all boundaries are then appended to the main cache.  No
compressed target-history reconstruction and no cumulative source replay are
allowed.

`session.py` deliberately depends on a tiny `KVBackend` protocol.  A repaired
model runtime should implement its two append methods:

- `append_token_ids`: ordinary Qwen `input_ids` append;
- `append_source_codes`: causal frontend `forward_chunk`, projection, then Qwen
  `inputs_embeds` append using the incoming persistent cache.

Both methods must return the new `past_key_values`.  The state machine passes
that exact cache into the next operation.

This solves prompt/KV parity only.  Real PCM streaming additionally requires
runtime-exact causal WhisperVQ codes and commit timestamps in the training data.
