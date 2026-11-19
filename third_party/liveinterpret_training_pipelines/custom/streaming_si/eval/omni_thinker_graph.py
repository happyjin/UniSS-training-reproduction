"""CUDA Graph for the Repro-Omni *thinker* decode step under HF beam search (transformers 4.52.x).

Why: profiling shows thinker.generate is ~35% of a streaming decision and every token costs ~30 ms eager
(2000+ python-dispatched kernels), but the GPU only needs ~12-16 ms. The Talker already runs under a graph.

How (nothing about decoding math changes):
  * The agent prefills with a DynamicCache and then calls thinker.generate(past_key_values=<cache of batch=beams>).
    `generate` is wrapped: that DynamicCache is copied into a preallocated StaticBeamCache of the smallest
    capacity bucket that fits (prompt + max_new_tokens) and the static cache is handed to HF instead.
  * Every HF decode step calls thinker.forward with q_len == 1 (including the first step). `forward` is wrapped:
    for a StaticBeamCache it copies ids / cache_position / rope_deltas into static buffers and replays a
    captured graph of (beam-reorder gather -> text model -> lm_head). Same position rule as the original
    forward: position = cache_position + rope_deltas. Attention mask = (arange(capacity) <= cache_position).
  * HF beam search reorders the cache after every step via Cache.reorder_cache(beam_idx). Here that only
    records beam_idx; the gather runs inside the next replay (identity when nothing is pending).
  * Anything that does not fit (not a DynamicCache, beams == 1, batch != beams, too long, capture failed)
    falls through to the untouched original code path.

Env:
  OMNI_THINKER_GRAPH          graph | static | off   (static = same static-cache path without graph replay,
                              used to separate "static cache numerics" from "graph")
  OMNI_THINKER_GRAPH_BUCKETS  default 384,512,768,1024,2048
  OMNI_THINKER_GRAPH_STATS    path of a json file written at exit
"""
import atexit
import json
import os
import time
from collections import Counter
from pathlib import Path

import torch
from transformers.cache_utils import Cache, DynamicCache

DEFAULT_BUCKETS = '384,512,768,1024,2048'


class StaticBeamCache(Cache):
    """Fixed-address KV cache for one (batch, capacity). Not a StaticCache subclass on purpose: HF's
    prepare_inputs_for_generation would otherwise build a 4D mask for it on every step."""

    def __init__(self, text_cfg, batch, capacity, device, dtype):
        super().__init__()
        n = text_cfg.num_hidden_layers
        kv = text_cfg.num_key_value_heads
        hd = getattr(text_cfg, 'head_dim', None) or text_cfg.hidden_size // text_cfg.num_attention_heads
        shape = (batch, kv, capacity, hd)
        self.batch, self.capacity, self.n_layers = batch, capacity, n
        self.key_cache = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(n)]
        self.value_cache = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(n)]
        self.beam = torch.arange(batch, device=device)         # applied inside the next replay
        self.identity = torch.arange(batch, device=device)
        self.pending = False
        self.active = False
        self._len = 0                                           # python int: HF bookkeeping only
        self.graph = None

    # ---- Cache API used by HF / the model ----
    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        pos = cache_kwargs['cache_position']
        k, v = self.key_cache[layer_idx], self.value_cache[layer_idx]
        k.index_copy_(2, pos, key_states)
        v.index_copy_(2, pos, value_states)
        return k, v

    def get_seq_length(self, layer_idx=0):
        return self._len

    def get_max_cache_shape(self):
        return self.capacity

    def reorder_cache(self, beam_idx):
        self.beam.copy_(beam_idx)       # int32 -> int64 on copy
        self.pending = True

    # ---- loading ----
    def load_from(self, dyn):
        with torch.inference_mode():
            length = dyn.key_cache[0].shape[-2]
            for i in range(self.n_layers):
                self.key_cache[i][:, :, :length].copy_(dyn.key_cache[i])
                self.value_cache[i][:, :, :length].copy_(dyn.value_cache[i])
        self._len = length
        self.pending = False


class StepGraph:
    def __init__(self, thinker, cache):
        self.th, self.cache = thinker, cache
        dev = cache.key_cache[0].device
        self.dtype = cache.key_cache[0].dtype
        B, C = cache.batch, cache.capacity
        self.B, self.C = B, C
        self.ids = torch.zeros(B, 1, dtype=torch.long, device=dev)
        self.cpos = torch.zeros(1, dtype=torch.long, device=dev)
        self.rd = torch.zeros(1, 1, dtype=torch.long, device=dev)
        self.arange = torch.arange(C, device=dev)
        self.min = torch.finfo(self.dtype).min
        self.graph = None
        self.logits = None

    def body(self):
        c, th = self.cache, self.th
        for k, v in zip(c.key_cache, c.value_cache):            # pending beam reorder (identity if none)
            k.copy_(k.index_select(0, c.beam))
            v.copy_(v.index_select(0, c.beam))
        delta = self.cpos[0] + self.rd                          # (1,1): same rule as the original forward
        position_ids = delta.expand(self.B, 1).unsqueeze(0).expand(3, -1, -1)
        allowed = (self.arange <= self.cpos[0]).view(1, 1, 1, self.C)
        mask = torch.zeros(1, 1, 1, self.C, dtype=self.dtype, device=self.arange.device)
        mask = mask.masked_fill(~allowed, self.min).expand(self.B, 1, 1, self.C)
        out = th.model(input_ids=self.ids, attention_mask=mask, position_ids=position_ids,
                       past_key_values=c, cache_position=self.cpos, use_cache=True, return_dict=True)
        return th.lm_head(out.last_hidden_state)

    def capture(self):
        with torch.inference_mode():
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    self.body()
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                self.logits = self.body()
            self.graph = graph

    def run(self, ids, cpos, rd, mode):
        c = self.cache
        with torch.inference_mode():
            self.ids.copy_(ids)
            self.cpos.copy_(cpos)
            self.rd.copy_(rd)
            if not c.pending:
                c.beam.copy_(c.identity)
            c.pending = False
            if mode == 'graph':
                self.graph.replay()
                out = self.logits
            else:
                out = self.body()
            c._len += 1
            return out.clone()


class ThinkerGraphAccel:
    def __init__(self, model, mode, buckets):
        self.model, self.th, self.mode = model, model.thinker, mode
        self.buckets = tuple(sorted(buckets))
        self.entries = {}                                       # batch -> {capacity: cache}
        self.stats = Counter()
        self.capture_sec = {}
        self.orig_forward = self.th.forward
        self.orig_generate = self.th.generate
        self.th.forward = self._forward
        self.th.generate = self._generate

    def _caches_for(self, batch):
        if batch in self.entries:
            return self.entries[batch]
        text_cfg = self.th.model.config
        p = next(self.th.parameters())
        group = {}
        for cap in self.buckets:
            cache = StaticBeamCache(text_cfg, batch, cap, p.device, p.dtype)
            cache.graph = StepGraph(self.th, cache)
            t = time.perf_counter()
            if self.mode == 'graph':
                try:
                    cache.graph.capture()
                except Exception as exc:                        # keep serving on the original path
                    self.stats['capture_failed'] += 1
                    self.stats['capture_error'] = repr(exc)[:300]
                    cache = None
            if cache is not None:
                self.capture_sec[f'{batch}x{cap}'] = round(time.perf_counter() - t, 2)
                group[cap] = cache
        self.entries[batch] = group
        return group

    def _generate(self, *args, **kwargs):
        pkv = kwargs.get('past_key_values')
        beams = int(kwargs.get('num_beams', 1) or 1)
        if self.mode in ('graph', 'static'):
            if not (beams > 1 and isinstance(pkv, DynamicCache) and len(pkv.key_cache) > 0):
                self.stats['skip_not_applicable'] += 1
            else:
                batch, length = pkv.key_cache[0].shape[0], pkv.key_cache[0].shape[-2]
                new = int(kwargs.get('max_new_tokens') or 0)
                need = length + new + 1
                group = self._caches_for(batch) if batch == beams and new > 0 else {}
                cap = next((c for c in sorted(group) if c >= need), None)
                if cap is None:
                    self.stats['skip_too_long_or_unavailable'] += 1
                else:
                    cache = group[cap]
                    cache.load_from(pkv)
                    cache.active = True
                    kwargs['past_key_values'] = cache
                    self.stats['generate_accelerated'] += 1
                    self.stats[f'bucket_{cap}'] += 1
                    try:
                        return self.orig_generate(*args, **kwargs)
                    finally:
                        cache.active = False
        return self.orig_generate(*args, **kwargs)

    def _forward(self, *args, **kwargs):
        cache = kwargs.get('past_key_values')
        ids = kwargs.get('input_ids')
        if (isinstance(cache, StaticBeamCache) and cache.active and not args and ids is not None
                and ids.shape[1] == 1 and ids.shape[0] == cache.batch
                and kwargs.get('cache_position') is not None and kwargs.get('inputs_embeds') is None
                and self.th.rope_deltas is not None):
            from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
                Qwen2_5OmniThinkerCausalLMOutputWithPast)
            logits = cache.graph.run(ids, kwargs['cache_position'], self.th.rope_deltas, self.mode)
            self.stats['steps_accelerated'] += 1
            return Qwen2_5OmniThinkerCausalLMOutputWithPast(
                logits=logits, past_key_values=cache, rope_deltas=self.th.rope_deltas)
        self.stats['steps_original'] += 1
        return self.orig_forward(*args, **kwargs)

    def dump(self, path):
        data = dict(mode=self.mode, buckets=self.buckets, stats=dict(self.stats), capture_sec=self.capture_sec,
                    peak_alloc_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2))
        Path(path).write_text(json.dumps(data, indent=2))
        return data


def maybe_install(model):
    mode = os.environ.get('OMNI_THINKER_GRAPH', 'off')
    if mode not in ('graph', 'static'):
        return None
    if getattr(model, '_thinker_graph_accel', None) is not None:
        return model._thinker_graph_accel
    buckets = [int(x) for x in os.environ.get('OMNI_THINKER_GRAPH_BUCKETS', DEFAULT_BUCKETS).split(',')]
    accel = ThinkerGraphAccel(model, mode, buckets)
    model._thinker_graph_accel = accel
    stats_path = os.environ.get('OMNI_THINKER_GRAPH_STATS')
    if stats_path:
        atexit.register(lambda: accel.dump(stats_path))
    print(f'[thinker-graph] installed mode={mode} buckets={accel.buckets}', flush=True)
    return accel
