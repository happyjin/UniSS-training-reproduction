"""Opt-in, single-session Qwen2.5-Omni talker static cache / CUDA graph probe.

Copied into the staged official repository by patch_omni_repo.py. The owning
inference instance must not be used by concurrent sessions (same as its RoPE state).
"""
import os
from types import SimpleNamespace

import torch
from transformers import DynamicCache, StaticCache


class PositionedCache(StaticCache):
    """Avoid scanning KV values and synchronizing to discover the valid length."""
    def get_seq_length(self, layer_idx=0):
        return self.valid_length


class Slot:
    def __init__(self, talker, capacity):
        weight = next(talker.parameters())
        self.cache = PositionedCache(
            config=talker.model.config, max_batch_size=1, max_cache_len=capacity,
            device=weight.device, dtype=weight.dtype,
        )
        self.cache.valid_length = 0
        self.capacity = capacity
        self.graph = None
        self.disabled = False


class TalkerAccelerator:
    def __init__(self, talker):
        self.talker = talker
        self.original = talker.forward
        self.buckets = tuple(sorted(set(int(x) for x in os.environ.get(
            'SIMULS2ST_TALKER_CACHE_BUCKETS', '1024,2048,4096').split(','))))
        if not self.buckets or self.buckets[0] <= 0:
            raise ValueError('Cache capacities must be positive')
        self.slots = {}
        self.fallbacks = 0
        self.replays = 0
        self.validations = 0
        self.validation_slots = {}

    @staticmethod
    def dynamic(cache):
        out = DynamicCache()
        length = int(cache.get_seq_length())
        for i, (k, v) in enumerate(zip(cache.key_cache, cache.value_cache)):
            out.update(k[:, :, :length].clone(), v[:, :, :length].clone(), i)
        out._omni_accel_fallback = True
        return out

    def adopt(self, cache, required):
        if isinstance(cache, PositionedCache) and required <= cache.max_cache_len:
            return cache
        capacity = next((b for b in self.buckets if b >= required), None)
        if capacity is None:
            self.fallbacks += 1
            print(f'[talker-accel] overflow required={required}; dynamic eager fallback', flush=True)
            return self.dynamic(cache) if isinstance(cache, PositionedCache) else cache
        slot = self.slots.get(capacity)
        if slot is None:
            slot = self.slots[capacity] = Slot(self.talker, capacity)
        length = int(cache.get_seq_length())
        slot.cache.reset()
        for dst_k, dst_v, src_k, src_v in zip(
            slot.cache.key_cache, slot.cache.value_cache, cache.key_cache, cache.value_cache
        ):
            dst_k[:, :, :length].copy_(src_k[:, :, :length])
            dst_v[:, :, :length].copy_(src_v[:, :, :length])
        slot.cache.valid_length = length
        print(f'[talker-accel] adopt capacity={capacity} valid={length}', flush=True)
        return slot.cache

    def direct_step(self, slot, embeds, position_ids, cache_position, mask):
        # Bypass the outer talker.forward's Python RoPE branch. All positions are
        # supplied explicitly; the projection, decoder and codec head stay intact.
        out = self.talker.model(
            inputs_embeds=self.talker.thinker_to_talker_proj(embeds),
            attention_mask=mask, position_ids=position_ids,
            cache_position=cache_position, past_key_values=slot.cache,
            use_cache=True, return_dict=True,
        )
        return self.talker.codec_head(out.last_hidden_state).float()

    def step(self, slot, embeds, positions, cpos, mode):
        if mode == 'static' or slot.disabled:
            mask = torch.where(
                torch.arange(slot.capacity, device=embeds.device) <= cpos[0],
                0., torch.finfo(embeds.dtype).min,
            ).to(embeds.dtype).view(1, 1, 1, -1)
            return self.direct_step(slot, embeds, positions, cpos, mask)
        if slot.graph is None:
            slot.x = torch.empty_like(embeds)
            slot.positions = torch.empty_like(positions)
            slot.cpos = torch.empty_like(cpos)
            slot.mask = torch.empty((1, 1, 1, slot.capacity), device=embeds.device, dtype=embeds.dtype)
        slot.x.copy_(embeds)
        slot.positions.copy_(positions)
        slot.cpos.copy_(cpos)
        slot.mask.copy_(torch.where(
            torch.arange(slot.capacity, device=embeds.device) <= cpos[0],
            0., torch.finfo(embeds.dtype).min,
        ).to(embeds.dtype).view(1, 1, 1, -1))
        if slot.graph is None:
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    self.direct_step(slot, slot.x, slot.positions, slot.cpos, slot.mask)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            try:
                with torch.cuda.graph(graph):
                    slot.logits = self.direct_step(slot, slot.x, slot.positions, slot.cpos, slot.mask)
            except RuntimeError as exc:
                # Only recover recognized graph-capture incompatibility. Model,
                # indexing and CUDA errors must propagate instead of hiding bugs.
                if 'captur' not in str(exc).lower():
                    raise
                torch.cuda.synchronize()
                slot.disabled = True
                print(f'[talker-accel] graph capture unsupported; static eager: {exc}', flush=True)
                return self.step(slot, embeds, positions, cpos, 'static')
            slot.graph = graph
            print(f'[talker-accel] graph captured capacity={slot.capacity}', flush=True)
        slot.graph.replay()
        self.replays += 1
        # Caller modifies scores in-place. Never expose the graph's output buffer.
        return slot.logits.clone()

    def _forward(self, **kw):
        mode = os.environ.get('SIMULS2ST_TALKER_ACCEL', 'eager')
        embeds = kw['inputs_embeds']
        cache = kw['past_key_values']
        if (mode not in {'static', 'graph'} or embeds.device.type != 'cuda'
                or embeds.shape[0] != 1 or cache is None
                or getattr(cache, '_omni_accel_fallback', False)):
            return self.eager(**kw)
        length = int(cache.get_seq_length())
        seq_len = embeds.shape[1]
        cache = self.adopt(cache, length + seq_len)
        kw['past_key_values'] = cache
        if not isinstance(cache, PositionedCache):
            cache._omni_accel_fallback = True
            return self.eager(**kw)
        cpos = torch.arange(length, length + seq_len, device=embeds.device)
        if seq_len != 1 or kw['position_ids'] is not None or self.talker.rope_deltas is None:
            out = self.eager(**kw)
            cache.valid_length = length + seq_len
            return out
        positions = (cpos.view(1, -1) + self.talker.rope_deltas).unsqueeze(0).expand(3, -1, -1).contiguous()
        logits = self.step(self.slots[cache.max_cache_len], embeds, positions, cpos, mode)
        cache.valid_length = length + seq_len
        return SimpleNamespace(logits=logits, past_key_values=cache)

    def forward(self, **kw):
        # Diagnostic only: run dynamic-cache eager on the identical incoming KV
        # and embeddings, without changing the generated trajectory. Never use
        # these shadow runs as performance measurements.
        limit = int(os.environ.get('SIMULS2ST_TALKER_VALIDATE_STEPS', '0'))
        reference = None
        static_reference = None
        if self.validations < limit and kw['inputs_embeds'].shape[1] == 1:
            shadow = dict(kw)
            shadow['past_key_values'] = self.dynamic(kw['past_key_values'])
            shadow['inputs_embeds'] = kw['inputs_embeds'].clone()
            reference = self.eager(**shadow).logits.detach()
            length = int(kw['past_key_values'].get_seq_length())
            capacity = next((b for b in self.buckets if b >= length + 1), None)
            if capacity is not None and os.environ.get('SIMULS2ST_TALKER_VALIDATE_STATIC') == '1':
                slot = self.validation_slots.get(capacity)
                if slot is None:
                    slot = self.validation_slots[capacity] = Slot(self.talker, capacity)
                slot.cache.reset()
                for dk, dv, sk, sv in zip(slot.cache.key_cache, slot.cache.value_cache,
                                        kw['past_key_values'].key_cache, kw['past_key_values'].value_cache):
                    dk[:, :, :length].copy_(sk[:, :, :length])
                    dv[:, :, :length].copy_(sv[:, :, :length])
                slot.cache.valid_length = length
                static_shadow = dict(kw, past_key_values=slot.cache,
                                     inputs_embeds=kw['inputs_embeds'].clone())
                static_reference = self.eager(**static_shadow).logits.detach()
        result = self._forward(**kw)
        if reference is not None:
            self.validations += 1
            difference = (reference - result.logits).abs()
            print(f'[talker-shadow] n={self.validations} '
                  f'max_abs={difference.max().item():.6f} '
                  f'mean_abs={difference.mean().item():.6f} '
                  f'top1_equal={bool(torch.equal(reference.argmax(-1), result.logits.argmax(-1)))}', flush=True)
            if static_reference is not None:
                delta = (static_reference - result.logits).abs()
                print(f'[talker-static-control] n={self.validations} '
                      f'max_abs={delta.max().item():.6f} mean_abs={delta.mean().item():.6f} '
                      f'exact={torch.equal(static_reference, result.logits)} '
                      f'top1_equal={torch.equal(static_reference.argmax(-1), result.logits.argmax(-1))}', flush=True)
        return result

    def eager(self, **kw):
        length = int(kw['past_key_values'].get_seq_length()) if kw['past_key_values'] is not None else 0
        n = kw['inputs_embeds'].shape[1]
        kw['cache_position'] = torch.arange(length, length + n, device=kw['inputs_embeds'].device)
        kw['attention_mask'] = torch.ones((1, length + n), device=kw['inputs_embeds'].device, dtype=torch.long)
        return self.original(**kw)


def forward(talker, **kwargs):
    accelerator = getattr(talker, '_static_graph_accelerator', None)
    if accelerator is None:
        accelerator = talker._static_graph_accelerator = TalkerAccelerator(talker)
    return accelerator.forward(**kwargs)
