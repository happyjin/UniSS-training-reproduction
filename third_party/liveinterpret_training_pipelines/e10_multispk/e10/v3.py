"""E10 v3: LiveInterpret S5 trained with baseline B's data, hyper-parameters, trainable parts and speaker stream,
in LiveInterpret's own chunk-as-turn streaming format (reports/mist_w3po_v1/e10v3_prereg.zh-CN.md).

What changes against E10 v1/v2 (e10.patch / e10.tuner, left untouched):
  * base: the S5 adapter (checkpoint-29773) is merged into the S4 weights first (B: LoRA on a merged base); stage 2
    additionally merges the stage-1 (O2) adapter and takes over its speaker projector, like B's INIT_MERGED=1;
  * trainable: a fresh LoRA r32/alpha 64 on the Thinker's attention projections only (full-attention q/k/v/o and
    the Gated-DeltaNet in_proj_qkv/in_proj_z/out_proj; B: q/k/v/o, no MLP), the eight speaker-tag rows, and B's
    SDProjector (Conv1d stride 5 + MLP, last layer zero-initialised). Audio encoder and Talker frozen. All at one lr;
  * speaker stream as B: one soft speaker token per 5 audio frames, a separate sequence position holding
    SDProjector(post[5j:5j+5]). LiveInterpret's KV cache cannot revisit old positions, so block j's token is placed
    right after the audio frame at which its posterior exists under B's lowlat rule (frame 5j+5+13 for streaming
    records); offline records (full posterior, as B's offline rows) place it right after the block itself;
    single-speaker sd records have no speaker stream (B: has_sd False);
  * decision loss only on streaming records: offline / sd records keep their text loss but their wait/W labels are
    masked (B's offline rows never train an empty output on a prefix; v1/v2 expanded them into wait trajectories).

The placeholder position uses the unused `<|pause:1|>` row; its embedding is always overwritten.
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .common import N_SLOT, perturb

K = 5                       # audio frames per speaker token (B: K = 5 audio tokens, 80 ms each)
STREAM_DELAY = 13           # B's lowlat rule: posterior of frame f exists 1040 ms = 13 frames later
SLOT_TOKEN = "<|pause:1|>"
V3_METADATA = "e10v3_metadata.json"
V3_FORMAT = "e10v3_peft_v1"
SPK_PROJ_FILE = "e10_spk_proj.pt"
LORA_TARGET = (r"^thinker_model\.model\.layers\.\d+\."
               r"(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|linear_attn\.(?:in_proj_qkv|in_proj_z|out_proj))$")


# ------------------------------------------------------------------------------------------- speaker stream layout
def block_due(j: int, delay: int) -> int:
    """Number of audio frames that must have been fed before block j's speaker token can be placed."""
    return K * (j + 1) + delay


def slots_in_run(g0: int, n: int, delay: int, n_total: int | None = None, offline: bool = False):
    """Audio frames g0+1 .. g0+n are fed (g = frames fed so far). Returns [(after, j)]: block j's token goes after
    the `after`-th frame of this run (1..n). Offline: also the partial last block once all n_total frames are in."""
    out = []
    j = max(0, (g0 - delay) // K - 1)
    while True:
        a = block_due(j, delay)
        if a > g0 + n:
            break
        if a > g0:
            out.append((a - g0, j))
        j += 1
    if offline and n_total is not None and n_total % K and g0 < n_total <= g0 + n:
        out.append((n_total - g0, n_total // K))
    return out


def n_slots(n_total: int, delay: int, offline: bool) -> int:
    return len(slots_in_run(0, n_total, delay, n_total, offline))


class SDProjector(nn.Module):
    """baseline B's speaker-stream projector (experiments/uniss_qwen3_omni_v1/training/thinker_gstar_model.py)."""

    def __init__(self, c_in: int = N_SLOT, d_llm: int = 2560, hidden: int = 1024, stride: int = K):
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv1d(c_in, hidden, kernel_size=stride, stride=stride)
        self.mlp = nn.Sequential(nn.GELU(), nn.Linear(hidden, d_llm), nn.GELU(), nn.Linear(d_llm, d_llm))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # [B, L, C] -> [B, ceil(L/stride), d]
        pad = (-x.shape[1]) % self.stride
        if pad:
            x = F.pad(x, (0, 0, 0, pad))
        return self.mlp(self.conv(x.transpose(1, 2)).transpose(1, 2))


def spk_vectors(model, post: torch.Tensor, n: int) -> torch.Tensor:
    """post (frames, 8) -> the first n block vectors (n, hidden) in the projector's dtype."""
    proj = model.e10_spk_proj
    p = post.to(device=next(proj.parameters()).device, dtype=next(proj.parameters()).dtype)
    need = n * K
    if p.size(0) < need:
        p = torch.cat([p, p.new_zeros(need - p.size(0), p.size(1))])
    return proj(p[None])[0][:n]


# --------------------------------------------------------------------------------------------- data-side patches
def _rec(ds, flat_index):
    fi, li = ds.items[flat_index]
    return ds.indexes[fi].read(li)


def build_full_view(ds, prepared, v3, slot_id):
    """S4StreamingDataset._build_view(kind='full') with the speaker-token positions and the decision mask."""
    offline = v3["layout"] == "offline"
    has_spk = v3["layout"] != "none"
    delay = 0 if offline else STREAM_DELAY
    mask_dec = bool(v3["mask_decision"])
    n_total = int(sum(prepared.audio_seg_lens))
    ids = list(prepared.instr_ids)
    labels = [-100] * len(ids)
    answer = [0] * len(ids)
    src_end_done, w, g, n_spk = False, 0, 0, 0
    for i, s in enumerate(prepared.steps):
        ids.append(ds.chunk_token_id)
        labels.append(-100 if i == 0 else ds.chunk_token_id)
        answer.append(0)
        n = prepared.audio_seg_lens[i]
        if n:
            cuts = slots_in_run(g, n, delay, n_total, offline) if has_spk else []
            prev = 0
            for after, j in cuts:
                if j != n_spk:
                    raise RuntimeError(f"speaker block order {j} != {n_spk}")
                ids += [ds.audio_token_id] * (after - prev)
                ids.append(slot_id)
                prev = after
                n_spk += 1
            ids += [ds.audio_token_id] * (n - prev)
            labels += [-100] * (n + len(cuts))
            answer += [0] * (n + len(cuts))
            g += n
        if ds.use_src_end and not src_end_done and s.get("source_stream_done"):
            ids.append(ds.src_end_token_id)
            labels.append(-100)
            answer.append(0)
            src_end_done = True
        is_write = s["action"] == "write"
        dec = ds.write_token_id if is_write else ds.wait_token_id
        ids.append(dec)
        labels.append(-100 if mask_dec else dec)
        answer.append(0)
        if is_write:
            tid = prepared.text_ids[w]
            w += 1
            ids += tid
            labels += tid
            answer += [1] * len(tid)
    ids.append(ds.eos_token_id)
    labels.append(ds.eos_token_id)
    answer.append(0)
    if len(ids) > ds.max_length:
        return ds._skip("too_long")
    return {
        "record_id": prepared.record_id,
        "input_ids": torch.tensor(ids[:-1], dtype=torch.long),
        "labels": torch.tensor(labels[1:], dtype=torch.long),
        "answer_mask": torch.tensor(answer[:-1], dtype=torch.long),
        "mel": prepared.mel,
        "mel_len": prepared.mel.size(-1),
        "audio_seg_lens": torch.tensor(prepared.audio_seg_lens, dtype=torch.long),
        "speech_tokens": torch.tensor(prepared.speech_tokens, dtype=torch.long),
        "speech_len": len(prepared.speech_tokens),
        "text_seg_lens": torch.tensor([len(t) for t in prepared.text_ids], dtype=torch.long),
        "speech_seg_lens": torch.tensor(prepared.speech_seg_lens, dtype=torch.long),
        "e10v3_post": v3.get("post") or "" if has_spk else "",
        "e10v3_n_spk": n_spk,
    }


def _patch_dataset():
    from streaming_si.data import s4_streaming_dataset as M
    cls = M.S4StreamingDataset
    orig_prepare, orig_view = cls._prepare_record, cls._build_view

    def _prepare_record(self, flat_index, forced_m=None):
        prepared = orig_prepare(self, flat_index, forced_m)
        if prepared is None:
            return None
        e10 = _rec(self, flat_index).get("e10") or {}
        text = e10.get("instruction")
        if text:
            key = ("e10", text)
            if key not in self._instr_cache:
                self._instr_cache[key] = self.tokenizer.encode(text, add_special_tokens=False)
            prepared = __import__("dataclasses").replace(prepared, instr_ids=list(self._instr_cache[key]))
        return prepared

    def _build_view(self, prepared, kind, live_turn=None, include_view_fields=False):
        e10 = _rec(self, prepared.flat_index).get("e10") or {}
        v3 = e10.get("v3")
        if v3 is None:
            raise RuntimeError(f"{prepared.record_id}: not an E10 v3 record")
        if kind != "full":
            raise RuntimeError("E10 v3 records support only full views")
        if not hasattr(self, "_e10v3_slot"):
            got = self.tokenizer.encode(SLOT_TOKEN, add_special_tokens=False)
            if len(got) != 1:
                raise RuntimeError(f"{SLOT_TOKEN} is not one token: {got}")
            self._e10v3_slot = int(got[0])
        return build_full_view(self, prepared, v3, self._e10v3_slot)

    cls._prepare_record, cls._build_view = _prepare_record, _build_view


def _patch_pipeline():
    from streaming_si.data import s4_pipeline as P
    orig_encode, orig_collate = P.S4Encoder.encode, P.collate_s4
    p_perturb = float(os.environ.get("E10_POST_PERTURB", "0.2"))

    def encode(self, row):
        item = orig_encode(self, row)
        path = item.pop("e10v3_post", None)
        n_spk = int(item.pop("e10v3_n_spk", 0))
        n = int(torch.as_tensor(item["audio_seg_lens"]).sum())
        post = None
        if n_spk:
            if not path:
                raise RuntimeError(f"{item.get('record_id')}: speaker slots without a posterior")
            post = np.load(path).astype(np.float32)[:, :N_SLOT]
            if row["s4_split"] == "train" and p_perturb > 0:
                post = perturb(post, np.random.default_rng(int(row["s4_seed"]) & 0xFFFFFFFF), p_perturb)
            full = np.zeros((max(n, n_spk * K), N_SLOT), dtype=np.float32)
            m = min(len(post), len(full))
            full[:m, : post.shape[1]] = post[:m]
            post = torch.from_numpy(full)
        item["spk_post"] = post
        item["spk_n"] = n_spk
        return item

    def collate_s4(batch):
        posts = [b.pop("spk_post", None) for b in batch]
        ns = [int(b.pop("spk_n", 0)) for b in batch]
        out = orig_collate(batch)
        out["spk_posts"] = posts
        out["spk_ns"] = ns
        return out

    P.S4Encoder.encode, P.collate_s4 = encode, collate_s4


# -------------------------------------------------------------------------------------------- model-side patches
def fill_slots(model, tokens, h, posts, ns, slot_id):
    """Write SDProjector block vectors into the slot positions of h (out of place)."""
    out = None
    for b in range(h.size(0)):
        idx = (tokens[b] == slot_id).nonzero(as_tuple=True)[0]
        n = int(ns[b]) if ns is not None else 0
        if idx.numel() != n:
            raise RuntimeError(f"E10 v3 sample {b}: {idx.numel()} speaker slots in the sequence, expected {n}")
        if n == 0:
            continue
        v = spk_vectors(model, posts[b], n)
        if out is None:
            out = h.clone()
        out[b, idx] = v.to(out.dtype)
    return h if out is None else out


def _patch_model():
    from streaming_si.model import model_si as S
    from streaming_si.model import modeling_streaming_si as H
    orig_forward = H.StreamingSIForConditionalGeneration.forward
    orig_inject = S.SI.inject_audio_features

    def forward(self, *args, spk_posts=None, spk_ns=None, **kwargs):
        if spk_ns is None:
            return orig_forward(self, *args, **kwargs)
        self.__dict__["_e10v3"] = (spk_posts, spk_ns)
        try:
            return orig_forward(self, *args, **kwargs)
        finally:
            self.__dict__.pop("_e10v3", None)

    def inject_audio_features(self, tokens, h, audio_feats, seqlen=None, audio_seg_lens=None,
                              audio_feature_offsets=None):
        st = self.__dict__.get("_e10v3")
        if st is None:
            return orig_inject(self, tokens, h, audio_feats, seqlen=seqlen, audio_seg_lens=audio_seg_lens,
                               audio_feature_offsets=audio_feature_offsets)
        if audio_feature_offsets is not None:
            raise RuntimeError("E10 v3: crop views are not supported")
        marker = self.config.audio_ids[0]
        for b in range(h.size(0)):      # strict replacement of the per-segment check: every frame lands once
            n_pad = int((tokens[b] == marker).sum())
            n_feat = 0 if audio_feats[b] is None else int(audio_feats[b].size(0))
            exp = int(sum(int(x) for x in audio_seg_lens[b])) if audio_seg_lens is not None else n_feat
            if not n_pad == n_feat == exp:
                raise RuntimeError(f"E10 v3 sample {b}: audio_pad {n_pad}, features {n_feat}, expected {exp}")
        h = orig_inject(self, tokens, h, audio_feats, seqlen=seqlen, audio_seg_lens=None)
        return fill_slots(self, tokens, h, st[0], st[1], int(self.config.e10v3_slot_id))

    H.StreamingSIForConditionalGeneration.forward = forward
    S.SI.inject_audio_features = inject_audio_features


_INSTALLED = False


def install():
    global _INSTALLED
    if _INSTALLED:
        return
    from . import patch as v1
    v1._patch_jsonl_index()
    _patch_dataset()
    _patch_pipeline()
    _patch_model()
    _INSTALLED = True


# ------------------------------------------------------------------------------------------------------ PEFT / IO
def _s4():
    from streaming_si.integration import s4_peft
    return s4_peft


def _log(msg):
    try:
        from swift.utils import get_logger
        get_logger().info(msg)
    except Exception:  # noqa: BLE001
        print(msg, flush=True)


def spk_ids(raw):
    from .tuner import spk_ids_from_stream
    return spk_ids_from_stream(_s4()._check_token_ids(raw.config.s4_stream_token_ids))


def slot_id(tok):
    got = tok.encode(SLOT_TOKEN, add_special_tokens=False)
    if len(got) != 1:
        raise RuntimeError(f"{SLOT_TOKEN} is not one token")
    return int(got[0])


def add_spk_proj(raw):
    if getattr(raw, "e10_spk_proj", None) is None:
        w = raw.thinker.embed_tokens.weight
        raw.e10_spk_proj = SDProjector(N_SLOT, w.size(1)).to(device=w.device, dtype=w.dtype)
    return raw.e10_spk_proj


def merge_adapter(raw, adapter_dir):
    """raw <- raw + adapter (S5 adapter or an E10 v3 stage adapter), merged; returns the merged model."""
    from peft import PeftModel
    pm = PeftModel.from_pretrained(raw, adapter_dir, is_trainable=False)
    return pm.merge_and_unload()


def _refresh(raw):
    """SI keeps aliases (lm_head, embed_tokens) to the thinker's modules; re-point them after PEFT surgery."""
    s4 = _s4()
    try:
        return s4._refresh_thinker_aliases(raw)
    except Exception:  # noqa: BLE001
        return raw


def build_base(model, merges, spk_init=None, slot=None):
    """S4 model -> merged base for v3: S5 adapter, then (stage 2) the stage-1 adapter; speaker projector from
    spk_init (a stage dir holding e10_spk_proj.pt) or fresh (zero last layer)."""
    s4 = _s4()
    raw = s4.unwrap_s4_model(model)
    for d in merges:
        raw = _refresh(merge_adapter(raw, d))
    if slot is not None:
        raw.config.e10v3_slot_id = int(slot)
    proj = add_spk_proj(raw)
    if spk_init:
        proj.load_state_dict(torch.load(os.path.join(spk_init, SPK_PROJ_FILE), map_location="cpu"))
    return raw


def attach_lora(raw, rank=32, alpha=64, dropout=0.0):
    from peft import LoraConfig, get_peft_model
    s4 = _s4()
    sids = spk_ids(raw)
    cfg = LoraConfig(r=int(rank), lora_alpha=int(alpha), lora_dropout=float(dropout), bias="none",
                     target_modules=LORA_TARGET, trainable_token_indices=sids)
    pm = get_peft_model(raw, cfg)
    pm = _refresh(pm) if not hasattr(s4, "_refresh_thinker_aliases") else s4._refresh_thinker_aliases(pm)
    inner = s4.unwrap_s4_model(pm)
    ea = inner.thinker_model.model.embed_tokens.token_adapter
    ha = getattr(inner.thinker_model.lm_head, "token_adapter", None)
    if ha is None or ha.tied_adapter is not ea:
        raise RuntimeError("E10 v3 speaker rows are not tied to lm_head")
    if inner.lm_head is not inner.thinker_model.lm_head:
        raise RuntimeError("SI.lm_head alias not refreshed after PEFT")
    return pm


def set_trainable(pm):
    """Exactly: thinker LoRA A/B, the 8 speaker rows, e10_spk_proj. Everything else frozen."""
    s4 = _s4()
    raw = s4.unwrap_s4_model(pm)
    names = []
    for n, p in raw.named_parameters():
        on = ("lora_A" in n or "lora_B" in n) and n.startswith("thinker_model.") or \
             "trainable_tokens_delta" in n or n.startswith("e10_spk_proj.")
        p.requires_grad_(bool(on))
        if on:
            names.append(n)
    bad = [n for n in names if ("lora_" in n and not n.startswith("thinker_model.model.layers."))]
    if bad:
        raise RuntimeError(f"unexpected LoRA tensors: {bad[:5]}")
    n_lora = sum(1 for n in names if "lora_A" in n)
    if n_lora == 0 or not any(n.startswith("e10_spk_proj.") for n in names) or \
            not any("trainable_tokens_delta" in n for n in names):
        raise RuntimeError(f"E10 v3 trainable set incomplete: {len(names)} tensors")
    n = sum(p.numel() for p in raw.parameters() if p.requires_grad)
    _log(f"[E10 v3] trainable tensors={len(names)} LoRA modules={n_lora} params={n / 1e6:.2f}M")
    return names


def save_v3(model, path, state_dict=None, safe_serialization=True):
    from peft import PeftModel
    s4 = _s4()
    pm = getattr(model, "module", model)
    if not isinstance(pm, PeftModel):
        raise TypeError("save_v3 needs a PeftModel")
    raw = s4.unwrap_s4_model(pm)
    os.makedirs(path, exist_ok=True)
    if state_dict is not None:
        state_dict = {k: v for k, v in state_dict.items() if "e10_spk_proj" not in k}
    pm.save_pretrained(path, state_dict=state_dict, safe_serialization=safe_serialization,
                       save_embedding_layers=False, selected_adapters=["default"])
    torch.save({k: v.detach().float().cpu() for k, v in raw.e10_spk_proj.state_dict().items()},
               os.path.join(path, SPK_PROJ_FILE))
    with open(os.path.join(path, V3_METADATA), "w", encoding="utf-8") as f:
        json.dump({"format": V3_FORMAT, "stream_token_ids": s4._check_token_ids(raw.config.s4_stream_token_ids),
                   "spk_token_ids": spk_ids(raw), "slot_token": SLOT_TOKEN, "k": K, "stream_delay": STREAM_DELAY,
                   "merges": os.environ.get("E10V3_MERGES", ""), "stage": os.environ.get("E10V3_STAGE", "")},
                  f, indent=2)


def load_v3_chain(model, chain, slot):
    """Inference: S4 model + chain of adapter dirs (S5, [stage 1], stage 2): every LoRA merged, the last dir's
    speaker projector loaded."""
    s4 = _s4()
    raw = s4.unwrap_s4_model(model)
    for d in chain:
        raw = _refresh(merge_adapter(raw, d))
    raw.config.e10v3_slot_id = int(slot)
    proj = add_spk_proj(raw)
    proj.load_state_dict(torch.load(os.path.join(chain[-1], SPK_PROJ_FILE), map_location="cpu"))
    return raw


try:
    from swift.tuner_plugin import Tuner as _SwiftTuner
except Exception:  # noqa: BLE001
    _SwiftTuner = object


class E10v3Tuner(_SwiftTuner):
    """--tuner_type e10v3. Env: E10V3_MERGES (comma list: S5 adapter[,stage-1 dir]), E10V3_SPK_INIT (stage-1 dir
    or empty), LORA r/alpha from --lora_rank/--lora_alpha."""

    @staticmethod
    def prepare_model(args, model):
        merges = [d for d in os.environ.get("E10V3_MERGES", "").split(",") if d]
        if not merges:
            raise RuntimeError("E10V3_MERGES is required (at least the S5 adapter)")
        raw = build_base(model, merges, os.environ.get("E10V3_SPK_INIT") or None,
                         slot=_TOK_SLOT if _TOK_SLOT is not None else os.environ.get("E10V3_SLOT_ID"))
        if getattr(raw.config, "e10v3_slot_id", None) is None:
            raise RuntimeError("E10 v3: slot token id unknown (set E10V3_SLOT_ID)")
        if not os.environ.get("E10V3_SPK_INIT"):
            with torch.no_grad():
                info = raw.reinit_stream_token_rows(spk_ids(raw), seed=int(args.seed))
            _log(f"[E10 v3] fresh speaker rows {spk_ids(raw)}: norm {info['old_norm_mean']:.3f} -> "
                 f"{info['target_norm']:.3f}")
        pm = attach_lora(raw, args.lora_rank, args.lora_alpha, args.lora_dropout)
        set_trainable(pm)
        _log(f"[E10 v3] merged {merges}; speaker projector from {os.environ.get('E10V3_SPK_INIT') or 'fresh'}")
        return pm

    @staticmethod
    def save_pretrained(model, save_directory, state_dict=None, safe_serialization=True, **kwargs):
        save_v3(model, save_directory, state_dict=state_dict, safe_serialization=safe_serialization)

    @staticmethod
    def from_pretrained(model, model_id, **kwargs):
        raise RuntimeError("E10 v3 does not resume (adapter-only saves); restart the stage")


_TOK_SLOT = None


def register(tok_slot_id=None):
    global _TOK_SLOT
    _TOK_SLOT = tok_slot_id
    from swift.tuner_plugin import tuners_map
    tuners_map["e10v3"] = E10v3Tuner
