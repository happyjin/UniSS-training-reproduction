"""TurnCRF end-to-end (scheme 2), inference: the decode engine with the TurnCRF layer between the diarizer posteriors
and the speaker projector. Default-off and additive: no existing file is changed.

  TurnEngine(Engine)  core/engine.Engine with audio_embeds re-implemented: identical (it calls Engine.audio_embeds)
                      while no layer is set; with a layer, at every read the visible prefix P[:tau]
                      (tau = max(0, t - 1040 ms) // 80, the frames the engine does not zero; the whole window for
                      oracle "full" reads) and, when the layer has an e-MLP, the item's audio-encoder states aligned to
                      the 80 ms frames go through TurnLayer; the result replaces those frames, everything after is the
                      engine's code unchanged (zeroing, post_to_tokens, projector, interleave). Items with the same
                      tau run as one batch.
  main()              python -m experiments.mist_w3po_v1.crf.turn_decode [online_decode args] [--turn-crf PATH]
                      [--turn-device cuda|cpu]: without --turn-crf it IS evaluation/online_decode.main with the same
                      argv; with it, online_decode runs with its Engine swapped for a TurnEngine holding the layer
                      loaded from PATH (a turn_layer.pt file, TurnLayer.export, or a checkpoint dir holding one).

With the layer's gate g = 0 the projector input is bit-identical to v8's (test_turn_layer.py, smoke (i)).
"""
from __future__ import annotations

import sys
from collections import defaultdict

import numpy as np
import torch

from experiments.mist_w3po_v1.core import engine as E
from experiments.mist_w3po_v1.crf.turn_layer import TurnLayer, e_to_frames, load_layer


class TurnEngine(E.Engine):
    turn: TurnLayer | None = None

    def set_turn(self, layer: TurnLayer | None, device=None) -> None:
        if layer is not None:
            layer = layer.to(device if device is not None else self.dev).eval()
            layer.requires_grad_(False)
        self.turn = layer

    def _turn_posts(self, entries: list[dict]) -> None:
        """entries: {"post", "tau", "ai", "dur_s"} -> entry["post"] replaced (first tau frames) by the layer output."""
        layer = self.turn
        dev = layer.g.device
        use_e = layer.e_mlp is not None
        groups = defaultdict(list)
        for en in entries:
            if en["tau"] > 0:
                groups[en["tau"]].append(en)
        with torch.no_grad():
            for tau, ens in groups.items():
                P = torch.from_numpy(np.stack([en["post"][:tau] for en in ens])).to(dev)
                e = None
                if use_e:
                    e = torch.stack([e_to_frames(en["ai"], tau, en["dur_s"]).to(dev) for en in ens])
                out = layer(P, e).cpu().numpy()
                for en, o in zip(ens, out):
                    p = en["post"].copy()
                    p[:tau] = o
                    en["post"] = p

    @torch.no_grad()
    def audio_embeds(self, items, emb_dtype, lookahead_ms: int = E.LOOKAHEAD_MS, cache: dict | None = None):
        if self.turn is None:
            return super().audio_embeds(items, emb_dtype, lookahead_ms=lookahead_ms, cache=cache)
        # engine.Engine.audio_embeds with the TurnCRF step inserted before post_to_tokens (keep in sync)
        from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import _get_feat_extract_output_lengths

        keys, uniq = [], {}
        for j, it in enumerate(items):
            k = it.akey or f"__{j}"
            keys.append(k)
            if cache is None or k not in cache:
                uniq.setdefault(k, j)
        order = list(uniq)
        if not order:
            return [cache[k] for k in keys]
        wavs = []
        for k in order:
            it = items[uniq[k]]
            x = it.wav[: it.t_ms * 16]
            wavs.append(x if len(x) >= 1600 else np.pad(x, (0, 1600 - len(x))))
        f = self.pr.proc.feature_extractor(wavs, sampling_rate=16000, padding=True, truncation=False,
                                           return_attention_mask=True, return_tensors="pt")
        n_audio = _get_feat_extract_output_lengths(f["attention_mask"].sum(1)).tolist()
        a = self.thinker.get_audio_features(f["input_features"].to(self.dev, torch.bfloat16),
                                            f["attention_mask"].to(self.dev)).last_hidden_state.to(emb_dtype)
        entries, pos = [], 0
        for k, n, x in zip(order, n_audio, wavs):
            it = items[uniq[k]]
            ai = a[pos: pos + n]
            pos += n
            post = it.post.copy()
            if not it.full:
                tau = max(0, it.t_ms - lookahead_ms) // 80
                post[tau:] = 0.0
            else:
                tau = len(post)
            entries.append({"k": k, "n": n, "x": x, "ai": ai, "post": post, "tau": min(tau, len(post)),
                            "dur_s": len(x) / 16000})
        self._turn_posts(entries)
        out = {}
        for en in entries:
            pt = torch.as_tensor(E.post_to_tokens(en["post"], en["n"], len(en["x"]) / 16000)).to(self.dev, torch.bfloat16)
            sp = self.sd_proj(pt[None])[0]
            out[en["k"]] = (en["n"], E.interleave(en["ai"], sp.to(emb_dtype)))
        if cache is not None:
            cache.update({k: v for k, v in out.items() if not k.startswith("__")})
            return [out[k] if k in out else cache[k] for k in keys]
        return [out[k] for k in keys]


def _pop_opt(argv: list[str], name: str, default: str = "") -> tuple[list[str], str]:
    out, val, i = [], default, 0
    while i < len(argv):
        a = argv[i]
        if a == name:
            val = argv[i + 1]
            i += 2
            continue
        if a.startswith(name + "="):
            val = a.split("=", 1)[1]
            i += 1
            continue
        out.append(a)
        i += 1
    return out, val


def main(argv: list[str] | None = None) -> None:
    from experiments.mist_w3po_v1.evaluation import online_decode as od

    argv = list(sys.argv[1:] if argv is None else argv)
    argv, path = _pop_opt(argv, "--turn-crf")
    argv, tdev = _pop_opt(argv, "--turn-device", "")
    if not path:                                     # default: exactly online_decode
        sys.argv = [sys.argv[0]] + argv
        od.main()
        return
    layer = load_layer(path)
    print(f"turn-crf {path}: g={float(layer.g):.6g} init={layer.init_src} e_mlp={layer.e_mlp is not None}", flush=True)

    class _Bound(TurnEngine):
        @classmethod
        def from_merged(cls, *a, **kw):
            eng = super().from_merged(*a, **kw)
            eng.set_turn(layer, torch.device(tdev) if tdev else None)
            return eng

    orig = od.Engine
    od.Engine = _Bound
    try:
        sys.argv = [sys.argv[0]] + argv
        od.main()
    finally:
        od.Engine = orig


if __name__ == "__main__":
    main()
