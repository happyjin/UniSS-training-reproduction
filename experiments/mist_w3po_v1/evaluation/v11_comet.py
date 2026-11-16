"""v11 P1-4 (c): COMET (Unbabel/wmt22-comet-da) without the unbabel-comet package (no new packages may be installed):
the RegressionMetric forward re-implemented with transformers on the released checkpoint (hparams.yaml: XLM-R large,
layer mix with sparsemax layer weights and no layer norm, average pooling over non-pad tokens, features
[mt, ref, mt*ref, |mt-ref|, mt*src, |mt-src|], feed-forward 6144-3072-1024-1 with Tanh, no final activation).
`check` scores the package README example (10 到 15 分钟可以送到吗 ...), for which unbabel-comet prints
[0.8417137265205383, 0.7745391726493835]; this implementation gives [0.8417137265205383, 0.7745389342308044]
(checked 2026-11-15).

    python experiments/mist_w3po_v1/evaluation/v11_comet.py check
    python experiments/mist_w3po_v1/evaluation/v11_comet.py score --scores <v11_metrics scores dir> --out <dir> [--only <re>]
       every unit row (std run) of every score file -> <out>/<key>.json: per-unit COMET of hyp and hyp_dedup
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path

import torch

HUB = "/opt/dlami/nvme/jasonleeeli/cache/huggingface/hub"
CKPT = glob.glob(f"{HUB}/models--Unbabel--wmt22-comet-da/snapshots/*/checkpoints/model.ckpt")[0]
XLMR = glob.glob(f"{HUB}/models--FacebookAI--xlm-roberta-large/snapshots/*")[0]


def sparsemax(z: torch.Tensor) -> torch.Tensor:
    zs, _ = torch.sort(z, descending=True)
    k = torch.arange(1, z.numel() + 1, dtype=z.dtype, device=z.device)
    cs = zs.cumsum(0)
    sup = 1 + k * zs > cs
    kz = sup.nonzero().max() + 1
    tau = (cs[kz - 1] - 1) / kz
    return torch.clamp(z - tau, min=0)


class Comet(torch.nn.Module):
    def __init__(self, device="cuda:0"):
        super().__init__()
        from transformers import AutoTokenizer, XLMRobertaConfig, XLMRobertaModel
        self.tok = AutoTokenizer.from_pretrained(XLMR)
        self.enc = XLMRobertaModel(XLMRobertaConfig.from_pretrained(XLMR), add_pooling_layer=False)
        sd = torch.load(CKPT, map_location="cpu", weights_only=False)
        sd = sd.get("state_dict", sd)
        enc = {k[len("encoder.model."):]: v for k, v in sd.items() if k.startswith("encoder.model.")}
        missing, unexpected = self.enc.load_state_dict(enc, strict=False)
        missing = [m for m in missing if "position_ids" not in m]
        assert not missing, missing[:5]
        n = len([k for k in sd if k.startswith("layerwise_attention.scalar_parameters.")])
        self.w = torch.nn.Parameter(torch.cat([sd[f"layerwise_attention.scalar_parameters.{i}"].view(1) for i in range(n)]))
        self.gamma = torch.nn.Parameter(sd["layerwise_attention.gamma"].view(1))
        ff = sorted({k for k in sd if k.startswith("estimator.ff.")})
        lin = sorted({int(k.split(".")[2]) for k in ff})
        self.lins = torch.nn.ModuleList()
        for i in lin:
            W, b = sd[f"estimator.ff.{i}.weight"], sd[f"estimator.ff.{i}.bias"]
            L = torch.nn.Linear(W.shape[1], W.shape[0])
            L.weight.data, L.bias.data = W, b
            self.lins.append(L)
        self.keys_left = sorted(k for k in sd if not (k.startswith("encoder.model.") or k.startswith("layerwise_attention.")
                                                   or k.startswith("estimator.ff.")))
        self.eval().to(device)
        self.device = device

    @torch.no_grad()
    def embed(self, texts: list[str]) -> torch.Tensor:
        t = self.tok(texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(self.device)
        out = self.enc(input_ids=t["input_ids"], attention_mask=t["attention_mask"], output_hidden_states=True)
        hs = torch.stack(out.hidden_states, 0)                     # L, B, T, H
        wts = sparsemax(self.w.float())
        emb = self.gamma * (wts.view(-1, 1, 1, 1) * hs.float()).sum(0)
        m = t["attention_mask"].unsqueeze(-1).float()
        return (emb * m).sum(1) / m.sum(1)

    @torch.no_grad()
    def score(self, src: list[str], mt: list[str], ref: list[str], bs: int = 64) -> list[float]:
        res = []
        for i in range(0, len(mt), bs):
            s, h, r = self.embed(src[i:i + bs]), self.embed(mt[i:i + bs]), self.embed(ref[i:i + bs])
            x = torch.cat([h, r, h * r, (h - r).abs(), h * s, (h - s).abs()], 1)
            for j, L in enumerate(self.lins):
                x = L(x)
                if j < len(self.lins) - 1:
                    x = torch.tanh(x)
            res += x.view(-1).tolist()
        return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=["check", "score"])
    ap.add_argument("--scores", default="eval_outputs/mist_w3po_v1/v11_metrics/scores")
    ap.add_argument("--out", default="eval_outputs/mist_w3po_v1/v11_metrics/comet")
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    M = Comet()
    if a.cmd == "check":
        print("unused checkpoint keys:", M.keys_left)
        print(M.score(["10 到 15 分钟可以送到吗", "Pode ser entregue dentro de 10 a 15 minutos?"],
                      ["Can I receive my food in 10 to 15 minutes?", "Can you send it for 10 to 15 minutes?"],
                      ["Can it be delivered between 10 to 15 minutes?", "Can it be delivered between 10 to 15 minutes?"]))
        return
    od = Path(a.out)
    od.mkdir(parents=True, exist_ok=True)
    for f in sorted(Path(a.scores).glob("*.json")):
        if a.only and not re.search(a.only, f.stem):
            continue
        o = od / f.name
        if o.exists():
            print(f"{o} exists; kept", flush=True)
            continue
        rows = json.load(open(f))["std"]["unit_rows"]
        src, ref = [r["src"] for r in rows], [r["ref"] for r in rows]
        order = sorted(range(len(rows)), key=lambda i: -len(rows[i]["hyp"]) - len(ref[i]))
        out = {}
        for fld in ("hyp", "hyp_dedup"):
            sc = M.score([src[i] for i in order], [rows[i][fld] for i in order], [ref[i] for i in order])
            v = [0.0] * len(rows)
            for i, x in zip(order, sc):
                v[i] = x
            out[fld] = v
        res = {"key": f.stem, "units": [{"meeting": r["meeting"], "reg": r["reg"], "unit": r["unit"]} for r in rows],
               "comet": out["hyp"], "comet_dedup": out["hyp_dedup"],
               "mean": sum(out["hyp"]) / max(len(rows), 1), "mean_dedup": sum(out["hyp_dedup"]) / max(len(rows), 1)}
        with open(o, "x", encoding="utf-8") as fh:
            json.dump(res, fh)
        print(f"{f.stem}: COMET {res['mean']:.4f} dedup {res['mean_dedup']:.4f} ({len(rows)} units)", flush=True)


if __name__ == "__main__":
    main()
