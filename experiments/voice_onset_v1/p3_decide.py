"""voice_onset_v1 P3 decisions (reports/voice_onset_v1/P1_P3_prereg.zh-CN.md section 4) on the robust test main cells, UE
Thinker blocks. Per item: tau = floor(max(first write, 1000) / 1000) chunks heard; 1 s TitaNet chunk embeddings of the
source; trained Onset model -> deployed onset j = first hazard >= 0.5 (capped by the source length, K = 2) ->
g_ms = min(max(first write, 1000) + 1000 j, dur). Writes
  <hmo>/data/blocks_<d>.jsonl   UE blocks with write_ms = max(write_ms, g_ms) (text unchanged)        [UE_hmo]
  <hmo>/targets_<d>.jsonl       {id, key, tau_ms, g_ms, j, m (posterior mean at g)}                   [UE_hmo]
  <mbr>/targets_<d>.jsonl       same with j = 0 (posterior mean at tau), prompts = UE's               [UE_mbr]
CPU (nemo_diar).
    S1_DATA=<test data> python -m experiments.voice_onset_v1.p3_decide --model <p1/train/model.pt> --hmo <root> --mbr <root>
Refuses existing outputs.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
T = REPO / "eval_outputs/mist_w3po_v1/test_robust"
KEEP = {"rsc", "rsn", "obclean", "obid2", "obood1"}
SR = 16000


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--hmo", required=True)
    ap.add_argument("--mbr", required=True)
    a = ap.parse_args()
    H, M = Path(a.hmo), Path(a.mbr)
    for f in (H / "targets_en2zh.jsonl", M / "targets_en2zh.jsonl", H / "data" / "blocks_en2zh_w1.jsonl"):
        if f.exists():
            sys.exit(f"拒绝覆盖已有输出: {f}")
    (H / "data").mkdir(parents=True, exist_ok=True); M.mkdir(parents=True, exist_ok=True)
    import soundfile as sf
    import torch
    torch.set_num_threads(32)
    from nemo.collections.asr.models import EncDecSpeakerLabelModel
    from experiments.mist_w3po_v1.voicecond.idea1_spksim import NEMO
    from experiments.voice_onset_v1.p1_train import Onset
    ck = torch.load(a.model, map_location="cpu", weights_only=False)
    K = ck["K"]
    model = Onset(ck["mu0"]); model.load_state_dict(ck["state"]); model.eval()
    tn = EncDecSpeakerLabelModel.restore_from(NEMO, map_location="cpu").eval()

    @torch.no_grad()
    def emb(batch):
        t = torch.from_numpy(np.ascontiguousarray(batch, dtype=np.float32))
        _, e = tn.forward(input_signal=t, input_signal_length=torch.full((t.shape[0],), t.shape[1]))
        e = e.float().numpy()
        return e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-9)

    data = Path(os.environ["S1_DATA"])
    stats = {}
    for d in ("en2zh", "zh2en"):
        items = {r["id"]: r for r in jl(data / f"items_{d}.jsonl")}
        blocks = [r for r in jl(T / "UE" / f"thinker_{d}" / f"blocks_{d}_w1.jsonl") if r["id"].split("__")[0] in KEEP]
        th, tm, hb = [], [], []
        waits = {}
        for r in blocks:
            it = items[r["id"]]
            x, sr = sf.read(it["audio"], dtype="float32", always_2d=True)
            assert sr == SR
            x = x.mean(1)[: int(it["dur_ms"]) * 16]
            dur = len(x) // 16
            w0 = max(r["blocks"][0]["write_ms"], 1000)
            tau = w0 // 1000
            nfull = max(len(x) // SR, 1)
            jmax = max(0, min(K, nfull - tau))
            n = tau + K
            ch = np.zeros((n, SR), np.float32)
            for k in range(min(n, nfull)):
                ch[k] = x[k * SR:(k + 1) * SR] if (k + 1) * SR <= len(x) else np.pad(x[k * SR:], (0, (k + 1) * SR - len(x)))
            Y = emb(ch)
            rms = 10 * np.log10((ch.astype(np.float64) ** 2).mean(1) + 1e-10)
            with torch.no_grad():
                m, logP, w, yp = model.posterior(torch.tensor(Y)[None], torch.tensor(rms, dtype=torch.float32)[None])
                h = model.hazards(m, logP, w, yp, torch.tensor([tau]))[0].numpy() if tau + K - 1 <= n else np.ones(K)
            j = next((jj for jj in range(jmax) if h[jj] >= 0.5), jmax)
            g = min(w0 + 1000 * j, dur) if j > 0 else r["blocks"][0]["write_ms"]
            key = f"{r['id'].replace('/', '__')}_voice"
            cell = r["id"].split("__")[0]
            waits.setdefault(cell, []).append(j)
            th.append({"id": r["id"], "key": key, "tau_ms": r["blocks"][0]["write_ms"], "g_ms": int(g), "j": int(j),
                       "h": [round(float(v), 4) for v in h], "m": [round(float(v), 6) for v in m[0, tau + j].numpy()]})
            tm.append({"id": r["id"], "key": key, "tau_ms": r["blocks"][0]["write_ms"], "g_ms": r["blocks"][0]["write_ms"],
                       "j": 0, "m": [round(float(v), 6) for v in m[0, tau].numpy()]})
            nb = [dict(b, write_ms=max(b["write_ms"], int(g))) for b in r["blocks"]]
            hb.append(dict(r, blocks=nb))
        for root, rows in ((H, th), (M, tm)):
            with open(root / f"targets_{d}.jsonl", "x", encoding="utf-8") as fh:
                fh.writelines(json.dumps(x) + "\n" for x in rows)
        with open(H / "data" / f"blocks_{d}_w1.jsonl", "x", encoding="utf-8") as fh:
            fh.writelines(json.dumps(x, ensure_ascii=False) + "\n" for x in hb)
        stats[d] = {c: {"n": len(v), "mean_wait_s": round(float(np.mean(v)), 3), "dist": np.bincount(v, minlength=K + 1).tolist()}
                    for c, v in sorted(waits.items())}
        print(json.dumps({d: stats[d]}), flush=True)
    with open(H / "decide_stats.json", "x", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=1)


if __name__ == "__main__":
    main()
