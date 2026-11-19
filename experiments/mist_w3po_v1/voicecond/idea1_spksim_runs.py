"""Speaker-identity check for Idea 1 talker runs (M2, reported, not gated): TitaNet-large (independent of BiCodec)
cosine between each generated stream (talker results speech_path, results.jsonl or part_*.jsonl) and a reference wav
<ref dir>/<stream id>.wav -- for the esports streams the HDemucs vocals stem of that stream's prompt (the cleanest
available sample of the source speaker), for ID-1 the clean speech s. Paired summary per run vs the first run.
    venvs/nemo_diar/bin/python -m experiments.mist_w3po_v1.voicecond.idea1_spksim_runs --ref-dir <dir> [--ref-suffix _s]
        --runs <label>=<talker dir> ... --out <json> [--device cuda:0]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

NEMO = ("/opt/dlami/nvme/jasonleeeli/cache/hf_v6/models--nvidia--speakerverification_en_titanet_large/snapshots/"
        "0dc382f40121a5fbd34db10a2bb04d826c2be6a8/speakerverification_en_titanet_large.nemo")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref-dir", required=True)
    ap.add_argument("--ref-suffix", default="")
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    import soundfile as sf
    import torch
    from nemo.collections.asr.models import EncDecSpeakerLabelModel
    m = EncDecSpeakerLabelModel.restore_from(NEMO, map_location=a.device).eval()

    @torch.no_grad()
    def emb(p):
        x, sr = sf.read(str(p), dtype="float32", always_2d=True)
        x = x.mean(1)
        if sr != 16000:
            import torchaudio.functional as AF
            x = AF.resample(torch.from_numpy(x), sr, 16000).numpy()
        x = torch.from_numpy(x)[None].to(a.device)
        _, e = m.forward(input_signal=x, input_signal_length=torch.tensor([x.shape[1]], device=a.device))
        e = e[0].float().cpu().numpy()
        return e / (np.linalg.norm(e) + 1e-9)

    refs, res, per = {}, {}, {}
    for spec in a.runs:
        label, d = spec.split("=", 1)
        T = Path(d)
        files = [T / "results.jsonl"] if (T / "results.jsonl").exists() else sorted(T.glob("part_*.jsonl"))
        rows = [json.loads(l) for f in files for l in open(f, encoding="utf-8")]
        sims = {}
        for r in rows:
            if not r.get("speech_path") or r.get("error") not in (None, "None"):
                continue
            rp = Path(a.ref_dir) / f"{r['id']}{a.ref_suffix}.wav"
            if not rp.exists():
                continue
            if r["id"] not in refs:
                refs[r["id"]] = emb(rp)
            try:
                sims[r["id"]] = float(emb(r["speech_path"]) @ refs[r["id"]])
            except Exception:  # noqa: BLE001
                continue
        per[label] = sims
        v = np.array(list(sims.values()))
        res[label] = {"n": len(v), "median": float(np.median(v)), "mean": float(v.mean())}
    first = a.runs[0].split("=", 1)[0]
    rng = np.random.default_rng(0)
    for label in per:
        if label == first:
            continue
        ids = sorted(set(per[label]) & set(per[first]))
        d = np.array([per[label][i] - per[first][i] for i in ids])
        bs = np.median(d[rng.integers(0, len(d), (10000, len(d)))], 1)
        res[label][f"vs_{first}"] = {"median_diff": float(np.median(d)),
                                     "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "n": len(d)}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": res, "per_stream": per}), encoding="utf-8")
    print(json.dumps(res))


if __name__ == "__main__":
    main()
