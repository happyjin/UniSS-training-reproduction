"""E2E-v0's voice per speaker (docs/multi_speaker_si_plan_v2.zh-CN.md): does each speaker's translated speech
sound like that speaker, and not like the others in the piece?

For every talker stream "<piece>__S<k>" (si_infer results), the WavLM-SV embedding of its speech is compared with
the embedding of the reference speaker its label was matched to (e2e_ms_post score, "map") -- that speaker's own
utterances in the piece, from the clean IHM source audio -- and with every other reference speaker of the piece.
SIM = cosine to its own speaker; confused = some other speaker of the piece is closer. The cosine between the
reference speakers themselves is reported as the separability the comparison has to work with.

    python -m experiments.uniss_qwen3_omni_v1.evaluation.e2e_ms_voice --pieces ... --score-rows ... --results ... --out ...
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import numpy as np
import pyarrow.dataset as pads
import soundfile as sf
import torch

WAVLM = "/opt/dlami/nvme/jasonleeeli/evaluation_models/wavlm-base-plus-sv"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pieces", required=True)
    ap.add_argument("--score-rows", required=True, help="<score>.rows.jsonl from e2e_ms_post score")
    ap.add_argument("--results", required=True, help="si_infer results.jsonl of the per-speaker streams")
    ap.add_argument("--data", default="data/ami_multispk_si_v1")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    from transformers import AutoFeatureExtractor, WavLMForXVector

    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    fe = AutoFeatureExtractor.from_pretrained(WAVLM)
    model = WavLMForXVector.from_pretrained(WAVLM).eval().to(dev)

    @torch.no_grad()
    def emb(x: np.ndarray) -> np.ndarray:
        x = x[: 16000 * 20]
        if len(x) < 8000:
            x = np.pad(x, (0, 8000 - len(x)))
        v = model(**{k: t.to(dev) for k, t in fe(x, sampling_rate=16000, return_tensors="pt").items()}).embeddings[0].cpu()
        return torch.nn.functional.normalize(v, dim=0).numpy()

    data = Path(args.data)
    src = {r["record_id"]: r for r in pads.dataset(data / "index/chain_en_zh_ihm", format="parquet", partitioning="hive")
           .to_table(columns=["record_id", "source_audio_shard_uri", "source_audio_byte_offset",
                              "source_audio_byte_length"]).to_pylist()}

    def clean(rid: str) -> np.ndarray | None:
        r = src.get(rid)
        if r is None:
            return None
        parts = r["source_audio_shard_uri"].split("/")
        path = data / "shards" / "source_ihm" / Path(*parts[parts.index("source_subset=ihm") + 1:])
        with open(path, "rb") as fh:
            fh.seek(r["source_audio_byte_offset"])
            x, _ = sf.read(io.BytesIO(fh.read(r["source_audio_byte_length"])), dtype="float32")
        return x

    pieces = {json.loads(l)["id"]: json.loads(l) for l in open(args.pieces, encoding="utf-8")}
    maps = {json.loads(l)["id"]: json.loads(l)["map"] for l in open(args.score_rows, encoding="utf-8")}
    streams = {}
    for l in open(args.results, encoding="utf-8"):
        r = json.loads(l)
        pid, spk = r["id"].rsplit("__", 1)
        streams.setdefault(pid, {})[spk] = r["speech_path"]
    sims, conf, sep, rows = [], [], [], []
    for pid, ss in streams.items():
        p = pieces[pid]
        refs = {}
        for u in p["utts"]:
            if u["label"] and not u["hesitation"]:
                x = clean(u["record_id"])
                if x is not None:
                    refs.setdefault(f"S{u['label']}", []).append(x)
        ref_e = {k: emb(np.concatenate(v)) for k, v in refs.items()}
        ks = sorted(ref_e)
        sep += [float(ref_e[a] @ ref_e[b]) for i, a in enumerate(ks) for b in ks[i + 1:]]
        for spk, path in ss.items():
            own = maps.get(pid, {}).get(spk)
            if own not in ref_e:
                continue
            x, _ = sf.read(path, dtype="float32")
            e = emb(x)
            s_own = float(e @ ref_e[own])
            others = [float(e @ ref_e[k]) for k in ks if k != own]
            sims.append(s_own)
            if others:
                conf.append(max(others) > s_own)
            rows.append({"id": pid, "spk": spk, "ref": own, "sim": round(s_own, 4),
                         "max_other": round(max(others), 4) if others else None, "n_ref": len(ks)})
    res = {"streams": len(sims), "SIM_mean": round(float(np.mean(sims)), 4) if sims else None,
           "streams_in_multi_ref_pieces": len(conf),
           "confusion_rate": round(float(np.mean(conf)), 4) if conf else None,
           "ref_ref_cos_mean": round(float(np.mean(sep)), 4) if sep else None}
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    with open(out.with_suffix(".rows.jsonl"), "x", encoding="utf-8") as fh:
        fh.writelines(json.dumps(r) + "\n" for r in rows)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
