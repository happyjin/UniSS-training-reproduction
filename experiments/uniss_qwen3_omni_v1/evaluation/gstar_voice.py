"""Voice per speaker of G-STAR-SI speech (paper v2): does each output speaker's translated speech sound like the
person it was attributed to, and not like the others in the window?

Every talker stream "<window>__S<k>" (si_infer results) is mapped to a reference speaker by the session-level
matching of gstar_score (one chrF Hungarian matching of slots to speakers per meeting). Its WavLM-SV embedding is
compared with that speaker's reference voice -- up to --ref-s s of their own clean IHM utterances in the same meeting --
and with the voices of the other reference speakers of the window.
  SIM        cosine to the attributed speaker (streams whose slot matched a speaker)
  confusion  share of streams in multi-speaker windows for which another speaker of the window is closer
Beside them, the cosine between the reference voices themselves (how separable the speakers are).

    python -m experiments.uniss_qwen3_omni_v1.evaluation.gstar_voice --windows ... --hyp ... --results ... --out ...
"""

from __future__ import annotations

import argparse
import io
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.dataset as pads
import soundfile as sf
import torch

from experiments.uniss_qwen3_omni_v1.evaluation.gstar_score import match

WAVLM = "/opt/dlami/nvme/jasonleeeli/evaluation_models/wavlm-base-plus-sv"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--windows", required=True)
    ap.add_argument("--hyp", required=True)
    ap.add_argument("--results", required=True, help="si_infer results.jsonl of the per-slot streams")
    ap.add_argument("--data", default="data/ami_multispk_si_v1")
    ap.add_argument("--ref-s", type=float, default=20.0)
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

    W = {w["id"]: w for w in map(json.loads, open(args.windows, encoding="utf-8"))}
    H = {h["id"]: h for h in map(json.loads, open(args.hyp, encoding="utf-8"))}
    # session-level slot -> speaker matching, as gstar_score does it
    hyp_m, ref_m = defaultdict(lambda: defaultdict(str)), defaultdict(lambda: defaultdict(str))
    utts_m = defaultdict(list)
    for wid, w in W.items():
        if wid not in H:
            continue
        for s in H[wid]["segments"]:
            hyp_m[w["meeting"]][s["spk"]] += s["text"]
        for u in sorted(w["utts"], key=lambda u: u["begin_s"]):
            if not u["hesitation"]:
                ref_m[w["meeting"]][u["speaker_id"]] += u["zh"].strip()
                utts_m[(w["meeting"], u["speaker_id"])].append(u)
    smap = {m: match(dict(hyp_m[m]), dict(ref_m[m])) for m in hyp_m}
    # reference voice per (meeting, speaker): their longest clean utterances up to --ref-s s
    ref_e = {}
    for key, us in utts_m.items():
        xs, tot = [], 0.0
        for u in sorted(us, key=lambda u: u["end_s"] - u["begin_s"], reverse=True):
            x = clean(u["record_id"])
            if x is None:
                continue
            xs.append(x)
            tot += len(x) / 16000
            if tot >= args.ref_s:
                break
        if xs:
            ref_e[key] = emb(np.concatenate(xs))
    sims, conf, sep, rows = [], [], [], []
    for line in open(args.results, encoding="utf-8"):
        r = json.loads(line)
        wid, k = r["id"].rsplit("__S", 1)
        w = W[wid]
        m = w["meeting"]
        own = smap.get(m, {}).get(int(k))
        if own is None or (m, own) not in ref_e or not r.get("speech_path"):
            continue
        x, _ = sf.read(r["speech_path"], dtype="float32")
        if len(x) < 1600:
            continue
        e = emb(x)
        s_own = float(e @ ref_e[(m, own)])
        others = sorted({u["speaker_id"] for u in w["utts"] if not u["hesitation"]} - {own})
        o = [float(e @ ref_e[(m, s)]) for s in others if (m, s) in ref_e]
        sims.append(s_own)
        if o:
            conf.append(max(o) > s_own)
        rows.append({"id": r["id"], "ref": own, "sim": round(s_own, 4), "max_other": round(max(o), 4) if o else None})
    for m in {k[0] for k in ref_e}:
        ks = sorted(k for k in ref_e if k[0] == m)
        sep += [float(ref_e[a] @ ref_e[b]) for i, a in enumerate(ks) for b in ks[i + 1:]]
    res = {"streams": len(sims), "SIM_mean": round(float(np.mean(sims)), 4) if sims else None,
           "streams_in_multi_windows": len(conf), "confusion_rate": round(float(np.mean(conf)), 4) if conf else None,
           "ref_ref_cos_mean": round(float(np.mean(sep)), 4) if sep else None}
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    with open(out.with_suffix(".rows.jsonl"), "x", encoding="utf-8") as fh:
        fh.writelines(json.dumps(r) + "\n" for r in rows)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
