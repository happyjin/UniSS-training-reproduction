"""OpenBroadcast-SI v1: ONE evaluation command for any speech run on the benchmark (reports/mist_w3po_v1/
voicecond_bench.zh-CN.md). For the run <name> and its second TTS seed <name>_tts2 (decode/<n>, speech/<n>/talker):

  tcpASR-BLEU     decode/<n>/score_asr.json tcpBLEU_all (+ ov / 1spk): overlap_score with the REAL CV speaker ids
  text tcpBLEU    decode/<n>/score_text.json tcpBLEU_all (the Thinker text; identical for both seeds)
  blind BLEU      speaker-blind corpus BLEU: per stream, every output segment in time order (window start + segment
                  start) against every reference utterance in time order (game_blind_bleu's rule); ASR and text
  speech LAAL     chunk protocol: speech_latency_lit on the condition's chunk-protocol windows (zh2en via on_refset
                  --dr ali, en2zh via en2zh_lat) -> lat_chunk_<n>.json; unit.speech_laal_online.all.mean is the primary
                  latency (per-unit speech LAAL); EndOffset = unit.speech_end_online.all.mean (supplement only)
  SQUIM           torchaudio SQUIM objective (reference-free) STOI / PESQ / SI-SDR of every talker stream (first 20 s,
                  game_muffle_analysis.squim)
  texture         spectral flatness 300-4000 Hz and spectral contrast (game_voice_texture.feats; higher contrast and
                  lower flatness = less muffled)
  SIM             speaker similarity: WavLM-base-plus-SV (x-vector, independent of BiCodec) cosine between the talker
                  stream's speech and its real speaker's CLEAN CoVoST clips of that stream (mean of the clip
                  embeddings); the talker item (window, slot) is assigned to the real speaker whose utterances overlap
                  the slot's diarizer activity (> 0.5) in the window most; unassigned items are counted, not scored
Per-item rows -> items_<n>.jsonl, per-stream means (for paired tests by stream key) and run medians / means ->
eval_outputs/mist_w3po_v1/voicecond_bench/<name>/metrics.json. CPU only. Refuses to overwrite metrics.json.

    python -m experiments.mist_w3po_v1.voicecond.bench_eval --cond id2|ood1|clean --dir zh2en|en2zh --name <run>
        [--seeds 1|2] [--procs 32]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
DEC = REPO / "eval_outputs/mist_w3po_v1/decode"
SPD = REPO / "eval_outputs/mist_w3po_v1/speech"
OUT = REPO / "eval_outputs/mist_w3po_v1/voicecond_bench"
BD = REPO / "data/open_voicecond_v1/bench"
WD = "eval_outputs/mist_w3po_v1/data"
WAVLM = "/opt/dlami/nvme/jasonleeeli/evaluation_models/wavlm-base-plus-sv"
FRAME = 0.08
_M: dict = {}


def wins(cond: str, d: str) -> tuple[str, str]:
    return (f"{WD}/windows_obsi_{cond}_{d}_c2g06_v1/windows_test_online_{d}.jsonl",
            f"{WD}/windows_obsi_{cond}_{d}_c2g06_chunk_v1/windows_dev_online.jsonl")


def blind_bleu(d: str, W: dict, hyp_path: Path, refs: list[dict]) -> dict:
    from sacrebleu.metrics import BLEU
    hyp, ref = defaultdict(list), defaultdict(list)
    for h in map(json.loads, open(hyp_path, encoding="utf-8")):
        w = W.get(h["id"])
        if not w:
            continue
        for s in h.get("segments") or []:
            if (s.get("text") or "").strip():
                hyp[w["meeting"]].append((w["start_s"] + s["start_s"], s["text"].strip()))
    for r in refs:
        ref[r["meeting"]].append((r["start_s"], r["utts"][0]["ref"].strip()))
    sep, tok = (" ", "13a") if d == "zh2en" else ("", "zh")
    ms = sorted(ref)
    H = [sep.join(t for _, t in sorted(hyp[m])) for m in ms]
    R = [sep.join(t for _, t in sorted(ref[m])) for m in ms]
    sb = BLEU(tokenize=tok, effective_order=True)
    return {"corpus": round(BLEU(tokenize=tok).corpus_score(H, [R]).score, 2),
            "per_stream": {m: round(sb.sentence_score(h, [r]).score, 2) for m, h, r in zip(ms, H, R)},
            "len_ratio": round(sum(len(x) for x in H) / max(1, sum(len(x) for x in R)), 3)}


def chunk_lat(cond: str, d: str, n: str, out: Path) -> dict | None:
    f = out / f"lat_chunk_{n}.json"
    if not f.exists():
        ref = f"data/open_voicecond_v1/bench/{cond}/{d}/refs_windows.jsonl"
        cw = wins(cond, d)[1]
        if d == "zh2en":
            cmd = [sys.executable, "-m", "experiments.mist_w3po_v1.evaluation.on_refset", "--ref-win", ref, "--dr", "ali",
                   "experiments.mist_w3po_v1.evaluation.speech_latency_lit", "--run", str(DEC / n), "--talker",
                   str(SPD / n / "talker"), "--win", cw, "--out", str(f)]
        else:
            cmd = [sys.executable, "-m", "experiments.mist_w3po_v1.evaluation.en2zh_lat", "--ref-win", ref, "--work",
                   str(f) + ".work", "--run", str(DEC / n), "--talker", str(SPD / n / "talker"), "--win", cw, "--out", str(f)]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONPATH=f"{REPO}:{REPO}/third_party/Megatron-LM")
        with open(str(f) + ".log", "w") as lg:
            r = subprocess.run(cmd, cwd=REPO, env=env, stdout=lg, stderr=subprocess.STDOUT)
        if r.returncode != 0 or not f.exists():
            print(f"chunk latency failed for {n} (see {f}.log)", flush=True)
            return None
    L = json.load(open(f))
    u = L["unit"]
    g = lambda k, s="all": (u.get(k) or {}).get(s) or {}  # noqa: E731
    return {"speech_laal": g("speech_laal_online").get("mean"), "speech_laal_ov": g("speech_laal_online", "ov").get("mean"),
            "speech_laal_1spk": g("speech_laal_online", "1spk").get("mean"), "end_offset": g("speech_end_online").get("mean"),
            "text_laal": g("text_laal_online").get("mean"), "n_units": g("speech_laal_online").get("n"), "file": str(f.relative_to(REPO))}


def models():
    if not _M:
        import torch
        from transformers import AutoFeatureExtractor, WavLMForXVector
        torch.set_num_threads(2)
        _M["fe"] = AutoFeatureExtractor.from_pretrained(WAVLM, local_files_only=True)
        _M["sv"] = WavLMForXVector.from_pretrained(WAVLM, local_files_only=True).eval()
    return _M["fe"], _M["sv"]


def embed(x: np.ndarray) -> np.ndarray | None:
    import torch
    if len(x) < 16000 * 0.2:
        return None
    fe, sv = models()
    with torch.no_grad():
        e = sv(**fe(x[: 16000 * 30], sampling_rate=16000, return_tensors="pt")).embeddings[0].float().numpy()
    return e / (np.linalg.norm(e) + 1e-9)


def _item_job(a):
    from experiments.mist_w3po_v1.evaluation.game_muffle_analysis import read16, squim
    from experiments.mist_w3po_v1.evaluation.game_voice_texture import feats
    iid, path, refs = a
    out = {"id": iid}
    try:
        y = read16(path)
        out["dur_s"] = round(len(y) / 16000, 2)
        q = squim(y)
        if q:
            out.update(q)
        f = feats(path)
        if f:
            out.update({"flatness": f["flatness"], "contrast": f["contrast"]})
        if refs:
            e = embed(y)
            key = tuple(refs)
            if key not in _M:
                R = [r for r in (embed(read16(p)) for p in refs) if r is not None]
                _M[key] = (lambda c: c / np.linalg.norm(c))(np.mean(R, 0)) if R else None
            if e is not None and _M[key] is not None:
                out["sim"] = float(e @ _M[key])
    except Exception as ex:  # noqa: BLE001
        out["error"] = repr(ex)[:300]
    return out


def assign_speakers(W: dict, items: list[str], utts: dict) -> dict:
    """talker item (window, slot) -> real speaker with the most overlap with the slot's activity, or None"""
    res = {}
    for iid in items:
        wid, slot = iid.rsplit("__S", 1)
        w = W[wid]
        post = np.load(w["post"]["lowlat"]).astype(np.float32)
        k = int(slot) - 1
        act = post[:, k] > 0.5 if k < post.shape[1] else np.zeros(len(post), bool)
        best, bo = None, 0.0
        tot = defaultdict(float)
        for f in np.where(act)[0]:
            a, b = w["start_s"] + f * FRAME, w["start_s"] + (f + 1) * FRAME
            for u in utts[w["meeting"]]:
                o = min(b, u["end_s"]) - max(a, u["begin_s"])
                if o > 0:
                    tot[u["speaker_id"]] += o
        for s, o in tot.items():
            if o > bo:
                best, bo = s, o
        res[iid] = best
    return res


def stats(v: list) -> dict | None:
    v = [x for x in v if x is not None and np.isfinite(x)]
    if not v:
        return None
    return {"mean": round(float(np.mean(v)), 4), "median": round(float(np.median(v)), 4), "n": len(v)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cond", required=True, choices=["id2", "ood1", "clean"])
    ap.add_argument("--dir", required=True, choices=["zh2en", "en2zh"])
    ap.add_argument("--name", required=True)
    ap.add_argument("--seeds", type=int, default=2, choices=[1, 2])
    ap.add_argument("--procs", type=int, default=32)
    a = ap.parse_args()
    out = OUT / a.name
    mf = out / "metrics.json"
    if mf.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {mf}")
    out.mkdir(parents=True, exist_ok=True)
    C = BD / a.cond / a.dir
    meetings = json.load(open(C / "meetings.json"))
    refs = [json.loads(l) for l in open(C / "refs_windows.jsonl", encoding="utf-8")]
    W = {w["id"]: w for w in map(json.loads, open(REPO / wins(a.cond, a.dir)[0], encoding="utf-8"))}
    assert {w["meeting"] for w in W.values()} <= set(meetings), "windows of another set"
    utts = {m: [json.loads(l) for l in open(C / "utts" / f"{m}.jsonl", encoding="utf-8")] for m in meetings}
    clip_path = {}
    for d2 in ("zh2en", "en2zh"):
        if d2 == a.dir:
            for c in map(json.loads, open(BD / "src" / d2 / "pool.jsonl", encoding="utf-8")):
                clip_path[c["id"]] = str(REPO / c["path"])
    spk_clips = defaultdict(list)
    for m, us in utts.items():
        for u in us:
            spk_clips[(m, u["speaker_id"])].append(clip_path[u["clip"]])
    names = [a.name] + ([a.name + "_tts2"] if a.seeds == 2 else [])
    res = {"cond": a.cond, "dir": a.dir, "name": a.name, "seeds": {}, "per_stream": {}}
    import multiprocessing as mp
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    for n in names:
        R = {"run": n}
        for kind in ("text", "asr"):
            f = DEC / n / f"score_{kind}.json"
            if f.exists():
                s = json.load(open(f))
                R[f"tcp_{kind}"] = {k: s.get(k) for k in ("tcpBLEU_all", "tcpBLEU_ov", "tcpBLEU_1spk", "omitted_1spk",
                                                          "omitted_ov", "inserted_1spk", "inserted_ov")}
        for kind, hp in (("text", DEC / n / "hyp.jsonl"), ("asr", DEC / n / "hyp_asr.jsonl")):
            if hp.exists():
                R[f"blind_{kind}"] = blind_bleu(a.dir, W, hp, refs)
        R["latency_chunk"] = chunk_lat(a.cond, a.dir, n, out)
        T = SPD / n / "talker"
        rows = [r for r in map(json.loads, open(T / "results.jsonl", encoding="utf-8"))]
        ok = [r for r in rows if r.get("error") is None and r.get("speech_path")]
        spk = assign_speakers(W, [r["id"] for r in ok], utts)
        jobs = []
        for r in ok:
            m = W[r["id"].rsplit("__S", 1)[0]]["meeting"]
            s = spk.get(r["id"])
            jobs.append((r["id"], r["speech_path"], spk_clips.get((m, s), []) if s else []))
        with mp.get_context("fork").Pool(a.procs) as P:
            items = P.map(_item_job, jobs, chunksize=4)
        with open(out / f"items_{n}.jsonl", "w", encoding="utf-8") as fh:
            for it in items:
                it["speaker"] = spk.get(it["id"])
                it["meeting"] = W[it["id"].rsplit("__S", 1)[0]]["meeting"]
                fh.write(json.dumps(it) + "\n")
        R["talker_items"] = {"results": len(rows), "with_speech": len(ok), "errors": len(rows) - len(ok),
                             "speaker_assigned": sum(1 for v in spk.values() if v)}
        for k in ("pesq", "sisdr", "stoi", "contrast", "flatness", "sim"):
            R[k] = stats([it.get(k) for it in items])
        res["seeds"][n] = R
        ps = defaultdict(lambda: defaultdict(list))
        for it in items:
            for k in ("pesq", "sisdr", "stoi", "contrast", "flatness", "sim"):
                if it.get(k) is not None:
                    ps[it["meeting"]][k].append(it[k])
        res["per_stream"][n] = {m: {"stream": meetings[m]["stream"], **{k: round(float(np.mean(v)), 4) for k, v in d.items()},
                                    "blind_asr": (R.get("blind_asr") or {}).get("per_stream", {}).get(m)}
                                for m, d in sorted(ps.items())}
    sd = list(res["seeds"].values())
    pick = lambda f: [f(r) for r in sd]  # noqa: E731
    res["summary"] = {
        "tcpASR_BLEU": pick(lambda r: (r.get("tcp_asr") or {}).get("tcpBLEU_all")),
        "blind_ASR_BLEU": pick(lambda r: (r.get("blind_asr") or {}).get("corpus")),
        "text_tcpBLEU": (sd[0].get("tcp_text") or {}).get("tcpBLEU_all"),
        "text_blind_BLEU": (sd[0].get("blind_text") or {}).get("corpus"),
        "speech_LAAL_chunk": pick(lambda r: (r.get("latency_chunk") or {}).get("speech_laal")),
        "EndOffset_chunk": pick(lambda r: (r.get("latency_chunk") or {}).get("end_offset")),
        "PESQ_median": pick(lambda r: (r.get("pesq") or {}).get("median")),
        "SI_SDR_median": pick(lambda r: (r.get("sisdr") or {}).get("median")),
        "contrast_median": pick(lambda r: (r.get("contrast") or {}).get("median")),
        "flatness_median": pick(lambda r: (r.get("flatness") or {}).get("median")),
        "SIM_mean": pick(lambda r: (r.get("sim") or {}).get("mean")),
    }
    with open(mf, "x", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, ensure_ascii=False)
    print(json.dumps(res["summary"]))


if __name__ == "__main__":
    main()
