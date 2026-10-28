"""Chinese AMI meetings for zh->en multi-speaker interpreting (docs/multi_speaker_si_plan_v9_bidir_abc.zh-CN.md §2, §11).

No new speech is synthesised: every AMI utterance already has its Chinese speech from the en->zh chain (one wav per
utterance, the original speaker's voice cloned). These clips are laid back on each meeting's clock so the Chinese
meeting keeps the speakers, turns and overlaps of the English one; the English transcripts become the references.

  mix      per meeting: each clip's leading / trailing silence trimmed (40 dB under its peak, 50 ms kept), optionally
           its inner pauses squeezed (--squeeze s), its loudness set to its speaker's median loudness in the IHM
           source, placed at the utterance's original start (a speaker's next clip waits until the previous one
           ends: one person cannot say two things at once), summed, the meeting's own background (IHM-mix frames
           with no annotated speech) laid under it, scaled down if it would clip. Writes
           session_audio/<--name>/<meeting>.wav and <out>/utts/<meeting>.jsonl (both clocks, texts, quality flags)
  audit    the checks of plan §11.2 A1-A7 and the §11.5 quality gate: coverage, trimmed length against the English
           utterance, inner silences, overlap statistics against the English meeting, self-overlap, clipping,
           intelligibility (the chain's Whisper CER); reference windows on the Chinese clock (utterances that overlap
           or touch, <= 20 s) and how many the gate keeps, per split. Writes <out>/audit.json

    python -m experiments.uniss_qwen3_omni_v1.data_clean.build_ami_zh_mix mix --out data/ami_multispk_si_v1/zh_mix_v1
    python -m experiments.uniss_qwen3_omni_v1.data_clean.build_ami_zh_mix audit --out data/ami_multispk_si_v1/zh_mix_v1
"""

from __future__ import annotations

import argparse
import io
import json
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import soundfile as sf

DATA = Path("data/ami_multispk_si_v1")
SR = 16000
HOP = 160   # 10 ms
GATE = {"max_cer": 0.3, "max_inner_sil_s": 1.0, "max_ratio": 3.0, "short_en_s": 1.0}


def frame_db(x: np.ndarray) -> np.ndarray:
    n = len(x) // HOP
    if n == 0:
        return np.full(1, -120.0)
    e = np.sqrt(np.mean(x[: n * HOP].reshape(n, HOP) ** 2, axis=1) + 1e-12)
    return 20 * np.log10(e + 1e-12)


def trim(x: np.ndarray, below_db: float = 40.0, keep_s: float = 0.05):
    """Leading / trailing silence removed; returns the clip and its longest inner pause (s)."""
    db = frame_db(x)
    on = db > db.max() - below_db
    if not on.any():
        return x[:0], 0.0
    i0, i1 = int(np.argmax(on)), len(on) - int(np.argmax(on[::-1]))
    k = int(keep_s * SR / HOP)
    y = x[max(0, (i0 - k) * HOP): min(len(x), (i1 + k) * HOP)]
    inner, run = 0, 0
    for v in on[i0:i1]:
        run = 0 if v else run + 1
        inner = max(inner, run)
    return y, inner * HOP / SR


def squeeze(x: np.ndarray, max_pause_s: float, below_db: float = 40.0) -> np.ndarray:
    """Inner pauses longer than max_pause_s cut down to max_pause_s (taken from the pause's middle)."""
    db = frame_db(x)
    on = db > db.max() - below_db
    keep = int(max_pause_s * SR / HOP)
    out, i, n = [], 0, len(on)
    while i < n:
        j = i
        while j < n and on[j] == on[i]:
            j += 1
        a, b = i * HOP, j * HOP
        if not on[i] and i > 0 and j < n and j - i > keep:
            mid = (a + b) // 2
            out.append(x[mid - keep * HOP // 2: mid + keep * HOP // 2])
        else:
            out.append(x[a:b])
        i = j
    out.append(x[n * HOP:])
    return np.concatenate(out) if out else x


def load_meta():
    import pyarrow.dataset as pads

    from experiments.uniss_qwen3_omni_v1.data_clean.build_ami_ms_rows import is_hesitation, review

    rows = pads.dataset(DATA / "index/chain_en_zh_ihm", format="parquet", partitioning="hive").to_table(
        columns=["record_id", "meeting_id", "speaker_id", "begin_time", "end_time", "transcript", "translation_text",
                 "shard_uri", "byte_offset", "byte_length"]).to_pylist()
    src = pads.dataset(DATA / "index/source_ihm", format="parquet", partitioning="hive").to_table(
        columns=["record_id", "meeting_id", "speaker_id", "begin_time", "end_time", "transcript"]).to_pylist()
    rev = review(DATA, GATE["max_cer"])
    cer = {}
    import pyarrow.dataset as pads2
    for r in pads2.dataset(DATA / "annotations/ihm_all/tts_whisper_error_rates", format="parquet",
                           partitioning="hive").to_table(columns=["record_id", "payload_json"]).to_pylist():
        try:
            cer[r["record_id"]] = json.loads(r["payload_json"]).get("whisper_cer")
        except (TypeError, ValueError):
            pass
    rms = {}
    for line in open(DATA / "reports/audio_check_rows.jsonl", encoding="utf-8"):
        x = json.loads(line)
        if x["side"] == "source" and x.get("ok"):
            rms[x["record_id"]] = x["rms"]
    tts = {r["record_id"]: r for r in rows}
    by_m = defaultdict(list)
    for u in src:   # every AMI utterance (also those without Chinese speech: they decide the windows' integrity)
        t = tts.get(u["record_id"])
        ok, zh, why = rev.get(u["record_id"], (False, "", "no_translation"))
        by_m[u["meeting_id"]].append({
            "record_id": u["record_id"], "speaker_id": u["speaker_id"], "en_begin": u["begin_time"],
            "en_end": u["end_time"], "en": u["transcript"], "zh": zh, "hes": is_hesitation(u["transcript"]),
            "why": why, "cer": cer.get(u["record_id"]), "src_rms": rms.get(u["record_id"]),
            "shard": t and t["shard_uri"], "off": t and t["byte_offset"], "len": t and t["byte_length"]})
    return by_m


def clip_of(u: dict):
    parts = u["shard"].split("/")
    path = DATA / "shards" / "target_ihm_en_zh" / Path(*parts[parts.index("source_subset=ihm") + 1:])
    with open(path, "rb") as fh:
        fh.seek(u["off"])
        x, sr = sf.read(io.BytesIO(fh.read(u["len"])), dtype="float32", always_2d=True)
    x = x.mean(1)
    if sr != SR:
        import scipy.signal as ss
        x = ss.resample_poly(x, SR, sr).astype(np.float32)
    return x


def mix_one(job):
    m, us, args = job
    sess = DATA / "session_audio" / "ihm_mix" / f"{m}.Mix-Headset.wav"
    bg, _ = sf.read(str(sess), dtype="float32", always_2d=True)
    bg = bg.mean(1)
    n_sess = len(bg)
    spk_rms = defaultdict(list)
    for u in us:
        if u["src_rms"]:
            spk_rms[u["speaker_id"]].append(u["src_rms"])
    spk_rms = {k: float(np.median(v)) for k, v in spk_rms.items()}
    out = np.zeros(n_sess + 30 * SR, np.float32)
    last_end = {}
    rows = []
    for u in sorted(us, key=lambda u: (u["en_begin"], u["en_end"])):
        r = {k: u[k] for k in ("record_id", "speaker_id", "en_begin", "en_end", "en", "zh", "hes", "why", "cer")}
        r.update(placed=False, flags=[])
        if u["hes"]:
            r["flags"].append("hesitation")
            rows.append(r)
            continue
        if not u["shard"] or u["why"] in ("no_text", "silent_or_bad_audio", "no_translation"):
            r["flags"].append("no_zh_speech")
            rows.append(r)
            continue
        try:
            x = clip_of(u)
        except Exception as e:   # noqa: BLE001
            r["flags"].append(f"read_error:{type(e).__name__}")
            rows.append(r)
            continue
        y, inner = trim(x)
        if args.squeeze > 0 and inner > args.squeeze:
            y = squeeze(y, args.squeeze)
            y, inner = trim(y)
        if len(y) < int(0.1 * SR):
            r["flags"].append("empty_after_trim")
            rows.append(r)
            continue
        act = frame_db(y)
        active = y[: (len(y) // HOP) * HOP].reshape(-1, HOP)[act > act.max() - 30]
        g = 1.0
        if spk_rms.get(u["speaker_id"]) and len(active):
            g = float(np.clip(spk_rms[u["speaker_id"]] / (np.sqrt(np.mean(active ** 2)) + 1e-9), 0.05, 10.0))
        y = y * g
        start = u["en_begin"]
        shift = max(0.0, last_end.get(u["speaker_id"], -1.0) + 0.05 - start)
        start += shift
        a = int(round(start * SR))
        if a + len(y) > len(out):
            out = np.concatenate([out, np.zeros(a + len(y) - len(out) + SR, np.float32)])
        out[a: a + len(y)] += y
        end = start + len(y) / SR
        last_end[u["speaker_id"]] = end
        en_dur = max(1e-3, u["en_end"] - u["en_begin"])
        r.update(placed=True, zh_begin=round(start, 3), zh_end=round(end, 3), shift_s=round(shift, 3),
                 ratio=round((end - start) / en_dur, 3), inner_sil_s=round(inner, 3), gain=round(g, 3))
        rows.append(r)
    n_out = max(n_sess, int(np.max(np.nonzero(out)[0]) + 1 if out.any() else n_sess) + SR // 2)
    out = out[:n_out]
    # background: the IHM-mix frames with no annotated speech (any utterance, hesitations included, +-0.2 s)
    busy = np.zeros(n_sess, bool)
    for u in us:
        busy[max(0, int((u["en_begin"] - 0.2) * SR)): int((u["en_end"] + 0.2) * SR)] = True
    quiet = bg[~busy]
    if args.background and len(quiet) > 5 * SR:
        reps = int(np.ceil(n_out / len(quiet)))
        out += np.tile(quiet, reps)[:n_out]
    peak = float(np.max(np.abs(out))) if len(out) else 0.0
    scale = 0.97 / peak if peak > 0.97 else 1.0
    out *= scale
    wav = Path(args.audio_dir or DATA / "session_audio" / args.name) / f"{m}.wav"
    sf.write(str(wav), out, SR, subtype="PCM_16")
    with open(Path(args.out) / "utts" / f"{m}.jsonl", "x", encoding="utf-8") as fh:
        fh.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    return {"meeting": m, "dur_s": round(n_out / SR, 1), "peak_before": round(peak, 3), "scale": round(scale, 3),
            "background_s": round(len(quiet) / SR, 1)}


def mix(args) -> None:
    out = Path(args.out)
    adir = Path(args.audio_dir or DATA / "session_audio" / args.name)
    if out.exists() or adir.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out} / {adir}")
    (out / "utts").mkdir(parents=True)
    adir.mkdir(parents=True)
    by_m = load_meta()
    meetings = sorted(m for m in by_m if (DATA / "session_audio/ihm_mix" / f"{m}.Mix-Headset.wav").exists())
    if args.limit:
        meetings = meetings[: args.limit]
    with Pool(args.procs) as p:
        res = p.map(mix_one, [(m, by_m[m], args) for m in meetings], chunksize=1)
    (out / "mix.json").write_text(json.dumps({"args": vars(args), "meetings": res}, indent=1), encoding="utf-8")
    print(json.dumps({"meetings": len(res), "out": str(out)}))


def overlap_stats(segs: list[tuple[str, float, float]]) -> dict:
    """Speech (union), overlap (>= 2 speakers) in seconds on 10 ms frames."""
    if not segs:
        return {"speech_s": 0.0, "overlap_s": 0.0}
    n = int(max(b for _, _, b in segs) * 100) + 2
    cnt = np.zeros(n, np.int16)
    per = defaultdict(lambda: np.zeros(n, bool))
    for s, a, b in segs:
        per[s][int(a * 100): int(b * 100)] = True
    for v in per.values():
        cnt += v
    return {"speech_s": float((cnt > 0).sum() / 100), "overlap_s": float((cnt > 1).sum() / 100)}


def groups_of(us: list[dict], key_b: str, key_e: str, max_s: float = 20.0) -> list[list[dict]]:
    us = sorted(us, key=lambda u: u[key_b])
    gs, cur, end = [], [], -1.0
    for u in us:
        if cur and (u[key_b] > end or u[key_e] - cur[0][key_b] > max_s):
            gs.append(cur)
            cur = []
        cur.append(u)
        end = max(end, u[key_e]) if len(cur) > 1 else u[key_e]
    if cur:
        gs.append(cur)
    return gs


def bad_reason(u: dict) -> str | None:
    """§11.5 gate for one non-hesitation utterance of the Chinese meeting."""
    if not u["placed"]:
        return [f for f in u["flags"] if f != "hesitation"][0] if u["flags"] else "not_placed"
    if u["cer"] is None:
        return "no_cer"
    if u["cer"] > GATE["max_cer"]:
        return "cer"
    if u["inner_sil_s"] > GATE["max_inner_sil_s"]:
        return "inner_silence"
    if u["ratio"] > GATE["max_ratio"] and (u["en_end"] - u["en_begin"]) >= GATE["short_en_s"]:
        return "too_long"
    return None


def audit(args) -> None:
    out = Path(args.out)
    split_of = json.load(open(DATA / "meetings.json"))
    U = {f.stem: [json.loads(l) for l in open(f, encoding="utf-8")] for f in sorted((out / "utts").glob("*.jsonl"))}
    res = {"gate": GATE, "splits": {}, "meetings": {}}
    agg = defaultdict(lambda: defaultdict(float))
    ratios = defaultdict(list)
    for m, us in U.items():
        sp = split_of[m]
        talk = [u for u in us if not u["hes"]]
        placed = [u for u in talk if u["placed"]]
        bad = defaultdict(int)
        for u in talk:
            b = bad_reason(u)
            u["_bad"] = b
            if b:
                bad[b] += 1
        for u in placed:
            d = u["en_end"] - u["en_begin"]
            ratios["lt1s" if d < 1 else "1to3s" if d < 3 else "ge3s"].append(u["ratio"])
        en = overlap_stats([(u["speaker_id"], u["en_begin"], u["en_end"]) for u in placed])
        zh = overlap_stats([(u["speaker_id"], u["zh_begin"], u["zh_end"]) for u in placed])
        self_ov = 0
        last = {}
        for u in sorted(placed, key=lambda u: u["zh_begin"]):
            if u["zh_begin"] < last.get(u["speaker_id"], -1) - 1e-3:
                self_ov += 1
            last[u["speaker_id"]] = max(last.get(u["speaker_id"], -1), u["zh_end"])
        # windows on the Chinese clock, built from every non-hesitation utterance (placed or not), then gated
        g_en = groups_of(placed, "en_begin", "en_end")
        g_zh = groups_of([u for u in placed], "zh_begin", "zh_end")
        kept = [g for g in g_zh if not any(u["_bad"] for u in g)]
        multi = lambda gs: sum(len({u["speaker_id"] for u in g}) > 1 for g in gs) / max(1, len(gs))   # noqa: E731
        mm = {"split": sp, "utts": len(talk), "placed": len(placed), "coverage": len(placed) / max(1, len(talk)),
              "bad": dict(bad), "overlap_ratio_en": en["overlap_s"] / max(1e-6, en["speech_s"]),
              "overlap_ratio_zh": zh["overlap_s"] / max(1e-6, zh["speech_s"]), "speech_s_en": en["speech_s"],
              "speech_s_zh": zh["speech_s"], "self_overlap": self_ov, "windows_en": len(g_en),
              "windows_zh": len(g_zh), "windows_kept": len(kept), "multi_share_en": multi(g_en),
              "multi_share_zh": multi(g_zh), "multi_share_kept": multi(kept)}
        res["meetings"][m] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in mm.items()}
        a = agg[sp]
        for k in ("utts", "placed", "self_overlap", "windows_en", "windows_zh", "windows_kept"):
            a[k] += mm[k]
        for k in ("speech_s_en", "speech_s_zh"):
            a[k] += mm[k]
        a["overlap_s_en"] += en["overlap_s"]
        a["overlap_s_zh"] += zh["overlap_s"]
        a["multi_en"] += mm["multi_share_en"] * len(g_en)
        a["multi_zh"] += mm["multi_share_zh"] * len(g_zh)
        a["multi_kept"] += mm["multi_share_kept"] * len(kept)
        for k, v in bad.items():
            a["bad_" + k] += v
    for sp, a in agg.items():
        res["splits"][sp] = {
            "utts": int(a["utts"]), "coverage": round(a["placed"] / a["utts"], 4),
            "overlap_ratio_en": round(a["overlap_s_en"] / a["speech_s_en"], 4),
            "overlap_ratio_zh": round(a["overlap_s_zh"] / a["speech_s_zh"], 4),
            "speech_h_en": round(a["speech_s_en"] / 3600, 2), "speech_h_zh": round(a["speech_s_zh"] / 3600, 2),
            "self_overlap": int(a["self_overlap"]), "windows_en": int(a["windows_en"]),
            "windows_zh": int(a["windows_zh"]), "windows_kept": int(a["windows_kept"]),
            "multi_share_en": round(a["multi_en"] / max(1, a["windows_en"]), 4),
            "multi_share_zh": round(a["multi_zh"] / max(1, a["windows_zh"]), 4),
            "multi_share_kept": round(a["multi_kept"] / max(1, a["windows_kept"]), 4),
            "bad": {k[4:]: int(v) for k, v in a.items() if k.startswith("bad_")}}
    res["ratio_quantiles"] = {k: {q: round(float(np.quantile(v, q)), 2) for q in (0.5, 0.75, 0.9, 0.99)}
                              for k, v in ratios.items()}
    mix_info = json.load(open(out / "mix.json"))
    res["clipping"] = {"meetings_scaled": sum(1 for r in mix_info["meetings"] if r["scale"] < 1),
                       "min_scale": min(r["scale"] for r in mix_info["meetings"])}
    p = out / args.name
    if p.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {p}")
    p.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps({"splits": res["splits"], "ratio_quantiles": res["ratio_quantiles"], "clipping": res["clipping"]},
                     indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("mix")
    a.add_argument("--out", required=True)
    a.add_argument("--name", default="zh_mix")
    a.add_argument("--audio-dir", default="", help="where the meeting wavs go (default session_audio/<name>)")
    a.add_argument("--squeeze", type=float, default=0.0, help="cut inner pauses longer than this (s); 0 = off")
    a.add_argument("--background", type=int, default=1)
    a.add_argument("--procs", type=int, default=48)
    a.add_argument("--limit", type=int, default=0)
    a = sub.add_parser("audit")
    a.add_argument("--out", required=True)
    a.add_argument("--name", default="audit.json")
    args = ap.parse_args()
    {"mix": mix, "audit": audit}[args.cmd](args)


if __name__ == "__main__":
    main()
