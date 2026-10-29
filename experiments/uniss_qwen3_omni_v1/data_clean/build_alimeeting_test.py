"""Real Mandarin meetings as a Zh->En test set (paper v3, review: the Chinese AMI meetings are re-voiced, not real).

AliMeeting (OpenSLR 119; Yu et al., ICASSP 2022) has real Mandarin meetings of 2-4 people with one headset per
participant and human transcripts with speaker and time. Test only: nothing here is trained or tuned on.

  prep     headsets summed into one channel per meeting (as AMI IHM-mix; scaled down only if the peak exceeds 0.9)
           -> session_audio/near_mix/<m>.wav; utterances from each headset's TextGrid -> utts/<m>.jsonl
           (filler-only utterances are hesitations: heard, not translated, as in AMI); items to translate -> mt_items.jsonl
  audit    references (translate_zh2en_refs.py) checked: empty, Chinese left in the output, a prompt or explanation
           echoed, or a length ratio far from the source -> the utterance fails; writes refs_audit.json
  windows  utterance groups (overlapping or touching, at most 20 s; build_ami_zh_mix.groups_of) cut with up to 0.3 s /
           0.5 s of padding (half the gap to neighbouring speech); dropped if another utterance reaches into the window,
           if it is longer than 20.5 s, or if any of its utterances failed the reference audit. Diarizer posteriors of
           both presets sliced per window. Fields as the Chinese AMI test windows (dir zh2en: "zh" the source, "ref"
           the English reference).

    python -m experiments.uniss_qwen3_omni_v1.data_clean.build_alimeeting_test prep --raw data/raw/alimeeting/Test_Ali \
        --out data/alimeeting_si_v1
    python -m ... audit --out data/alimeeting_si_v1
    python -m ... windows --out data/alimeeting_si_v1
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

FILLERS = set("嗯啊呃哦额唉哎呀噢嘿哈呵喔诶欸哼嗷")
CJK = re.compile(r"[一-鿿]")


def textgrid(path: Path) -> list[tuple[float, float, str]]:
    s = path.read_text(encoding="utf-8")
    return [(float(a), float(b), t.strip()) for a, b, t in
            re.findall(r'xmin = ([\d.]+)\s*xmax = ([\d.]+)\s*text = "(.*?)"', s, re.S) if t.strip()]


def is_hes(zh: str) -> bool:
    return not set(re.sub(r"[\W_]", "", zh)) - FILLERS


def prep(args) -> None:
    raw, out = Path(args.raw), Path(args.out)
    near = next(raw.glob("*_near"))
    sess = out / "session_audio" / "near_mix"
    sess.mkdir(parents=True, exist_ok=True)
    (out / "utts").mkdir(exist_ok=True)
    by_m = defaultdict(list)
    for w in sorted((near / "audio_dir").glob("*.wav")):
        m, spk = re.match(r"(R\d+_M\d+)_N_(SPK\d+)", w.stem).groups()
        by_m[m].append((spk, w))
    meetings, items, stats = {}, [], defaultdict(int)
    for m, chans in sorted(by_m.items()):
        wav = sess / f"{m}.wav"
        if wav.exists():
            raise SystemExit(f"拒绝覆盖已有输出: {wav}")
        xs = [sf.read(str(w), dtype="float32")[0] for _, w in chans]
        x = np.zeros(max(len(a) for a in xs), dtype=np.float32)
        for a in xs:
            x[: len(a)] += a
        peak = float(np.abs(x).max())
        gain = min(1.0, 0.9 / max(peak, 1e-6))
        sf.write(str(wav), x * gain, 16000, subtype="PCM_16")
        us = []
        for spk, w in chans:
            tg = near / "textgrid_dir" / f"{w.stem}.TextGrid"
            for i, (b, e, zh) in enumerate(textgrid(tg)):
                u = {"record_id": f"{m}_{spk}_{i:04d}", "speaker_id": spk, "begin_s": b, "end_s": e, "zh": zh,
                     "hesitation": is_hes(zh)}
                us.append(u)
                stats["utts"] += 1
                stats["hesitations"] += u["hesitation"]
                if not u["hesitation"]:
                    items.append({"id": u["record_id"], "zh": zh})
        us.sort(key=lambda u: (u["begin_s"], u["end_s"]))
        with open(out / "utts" / f"{m}.jsonl", "x", encoding="utf-8") as fh:
            fh.writelines(json.dumps(u, ensure_ascii=False) + "\n" for u in us)
        meetings[m] = "test"
        print(f"{m}: {len(chans)} headsets, {len(x) / 16000:.0f} s, gain {gain:.3f}, {len(us)} utterances", flush=True)
    (out / "meetings.json").write_text(json.dumps(meetings, indent=0), encoding="utf-8")
    with open(out / "mt_items.jsonl", "x", encoding="utf-8") as fh:
        fh.writelines(json.dumps(i, ensure_ascii=False) + "\n" for i in items)
    print(json.dumps({"meetings": len(meetings), **stats, "to_translate": len(items)}))


def ref_fail(zh: str, en: str) -> str | None:
    if not en.strip():
        return "empty"
    if CJK.search(en):
        return "chinese_left"
    if re.search(r"(?i)\b(translat|here is|segment)\w*\b.*:", en) or "\n" in en.strip():
        return "echo"
    n_zh, n_en = len(re.findall(r"[一-鿿]", zh)), len(re.findall(r"[A-Za-z0-9']+", en))
    if n_zh >= 6 and not 0.15 <= n_en / n_zh <= 2.0:
        return "length_ratio"
    return None


def audit(args) -> None:
    out = Path(args.out)
    dst = out / "refs_audit.json"
    if dst.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {dst}")
    zh = {r["id"]: r["zh"] for r in map(json.loads, open(out / "mt_items.jsonl", encoding="utf-8"))}
    refs = {r["id"]: r["en"] for r in map(json.loads, open(out / "refs.jsonl", encoding="utf-8"))}
    fails, why = {}, defaultdict(int)
    for k, z in zh.items():
        r = ref_fail(z, refs.get(k, ""))
        if r:
            fails[k] = r
            why[r] += 1
    res = {"items": len(zh), "failed": len(fails), "by_reason": dict(why), "fails": fails,
           "examples": {r: [(k, zh[k], refs.get(k, "")) for k, v in list(fails.items()) if v == r][:5] for r in why}}
    dst.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k not in ("fails", "examples")}))


def windows(args) -> None:
    import torch

    from experiments.uniss_qwen3_omni_v1.data_clean.build_ami_gstar_si_rows import save_slice
    from experiments.uniss_qwen3_omni_v1.data_clean.build_ami_zh_mix import groups_of

    out = Path(args.out)
    dst = out / "windows_test_ref.jsonl"
    if dst.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {dst}")
    refs = {r["id"]: r["en"].strip() for r in map(json.loads, open(out / "refs.jsonl", encoding="utf-8"))}
    fails = json.load(open(out / "refs_audit.json"))["fails"]
    presets = args.presets.split(",")
    stats, W = defaultdict(int), []
    for m in sorted(json.load(open(out / "meetings.json"))):
        us = [json.loads(l) for l in open(out / "utts" / f"{m}.jsonl", encoding="utf-8")]
        posts = {p: torch.load(out / "diar" / f"nemotron_{p}" / f"{m}.post.pt", weights_only=False)["post80"]
                 .float().numpy() for p in presets}
        x, sr = sf.read(str(out / "session_audio" / "near_mix" / f"{m}.wav"), dtype="float32")
        dur = len(x) / sr
        gs = groups_of(us, "begin_s", "end_s")
        spans = [(min(u["begin_s"] for u in g), max(u["end_s"] for u in g)) for g in gs]
        for gi, g in enumerate(gs):
            a, b = spans[gi]
            talk = [u for u in g if not u["hesitation"]]
            if not talk:
                stats["drop_hesitation_only"] += 1
                continue
            if any(u["record_id"] in fails for u in talk):
                stats["drop_ref_audit"] += 1
                continue
            if b - a > args.max_win + 0.5:
                stats["drop_too_long"] += 1
                continue
            prev_end = max([s[1] for s in spans[:gi]] + [0.0])
            next_beg = min([s[0] for s in spans[gi + 1:]] + [dur])
            ws = max(0.0, a - min(0.3, max(0.0, a - prev_end) / 2))
            we = min(dur, b + min(0.5, max(0.0, next_beg - b) / 2))
            ids = {u["record_id"] for u in g}
            if any(u["record_id"] not in ids and min(we, u["end_s"]) - max(ws, u["begin_s"]) > 0.05 for u in us):
                stats["drop_straddle"] += 1
                continue
            wid = f"{m}_ali_{gi:04d}"
            wav = out / "audio" / "test" / f"{wid}.wav"
            wav.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(wav), x[round(ws * sr): round(we * sr)], sr)
            w = {"id": wid, "meeting": m, "split": "test", "wtype": "ref", "dir": "zh2en", "start_s": round(ws, 3),
                 "dur_ms": round((we - ws) * 1000), "audio": str(wav.resolve()), "strict_review": True,
                 "utts": [{"record_id": u["record_id"], "speaker_id": u["speaker_id"],
                           "begin_s": round(max(0.0, u["begin_s"] - ws), 3), "end_s": round(min(we, u["end_s"]) - ws, 3),
                           "zh": u["zh"] if not u["hesitation"] else "", "en": refs.get(u["record_id"], ""),
                           "ref": refs.get(u["record_id"], ""), "hesitation": u["hesitation"]} for u in g]}
            w["post"] = {p: save_slice(out, p, w, posts[p]) for p in presets}
            W.append(w)
            stats["kept"] += 1
            stats["multi"] += len({u["speaker_id"] for u in talk}) > 1
        print(m, dict(stats), flush=True)
    with open(dst, "x", encoding="utf-8") as fh:
        fh.writelines(json.dumps(w, ensure_ascii=False) + "\n" for w in W)
    stats["hours"] = round(sum(w["dur_ms"] for w in W) / 3.6e6, 2)
    (out / "windows_summary.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
    print(json.dumps(stats))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prep")
    p.add_argument("--raw", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("audit")
    p.add_argument("--out", required=True)
    p = sub.add_parser("windows")
    p.add_argument("--out", required=True)
    p.add_argument("--presets", default="offline,lowlat")
    p.add_argument("--max-win", type=float, default=20.0)
    args = ap.parse_args()
    {"prep": prep, "audit": audit, "windows": windows}[args.cmd](args)


if __name__ == "__main__":
    main()
