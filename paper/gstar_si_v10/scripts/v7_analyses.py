"""Paper v7 analyses on existing reference-window outputs (prereg_v7 §5; v6 review M08, M12, M15, T02, T09).

  count      M12: speaker counts on multi-speaker windows, per system: exact / under / over, and "wrong speakers"
             (right count, but a reference speaker gets no text after the session matching); under-counts split by
             the duration of the window's least-heard speaker (<= 0.5, 0.5-1, 1-2, > 2 s of speech) and by whether
             that speaker overlaps another
  zerolen    M15 / T09 / T02: output segments with start == end (both time tokens equal); the share at the 20 s
             token boundary; the share whose best-overlapping reference utterance lasts <= 1 s; then the fix
             beta >= alpha + 1 (end moved one 80 ms bin later, or start one bin earlier at the window end) written to
             eval_outputs/.../gstar_v7_20261101/zlfix/<dir>_<name>/hyp.jsonl and scored with gstar_score
  funnel     M08: per test set, meeting time -> reference-window audio -> kept speech -> kept overlapped speech ->
             kept <= 1 s utterances (hours and shares), from v6's filter analysis plus the meeting lengths

Writes generated/v7_extras.json (keys per analysis; an existing key is kept unless --redo).

    python scripts/v7_analyses.py count zerolen funnel        (from the paper folder)
"""
from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from systems import STREAMING, hyp_dir, windows  # noqa: E402
from v6_analyses import SEP, load, session_map, system  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
GEN = Path(__file__).resolve().parents[1] / "generated"
OUT = ROOT / "eval_outputs/uniss_qwen3_omni_v1/gstar_v7_20261101"
PY = "/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval/bin/python"
SYS = {"en2zh": ["O2", "C-ft2", "O2-nosd", "S2dl_d1", "O2_LA2_d1"], "zh2en": ["O2", "C-ft2", "O2-nosd", "S2dl_d1", "O2_LA2_d1"],
       "ali": ["O2", "C-ft2", "S2dl_d1", "O2_LA2_d1"]}


def talk(w):
    return [u for u in w["utts"] if not u["hesitation"]]


def count():
    res = {}
    for dr, names in SYS.items():
        for name in names:
            W, H = system(dr, name)
            if W is None:
                continue
            sm = session_map(dr, W, H)
            c = Counter()
            under = defaultdict(Counter)
            for w in W:
                spk = defaultdict(float)
                for u in talk(w):
                    spk[u["speaker_id"]] += u["end_s"] - u["begin_s"]
                n_ref = len(spk)
                if n_ref < 2:
                    continue
                txt = defaultdict(str)
                for s in H[w["id"]]["segments"]:
                    if s["text"].strip():
                        txt[s["spk"]] += s["text"]
                n_hyp = len(txt)
                m = sm.get(w["meeting"], {})
                got = {m.get(k) for k in txt}
                least = min(spk, key=spk.get)
                d = spk[least]
                band = "le0.5" if d <= 0.5 else "0.5-1" if d <= 1 else "1-2" if d <= 2 else "gt2"
                spans = [(u["begin_s"], u["end_s"]) for u in talk(w) if u["speaker_id"] == least]
                others = [(u["begin_s"], u["end_s"]) for u in talk(w) if u["speaker_id"] != least]
                ov = any(min(a[1], b[1]) > max(a[0], b[0]) for a in spans for b in others)
                c["windows"] += 1
                if n_hyp == n_ref:
                    c["exact"] += 1
                    if not all(k in got for k in spk):
                        c["wrong_speakers"] += 1
                elif n_hyp < n_ref:
                    c["under"] += 1
                    under[band]["under"] += 1
                    under["overlapped" if ov else "not_overlapped"]["under"] += 1
                    under[band]["least_missing"] += least not in got
                else:
                    c["over"] += 1
                under[band]["windows"] += 1
                under["overlapped" if ov else "not_overlapped"]["windows"] += 1
            n = c["windows"]
            res[f"{dr}:{name}"] = {"multi_windows": n, **{k: round(c[k] / n, 4) for k in ("exact", "under", "over", "wrong_speakers")},
                                   "under_by_least_speaker": {b: {"windows": v["windows"], "under": round(v["under"] / max(v["windows"], 1), 4),
                                                                  "least_speaker_missing": round(v["least_missing"] / max(v["windows"], 1), 4)}
                                                              for b, v in sorted(under.items())}}
            print(dr, name, res[f"{dr}:{name}"]["exact"], res[f"{dr}:{name}"]["under"], flush=True)
    return res


def zerolen():
    res = {}
    for dr, names in (("en2zh", ["O2", "S2dl_d1", "C-ft2"]), ("zh2en", ["O2", "S2dl_d1", "C-ft2"]), ("ali", ["O2", "S2dl_d1", "C-ft2"])):
        for name in names:
            W, H = system(dr, name)
            if W is None:
                continue
            WD = {w["id"]: w for w in W}
            n = z = zb = zs = 0
            fixed = []
            for wid, h in H.items():
                if wid not in WD:
                    continue
                w = WD[wid]
                dur = w["dur_ms"] / 1000
                segs = []
                for s in h["segments"]:
                    s = dict(s)
                    n += 1
                    if s["end_s"] <= s["start_s"]:
                        z += 1
                        zb += s["start_s"] >= min(dur, 20.0) - 0.08
                        best, bo = None, 0.0
                        for u in talk(w):
                            u0, u1 = u["begin_s"], u["end_s"]   # window clock
                            o = min(u1, s["start_s"] + 0.5) - max(u0, s["start_s"] - 0.5)
                            if o > bo:
                                best, bo = u, o
                        zs += best is not None and best["end_s"] - best["begin_s"] <= 1.0
                        if s["start_s"] + 0.08 <= dur:
                            s["end_s"] = round(s["start_s"] + 0.08, 2)
                        else:
                            s["start_s"] = round(max(0.0, s["end_s"] - 0.08), 2)
                    segs.append(s)
                fixed.append({**h, "segments": segs})
            d = OUT / "zlfix" / f"{dr}_{name}"
            d.mkdir(parents=True, exist_ok=True)
            hp = d / "hyp.jsonl"
            if not hp.exists():
                with open(hp, "x", encoding="utf-8") as fh:
                    for h in fixed:
                        fh.write(json.dumps(h, ensure_ascii=False) + "\n")
            sp = d / "score.json"
            if not sp.exists():
                subprocess.run(f"PYTHONPATH=. {PY} -m experiments.uniss_qwen3_omni_v1.evaluation.gstar_score --windows "
                               f"{windows(dr, name)} --hyp {hp} --post-key {'lowlat' if name in STREAMING else 'offline'} --out {sp} > {d}/score.log 2>&1",
                               shell=True, cwd=ROOT, check=True)
            before = json.load(open(hyp_dir(dr, name) / "score.json"))["all"]
            after = json.load(open(sp))["all"]
            keys = ("cpBLEU_session", "cpBLEU_window", "count_exact")
            res[f"{dr}:{name}"] = {"segments": n, "zero_len": round(z / max(n, 1), 4),
                                   "zero_len_at_20s_end": round(zb / max(z, 1), 4),
                                   "zero_len_short_ref_le1s": round(zs / max(z, 1), 4),
                                   "before": {k: before.get(k) for k in keys} | {"DER": before["DER_output"]["DER"]},
                                   "after_fix": {k: after.get(k) for k in keys} | {"DER": after["DER_output"]["DER"]}}
            print(dr, name, res[f"{dr}:{name}"], flush=True)
    return res


def funnel():
    v6 = json.load(open(GEN / "v6_extras.json"))["filter"]
    res = {}
    for dr in ("en2zh", "zh2en", "ali"):
        W = load(windows(dr, "O2"))
        win_h = sum(w["dur_ms"] for w in W) / 3.6e6
        meet = {}
        for w in W:
            meet[w["meeting"]] = max(meet.get(w["meeting"], 0.0), w["start_s"] + w["dur_ms"] / 1000)
        f = v6.get(dr, {})
        res[dr] = {"meetings": len(meet), "meeting_hours_upto_last_window": round(sum(meet.values()) / 3600, 2),
                   "window_audio_hours": round(win_h, 2), "windows": len(W),
                   "speech_hours": f.get("all", {}).get("hours"), "kept_speech": f.get("all", {}).get("kept_dur"),
                   "overlap_hours": f.get("overlapped", {}).get("hours"), "kept_overlap": f.get("overlapped", {}).get("kept_dur"),
                   "short_hours": f.get("short_le1s", {}).get("hours"), "kept_short": f.get("short_le1s", {}).get("kept_dur")}
        print(dr, res[dr], flush=True)
    return res


def main() -> None:
    redo = "--redo" in sys.argv
    todo = [a for a in sys.argv[1:] if not a.startswith("--")]
    p = GEN / "v7_extras.json"
    ex = json.load(open(p)) if p.exists() else {}
    for k in todo:
        if k in ex and not redo:
            print("kept", k)
            continue
        ex[k] = {"count": count, "zerolen": zerolen, "funnel": funnel}[k]()
        with open(p.with_suffix(".tmp"), "w") as fh:
            json.dump(ex, fh, indent=1, ensure_ascii=False)
        p.with_suffix(".tmp").replace(p)


if __name__ == "__main__":
    main()
