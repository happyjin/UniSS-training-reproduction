"""Game (HOK esports commentary) listening demo, step 1 (reports/mist_w3po_v1/game_multispk_demo_plan.zh-CN.md §4):
choose one 5-min continuous excerpt per video from the ORIGINAL part audio (source/<video>_partNN.wav), using only the
pipeline metadata (records/*.json, diarization/*.json) -- never any system output.

Per (video, part) every packed clip is mapped back to the part timeline with its source_mapping:
  speech    records' constituent_speech_intervals (part time)
  diar      diarization segments (packed time -> part time), labels "<segment>:<k>" (clip-local)
  turns     records' turns with source_text / target_text (packed start/end -> part time) = the silver references
Score of a window [t, t+DUR] (10 s steps): speech coverage x number of speaker changes between consecutive diar segments;
eligible only with speech coverage >= 0.70 and silver-turn coverage of the speech >= 0.80. The best window per video;
then the top N videos by score. CPU only.

    python -m experiments.mist_w3po_v1.data.game_select --dir zh2en|en2zh --n 5 --dur 300 --out <new json>
Refuses an existing output.
"""
from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
META = REPO / "data/game_esports_si_v1/meta"


def p2s(t: float, mp: list[dict]) -> float:
    """packed clip time -> part time"""
    for m in mp:
        if m["packed_start"] - 1e-6 <= t <= m["packed_end"] + 1e-6:
            return m["source_start"] + (t - m["packed_start"])
    return mp[-1]["source_end"] if t > mp[-1]["packed_end"] else mp[0]["source_start"]


def load_video(vdir: Path) -> dict[int, dict]:
    parts: dict[int, dict] = defaultdict(lambda: {"speech": [], "diar": [], "turns": []})
    for f in sorted(glob.glob(str(vdir / "records" / "*.json"))):
        r = json.load(open(f))
        mp = r.get("source_mapping") or []
        if not mp:
            continue
        P = parts[int(r["source_part"])]
        P["speech"] += [tuple(x) for x in r.get("constituent_speech_intervals") or []]
        for t in r.get("turns") or []:
            src, tgt = (t.get("source_text") or "").strip(), (t.get("target_text") or "").strip()
            if not src or not tgt or t.get("start") is None:
                continue
            P["turns"].append({"seg": r["segment_id"], "spk": f"{r['segment_id']}:{t.get('speaker_id')}",
                               "b": p2s(float(t["start"]), mp), "e": p2s(max(float(t["start"]), float(t["end"]) - 1e-3), mp),
                               "src": src, "tgt": tgt})
        d = vdir / "diarization" / f"{r['segment_id']}.json"
        if d.exists():
            for s in json.load(open(d)).get("segments") or []:
                P["diar"].append((p2s(s["start"], mp), p2s(max(s["start"], s["end"] - 1e-3), mp),
                                  f"{r['segment_id']}:{s['speaker']}"))
    return parts


def cover(iv: list[tuple], a: float, b: float) -> float:
    xs = sorted((max(a, s), min(b, e)) for s, e, *_ in iv if e > a and s < b)
    tot, cur_s, cur_e = 0.0, None, None
    for s, e in xs:
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                tot += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        tot += cur_e - cur_s
    return tot


def best_window(P: dict, dur: float) -> dict | None:
    if not P["speech"]:
        return None
    end = max(e for _, e in P["speech"])
    best = None
    t = 0.0
    while t + dur <= end + 1e-6:
        sp = cover(P["speech"], t, t + dur) / dur
        tc = cover([(x["b"], x["e"]) for x in P["turns"]], t, t + dur) / max(1e-6, sp * dur)
        if sp >= 0.70 and tc >= 0.80:
            d = sorted(x for x in P["diar"] if x[0] >= t and x[1] <= t + dur)
            ch = sum(1 for a, b in zip(d, d[1:]) if a[2] != b[2] and a[2].split(":")[0] == b[2].split(":")[0])
            sc = sp * ch
            if best is None or sc > best["score"]:
                best = {"t0": round(t, 2), "speech_cov": round(sp, 3), "turn_cov": round(min(tc, 1.0), 3),
                        "spk_changes": ch, "score": round(sc, 2),
                        "max_spk_in_clip": max([len({x[2] for x in d if x[2].startswith(s + ':')})
                                                for s in {x[2].split(':')[0] for x in d}] or [0])}
        t += 10.0
    return best


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True, choices=["zh2en", "en2zh"])
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--dur", type=float, default=300.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    cands = []
    for vdir in sorted((META / a.dir).iterdir()):
        parts = load_video(vdir)
        bw = None
        for p, P in parts.items():
            w = best_window(P, a.dur)
            if w and (bw is None or w["score"] > bw["score"]):
                bw = dict(w, part=p, n_turns=sum(1 for x in P["turns"] if x["b"] < w["t0"] + a.dur and x["e"] > w["t0"]))
        title = ""
        sm = vdir / "source_manifest.json"
        if sm.exists():
            try:
                title = json.loads(json.load(open(sm))["records"][0]["source_metadata_json"]).get("title") or ""
            except Exception:
                title = ""
        print(vdir.name, bw, title[:60], flush=True)
        if bw:
            cands.append(dict(bw, video=vdir.name, title=title))
    cands.sort(key=lambda x: -x["score"])
    sel = cands[: a.n]
    with open(out, "x", encoding="utf-8") as fh:
        json.dump({"dir": a.dir, "dur": a.dur, "rule": "max speech_cov x speaker changes; speech>=0.70, silver turns>=0.80",
                   "selected": sel, "candidates": cands}, fh, ensure_ascii=False, indent=1)
    print("selected", [(x["video"], x["part"], x["t0"], x["score"]) for x in sel])


if __name__ == "__main__":
    main()
