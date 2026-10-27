"""Sudden pauses on the continuous (talk) timeline: every silence >= 0.5 s between the first and last sound of a talk,
classified by why it happened, where it falls in the text, whether the source speaker was talking, and whether pauses
come in clusters. One line per system.
    python abrupt.py name=<run dir>=<items: realsi:en2zh|realsi:zh2en|acl>=<thinker blocks file>=<s1|plain> ...
Why (in this order):
  inside block    the silence lies inside one block: the Talker itself paused
  new sentence    between the last block of one segment and the first of the next: the next sentence had not yet been
                  committed (split by whether the source was silent too: the speaker's own gap vs the Thinker's wait)
  rule held       inside a segment, the Thinker's drafts during the silence held text beyond what was committed
                  (hygiene kept it back)
  Thinker waited  inside a segment, the drafts held nothing more: the Thinker chose to wait
Where: the seam type (S sentence / C clause / P phrase / P-fw after a function word / W inside a word), as
si_fluency_stats. Abrupt = mid-clause seam (P, P-fw, W) or inside a block, while the source is speaking.
Cluster = 3 or more pauses >= 0.5 s within 10 s."""
import json, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np, soundfile as sf
ROOT = Path("/opt/dlami/nvme/jasonleeeli/projects/UniSS"); sys.path.insert(0, str(ROOT))
from experiments.uniss_qwen3_omni_v1.evaluation.si_talk_timeline import load, schedule, segments, SR
from experiments.uniss_qwen3_omni_v1.evaluation.si_fluency_stats import seam_types
from experiments.uniss_qwen3_omni_v1.evaluation.si_metrics import laal, units
import re

def vad(y, thr_rel=30):
    hop = SR // 100; n = len(y) // hop
    db = 10 * np.log10(np.mean(y[: n * hop].reshape(n, hop) ** 2, 1) + 1e-12)
    loud = db[db > -70]
    return db > max(np.percentile(loud, 90) - thr_rel, -60)

def src_vad(path):
    x, sr = sf.read(path, dtype="float32", always_2d=True); x = x.mean(1)
    on = vad(x, 25); n = len(on); i = 0
    while i < n:
        if not on[i]:
            j = i
            while j < n and not on[j]: j += 1
            if j - i < 30 and i > 0 and j < n: on[i:j] = True
            i = j
        else: i += 1
    return on

def talk_wav(items, talk):
    if items == "acl":
        return next(json.loads(l)["wav"] for l in open(ROOT / "data/acl6060_dev_v1/talks.jsonl") if json.loads(l)["talk"] == talk)
    return str(ROOT / f"data/raw/realsi/RealSI_data/data/{items.split(':')[1]}/wav/{talk}.wav")

out = {}
for spec in sys.argv[1:]:
    name, run, items, thf, mode = spec.split("=")
    s1 = mode.startswith("s1")   # mode: plain | s1[b<buffer s>][q]  (q: squeeze inner silences)
    mb = re.search(r"b([\d.]+)", mode); buf = float(mb.group(1)) if mb else 0.0; sq = mode.endswith("q")
    rows, meta = load(Path(run), items)
    TH = {r["id"]: r for r in map(json.loads, open(thf))}
    tgt = rows[0]["tgt_lang"]; zh = tgt == "cmn"
    talks = defaultdict(list)
    for r in rows:
        if r.get("blocks") and r["id"] in meta: talks[meta[r["id"]][0]].append(r)
    why, where, src_sp, runs_before, cl, abrupt, total_sound = Counter(), Counter(), [], [], 0, 0, 0.0
    n_all = n1 = 0; per_min = []; laals = []; waits = []
    for talk, rs in talks.items():
        rs.sort(key=lambda r: meta[r["id"]][1]); segs, k0, rowmap = [], 0, {}
        for r in rs:
            segs.append(segments(r, meta[r["id"]][1], k0, trim=s1, squeeze_inner=sq)); k0 += len(r["blocks"]); rowmap[r["id"]] = r
        pl = schedule(segs, s1, 1.10, 1.0, 1.05, buf, 1.5)
        for r in rs:
            mine = sorted((x for x in pl if x["seg"] == r["id"]), key=lambda x: x["k"])
            ds = [x["start"] - x["offset"] for x in mine for _ in units(x["text"], tgt)]
            if ds: laals.append(laal(ds, r["source_duration_ms"], len(units(r.get("translation_ref") or "", tgt))))
        waits.extend((x["start"] - x["commit"]) / 1000 for x in pl)
        n = int(max(b["start"] + b["play_dur"] for b in pl) * 16) + SR
        y = np.zeros(n, dtype=np.float32)
        for b in pl:
            a = int(b["start"] * 16); y[a: a + len(b["audio"])] += b["audio"][: n - a]
        on = vad(y); idx = np.flatnonzero(on); f0, f1 = idx[0], idx[-1]
        sound = on[f0:f1 + 1].sum() / 100; total_sound += sound
        sv = src_vad(talk_wav(items, talk))
        pauses, i = [], f0
        while i <= f1:
            if not on[i]:
                j = i
                while j <= f1 and not on[j]: j += 1
                if (j - i) / 100 >= 0.5: pauses.append((i / 100, j / 100))
                i = j
            else: i += 1
        # seam types per segment
        types = {}
        for r in rs:
            b = r["blocks"]
            types[r["id"]] = seam_types(b, tgt, r["translated_text"]) if len(b) > 1 else []
        spans = [(b["start"] / 1000, (b["start"] + b["play_dur"]) / 1000, b) for b in pl]
        last_end = None; ptimes = []
        for p0, p1 in pauses:
            n_all += 1; n1 += (p1 - p0) >= 1.0; ptimes.append(p0)
            seg = sv[int(p0 * 100): int(p1 * 100)]; speaking = bool(len(seg)) and seg.mean() > 0.5
            src_sp.append(speaking)
            prev = [s for s in spans if s[0] < p0 + 0.05]
            nxt = [s for s in spans if s[0] >= p0 - 0.05 and s[0] <= p1 + 0.1]
            pb = max(prev, key=lambda s: s[0]) if prev else None
            nb = min(nxt, key=lambda s: s[0]) if nxt else None
            # speech run before this pause
            k = int(p0 * 100) - 1; m = k
            while m > f0 and (on[m] or (m - 1 > f0 and on[m - 1])) : m -= 1
            runs_before.append((k - m) / 100)
            if pb is not None and pb[1] > p1 - 0.05:
                c, w = "inside block (Talker)", "in-block"
            elif pb is None or nb is None:
                c, w = "other", "?"
            elif pb[2]["seg"] != nb[2]["seg"]:
                c, w = ("new sentence: source also silent" if not speaking else "new sentence: source speaking (Thinker not started)"), "S-between"
            else:
                sid = pb[2]["seg"]; kk = [x["k"] for x in pl if x["seg"] == sid]; k_local = kk.index(pb[2]["k"])
                w = types[sid][k_local] if k_local < len(types[sid]) else "?"
                r = rowmap[sid]; th = TH.get(sid); off = meta[sid][1]
                raw_before = "".join(x.get("raw", x["text"]) for x in th["blocks"][: k_local + 1]) if th else ""
                n_b = len(raw_before.replace(" ", "")) if zh else len(raw_before.split())
                nb_write = off + nb[2]["commit"] - off
                ex = [ (len(d["draft"].replace(" ", "")) if zh else len(d["draft"].split())) - n_b
                       for d in (th["drafts"] if th else []) if p0 * 1000 - off <= d["t"] < nb[2]["commit"] - off ]
                c = "rule held text the draft had" if ex and max(ex) > 0 else "Thinker waited (nothing drafted)"
            why[c] += 1; where[w] += 1
            if speaking and (w in ("P", "P-fw", "W", "in-block")): abrupt += 1
        pt = np.array(ptimes)
        cl += sum(1 for t in pt if np.sum((pt >= t - 5) & (pt <= t + 5)) >= 3)
    rb = np.array(runs_before)
    out[name] = {"pauses_ge0.5": n_all, "per_min_speech": round(60 * n_all / total_sound, 2), "share_ge1s": round(n1 / n_all, 3),
                 "why": {k: f"{v} ({v / n_all:.0%})" for k, v in why.most_common()},
                 "where": {k: f"{v / n_all:.0%}" for k, v in where.most_common()},
                 "source_speaking": round(float(np.mean(src_sp)), 3),
                 "abrupt_share (mid-clause or in-block, source speaking)": round(abrupt / n_all, 3),
                 "abrupt_per_min_speech": round(60 * abrupt / total_sound, 2),
                 "in_clusters (>=3 within 10 s)": round(cl / n_all, 3),
                 "speech_run_before_p50_s": round(float(np.median(rb)), 2), "run_before_lt1s": round(float(np.mean(rb < 1.0)), 3),
                 "continuous_laal_s": round(float(np.mean(laals)) / 1000, 3), "wait_p90_s": round(float(np.percentile(waits, 90)), 2)}
    print(name, json.dumps(out[name], ensure_ascii=False), flush=True)
