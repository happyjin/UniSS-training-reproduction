"""Where the pauses left after hygiene + cleaning come from: each >= 1 s pause on the timeline, its seam type, whether the
source was speaking, and why the next block came late -- the Thinker chose to wait (its draft held nothing beyond what was
committed), or a rule held text the draft already had, or the pause lies inside a block (the Talker). Also the tail.
    python residual.py name=<realsi run dir>=<thinker blocks file> ...
"""
import json, sys, re
from collections import Counter
import numpy as np, soundfile as sf
sys.path.insert(0, "/opt/dlami/nvme/jasonleeeli/projects/UniSS")
from experiments.uniss_qwen3_omni_v1.evaluation.si_fluency_stats import vad_pauses, seam_types, n_units

def src_on(path):
    x, sr = sf.read(path, dtype="float32", always_2d=True); x = x.mean(1); hop = sr // 100; n = len(x) // hop
    db = 10 * np.log10(np.mean(x[: n * hop].reshape(n, hop) ** 2, 1) + 1e-12); loud = db[db > -70]
    on = db > max(np.percentile(loud, 90) - 25, -55)
    i = 0
    while i < n:
        if not on[i]:
            j = i
            while j < n and not on[j]: j += 1
            if j - i < 30 and i > 0 and j < n: on[i:j] = True
            i = j
        else: i += 1
    return on

out = {}
for spec in sys.argv[1:]:
    name, run, thf = spec.split("=")
    TH = {r["id"]: r for r in map(json.loads, open(thf))}
    rows = [json.loads(l) for l in open(run + "/results.jsonl")]
    cause, seam, srcsp, gaps, tails, lastshare, lastdur, held_units = Counter(), Counter(), [], [], [], [], [], []
    n_utt_long = 0
    for r in rows:
        b, tgt = r["blocks"], r["tgt_lang"]; zh = tgt == "cmn"
        if not b: continue
        pauses, first, last, _ = vad_pauses(r["audio_path"])
        types = seam_types(b, tgt, r["translated_text"]) if len(b) > 1 else []
        on = src_on(r["source_audio_path"]); T = r["source_duration_ms"] / 1000
        th = TH[r["id"]]; drafts = th["drafts"]
        committed_at = []   # committed text length (units of the Thinker's raw text) after each block
        acc = 0
        tb = th["blocks"]
        long_here = False
        for p0, p1 in pauses:
            if p1 - p0 < 1.0: continue
            long_here = True
            k = next((k for k in range(len(b) - 1) if p0 <= b[k + 1]["start_ms"] / 1000 + 0.05 and p1 >= (b[k]["start_ms"] + b[k]["dur_ms"]) / 1000 - 0.05), None)
            seg = on[int(min(p0, T) * 100): int(min(p1, T) * 100)]
            srcsp.append(float(seg.mean()) if len(seg) else 0.0)
            if k is None:
                cause["inside block (Talker)"] += 1; continue
            seam[types[k]] += 1
            nb = b[k + 1]
            if nb["start_ms"] > nb["write_ms"] + 20:
                cause["queued (should not happen)"] += 1; continue
            # reads during the pause: did the draft hold text beyond what was committed by then?
            raw_before = "".join(x.get("raw", x["text"]) for x in tb[: k + 1])
            n_before = len(raw_before.replace(" ", "")) if zh else len(raw_before.split())
            extra = []
            for dr in drafts:
                if p0 * 1000 <= dr["t"] < nb["write_ms"]:
                    d = dr["draft"].replace(" ", "") if zh else dr["draft"].split()
                    extra.append(len(d) - n_before)
            mx = max(extra) if extra else 0
            if mx > 0:
                cause["rule held text the draft had"] += 1; held_units.append(mx)
            else:
                cause["Thinker waited (nothing drafted)"] += 1
            gaps.append(nb["write_ms"] / 1000 - p0)
        n_utt_long += long_here
        tails.append(last - T)
        lb = b[-1]
        lastdur.append(lb["dur_ms"] / 1000)
        lastshare.append(n_units(lb["text"], tgt) / max(1, n_units(r["translated_text"], tgt)))
    tot = sum(cause.values())
    out[name] = {"utts": len(rows), "utt_with_long_pause": n_utt_long, "long_pauses": tot,
                 "cause": {k: f"{v} ({v / tot:.0%})" for k, v in cause.most_common()},
                 "seam_of_long_pauses": dict(seam.most_common()),
                 "source_speaking_share_p50": round(float(np.median(srcsp)), 2),
                 "held_units_p50": float(np.median(held_units)) if held_units else None,
                 "tail_p50_p90": [round(float(np.median(tails)), 2), round(float(np.percentile(tails, 90)), 2)],
                 "last_block_dur_p50": round(float(np.median(lastdur)), 2),
                 "last_block_share_of_text_p50": round(float(np.median(lastshare)), 2),
                 "tail_minus_lastdur_p50": round(float(np.median(np.array(tails) - np.array(lastdur))), 2)}
    print(name, json.dumps(out[name], ensure_ascii=False), flush=True)
