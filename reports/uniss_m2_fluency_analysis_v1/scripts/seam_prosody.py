"""Prosody at block seams: F0 / energy jump across back-to-back seams vs inside blocks, and the F0 slope at the end of
blocks that end mid-sentence vs at a sentence end.  name=run_dir ... -> json on stdout."""
import json, re, sys
from multiprocessing import Pool
import numpy as np, soundfile as sf, librosa

SENT = set("。？！.?!")
W = 0.3   # seconds of voiced speech each side

def f0_db(x, sr):
    f0, vf, _ = librosa.pyin(x, fmin=60, fmax=500, sr=sr, frame_length=1024, hop_length=160)
    rms = librosa.feature.rms(y=x, frame_length=1024, hop_length=160)[0]
    n = min(len(f0), len(rms))
    st = 12 * np.log2(np.where(vf[:n] & ~np.isnan(f0[:n]), f0[:n], np.nan) / 100.0)
    return st, 20 * np.log10(rms[:n] + 1e-6)

def one(r):
    x, sr = sf.read(r["audio_path"], dtype="float32", always_2d=True)
    x = x.mean(1)
    out = {"seam": [], "inner": [], "end_slope": []}
    b = r["blocks"]
    def seg(a, c):
        a, c = max(0, int(a * sr)), min(len(x), int(c * sr))
        return x[a:c]
    for k, blk in enumerate(b):
        s, e = blk["start_ms"] / 1000, (blk["start_ms"] + blk["dur_ms"]) / 1000
        st, db = f0_db(seg(s, e), sr) if e - s > 0.4 else (None, None)
        if st is None:
            continue
        v = np.flatnonzero(~np.isnan(st))
        if len(v) < 10:
            continue
        # end slope over the last W s of voiced frames (semitones / s)
        tail = v[v >= v[-1] - int(W * 100)]
        if len(tail) >= 8:
            sl = np.polyfit(tail / 100, st[tail], 1)[0]
            txt = blk["text"].rstrip()
            kind = "final" if blk["final"] else ("sent" if txt and txt[-1] in SENT else "mid")
            out["end_slope"].append((kind, float(sl)))
        # inside the block: around its middle, if long enough
        if e - s >= 1.6:
            m = int((len(st)) / 2)
            a1 = v[(v < m) & (v >= m - int(W * 100))]; a2 = v[(v >= m) & (v < m + int(W * 100))]
            if len(a1) >= 5 and len(a2) >= 5:
                out["inner"].append((float(abs(np.median(st[a1]) - np.median(st[a2]))), float(abs(np.median(db[a1]) - np.median(db[a2])))))
        # seam to the next block if it follows within 0.15 s
        if k + 1 < len(b):
            nb = b[k + 1]
            ns, ne = nb["start_ms"] / 1000, (nb["start_ms"] + nb["dur_ms"]) / 1000
            if ns - e < 0.15 and ne - ns > 0.4:
                st2, db2 = f0_db(seg(ns, ne), sr)
                v2 = np.flatnonzero(~np.isnan(st2))
                a1 = v[v >= v[-1] - int(W * 100)]; a2 = v2[v2 <= (v2[0] + int(W * 100))] if len(v2) else v2
                if len(a1) >= 5 and len(a2) >= 5:
                    txt = blk["text"].rstrip()
                    kind = "sent" if txt and txt[-1] in SENT else "mid"
                    out["seam"].append((kind, float(abs(np.median(st[a1]) - np.median(st2[a2]))), float(abs(np.median(db[a1]) - np.median(db2[a2])))))
    return out

if __name__ == "__main__":
    res = {}
    for spec in sys.argv[1:]:
        name, d = spec.split("=", 1)
        rows = [json.loads(l) for l in open(d + "/results.jsonl")]
        with Pool(48) as p:
            outs = p.map(one, rows, chunksize=4)
        seam = [s for o in outs for s in o["seam"]]; inner = [s for o in outs for s in o["inner"]]; es = [s for o in outs for s in o["end_slope"]]
        q = lambda a: [round(float(np.median(a)), 2), round(float(np.percentile(a, 90)), 2)] if len(a) else None
        res[name] = {
            "n_seam_mid": sum(1 for s in seam if s[0] == "mid"), "n_inner": len(inner),
            "dF0_st_mid_seam_p50_p90": q([s[1] for s in seam if s[0] == "mid"]), "dF0_st_inner_p50_p90": q([s[0] for s in inner]),
            "dE_db_mid_seam_p50_p90": q([s[2] for s in seam if s[0] == "mid"]), "dE_db_inner_p50_p90": q([s[1] for s in inner]),
            "jump_gt3st_mid_seam": round(float(np.mean([s[1] > 3 for s in seam if s[0] == "mid"])), 3) if seam else None,
            "jump_gt3st_inner": round(float(np.mean([s[0] > 3 for s in inner])), 3) if inner else None,
            "end_slope_st_per_s_p50": {k: round(float(np.median([s[1] for s in es if s[0] == k])), 2) for k in ("mid", "sent", "final")},
            "end_falling_lt_-6": {k: round(float(np.mean([s[1] < -6 for s in es if s[0] == k])), 3) for k in ("mid", "sent", "final")},
            "n_end": {k: sum(1 for s in es if s[0] == k) for k in ("mid", "sent", "final")},
        }
        print(name, json.dumps(res[name]), flush=True)
    print(json.dumps(res, indent=1))
