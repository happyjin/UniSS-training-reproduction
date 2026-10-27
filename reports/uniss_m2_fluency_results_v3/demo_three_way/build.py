"""Continuous-talk listening page: M2 vs N1 + S1 on the same talk excerpt (left: source, right: interpretation)."""
import json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np, soundfile as sf
ROOT = Path("/opt/dlami/nvme/jasonleeeli/projects/UniSS"); sys.path.insert(0, str(ROOT))
from experiments.uniss_qwen3_omni_v1.evaluation.si_talk_timeline import load, schedule, segments, SR
PAGE = Path(__file__).resolve().parent
WIN = 100.0
E = ROOT / "eval_outputs/uniss_qwen3_omni_v1/si_t1c_20261021T181440Z/full_iter0001500"
A = ROOT / "eval_outputs/uniss_qwen3_omni_v1"
SPECS = [("en2zh", "realsi:en2zh", E / "system_swb-q-b1_m0.5/realsi_en2zh", E / "system_flu-hc3-b0p5_m0.5/realsi_en2zh", E / "system_flu7-sq-b1_m0.5/realsi_en2zh"),
         ("zh2en", "realsi:zh2en", E / "system_swb-q-b0-zh_m1/realsi_zh2en", E / "system_flu-hc4-bm1-zh_m1/realsi_zh2en", E / "system_flu7-sq-zh_m1/realsi_zh2en")]

def talk_wav(items, talk):
    if items == "acl":
        return next(json.loads(l)["wav"] for l in open(ROOT / "data/acl6060_dev_v1/talks.jsonl") if json.loads(l)["talk"] == talk)
    d = items.split(":")[1]
    return str(ROOT / f"data/raw/realsi/RealSI_data/data/{d}/wav/{talk}.wav")

def play(run, items, mode):
    s1 = mode != "plain"; buf = 1.2 if mode == "v3" else 0.0
    rows, meta = load(run, items)
    talks = defaultdict(list)
    for r in rows:
        if r.get("blocks") and r["id"] in meta: talks[meta[r["id"]][0]].append(r)
    out = {}
    for t, rs in talks.items():
        rs.sort(key=lambda r: meta[r["id"]][1]); segs, k0 = [], 0
        for r in rs:
            segs.append(segments(r, meta[r["id"]][1], k0, trim=s1, squeeze_inner=(mode == "v3"))); k0 += len(r["blocks"])
        out[t] = (schedule(segs, s1, 1.10, 1.0, 1.05, buf, 1.5), min(meta[r["id"]][1] for r in rs))
    return out

def norm(y):
    n = max(1, len(y) // (SR // 10)); loud = float(np.percentile(np.sqrt(np.mean(y[: n * (SR // 10)].reshape(n, -1) ** 2, axis=1)), 99)) or 1e-4
    g = min(0.12 / max(loud, 1e-4), 3.0 / (float(np.abs(y).max()) or 1.0)); return 0.95 * np.tanh(y * g / 0.95)

man = {"items": []}
for d, items, m2_run, n1_run, v3_run in SPECS:
    P = {"m2": play(m2_run, items, "plain"), "v2": play(n1_run, items, "v2"), "v3": play(v3_run, items, "v3")}
    # the worst talk for M2 (largest wait within the first WIN s) and a typical one
    score = {}
    for t, (pl, t0) in P["m2"].items():
        w = [(b["start"] - b["commit"]) / 1000 for b in pl if b["commit"] / 1000 < t0 / 1000 + WIN]
        score[t] = max(w) if w else 0
    order = sorted(score, key=score.get)
    picks = [order[-1], order[len(order) // 2], order[len(order) // 4]]
    for t in picks:
        x, sr = sf.read(talk_wav(items, t), dtype="float32", always_2d=True); x = x.mean(1)
        t0 = P["m2"][t][1] / 1000; a0 = int(t0 * SR); n = int((WIN + 8) * SR)
        src = norm(np.pad(x[a0: a0 + n], (0, max(0, n - len(x[a0: a0 + n])))))
        entry = {"dir": d, "talk": t, "start_s": round(t0, 1), "win": WIN, "kind": "M2 积压最重的一场" if t == order[-1] else ("典型的一场" if t == order[len(order) // 2] else "积压较轻的一场"), "sys": {}}
        for k in ("m2", "v2", "v3"):
            pl = P[k][t][0]; y = np.zeros(n, dtype=np.float32); lags = []; blocks = []
            for b in pl:
                s = b["start"] / 1000 - t0
                if b["commit"] / 1000 - t0 > WIN: continue
                if s >= WIN + 8: continue
                i = int(s * SR); seg = b["audio"][: max(0, n - i)]; y[i: i + len(seg)] += seg
                lags.append([round(b["commit"] / 1000 - t0, 2), round((b["start"] - b["commit"]) / 1000, 2)])
                blocks.append([round(s, 2), round(b["play_dur"] / 1000, 2), round(b["rate"], 2)])
            st = np.stack([src, norm(y)], 1)
            f = PAGE / f"audio/{d}_{t}_{k}.ogg"; f.parent.mkdir(parents=True, exist_ok=True)
            with sf.SoundFile(str(f), "x", SR, 2, format="OGG", subtype="VORBIS") as fh:
                for i in range(0, len(st), 32768): fh.write(st[i: i + 32768])
            lv = [l[1] for l in lags]
            entry["sys"][k] = {"audio": f"audio/{d}_{t}_{k}.ogg", "lags": lags, "blocks": blocks,
                               "lag_mean": round(float(np.mean(lv)), 2), "lag_max": round(float(np.max(lv)), 2),
                               "end_after": round(max(b[0] + b[1] for b in blocks) - WIN, 1)}
        man["items"].append(entry); print(d, t, entry["kind"], {k: (v["lag_mean"], v["lag_max"]) for k, v in entry["sys"].items()}, flush=True)
(PAGE / "manifest.json").write_text(json.dumps(man, ensure_ascii=False))
