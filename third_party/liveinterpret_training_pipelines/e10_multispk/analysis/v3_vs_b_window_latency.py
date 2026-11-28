import json, glob, re, ast, statistics as st
from collections import defaultdict
U = "/opt/dlami/nvme/jasonleeeli/projects/UniSS"
E = U + "/third_party/liveinterpret_training_pipelines/e10_multispk/output"
h = open(E + "/demo_game_v3/page/index.html", encoding="utf-8").read()
i = h.index("const DATA = ") + 13; D = json.loads(h[i:h.index("\n", i)].rstrip().rstrip(";"))
PG = {s["id"]: s for s in D["samples"]}
WIN = {}
for d in ("zh2en", "en2zh"):
    for w in map(json.loads, open(f"{U}/eval_outputs/mist_w3po_v1/data/windows_game_{d}_c2g06_v1/windows_test_online_{d}.jsonl")):
        WIN[w["id"]] = w
def toks(t, zh):
    return re.findall(r"[一-鿿]|[A-Za-z0-9']+", t) if zh else re.findall(r"[A-Za-z0-9']+", t)
# per window: list of (write_ms, spk, text) for each arm
A = defaultdict(lambda: defaultdict(list)); ERR = set()
for d in ("zh2en", "en2zh"):
    for f in glob.glob(f"{E}/demo_game_v3/raw_{d}/part_*.jsonl"):
        for r in map(json.loads, open(f)):
            if r.get("error"): ERR.add(r["id"])
            for g in r.get("segs") or []:
                if g["text"].strip(): A[r["id"]]["V3"].append((g["write_ms"], g["spk"], g["text"].strip(), g.get("dur_ms")))
    for r in map(json.loads, open(f"{U}/eval_outputs/mist_w3po_v1/speech/game_b_{d}__t1c/talker_in/blocks.jsonl")):
        wid, spk = r["id"].split("__")
        for b in r["blocks"]:
            if b["text"].strip(): A[wid]["B"].append((b["write_ms"], spk, b["text"].strip(), None))
def laal(ws, dur, ny, zh):
    d = []
    for wm, _, t, _ in sorted(ws, key=lambda x: x[0]):
        d += [wm] * len(toks(t, zh))
    if not d: return None
    tau = next((k + 1 for k, x in enumerate(d) if x >= dur), len(d))
    return sum(d[k] - k * dur / ny for k in range(tau)) / tau
res = defaultdict(lambda: defaultdict(list))
for wid, w in WIN.items():
    zh = w["dir"] == "en2zh"; dur = w["dur_ms"]; sid = w["meeting"]
    if sid not in PG: continue
    a, b = A[wid]["V3"], A[wid]["B"]
    if not a or not b: res[sid]["missing"].append(wid); continue
    na, nb = sum(len(toks(x[2], zh)) for x in a), sum(len(toks(x[2], zh)) for x in b)
    ny = max(na, nb)
    la, lb = laal(a, dur, ny, zh), laal(b, dur, ny, zh)
    R = res[sid]
    R["laal_V3"].append(la); R["laal_B"].append(lb); R["dur"].append(dur)
    R["first_V3"].append(min(x[0] for x in a)); R["first_B"].append(min(x[0] for x in b))
    R["nwrite_V3"].append(len({x[0] for x in a})); R["nwrite_B"].append(len({x[0] for x in b}))
    R["words_V3"].append(na); R["words_B"].append(nb)
    # V3 per-speaker playback queue delay (same rule as add_demo_arm)
    end = defaultdict(int); qd = []
    for wm, k, t, dm in a:
        stt = max(wm, end[k]); qd.append((stt - wm) * len(toks(t, zh))); end[k] = stt + (dm or 0)
    R["queue_V3_wsum"].append(sum(qd)); 
    # longest stretch without any write while window still open
    for arm, xs in (("V3", a), ("B", b)):
        ts = sorted({0} | {x[0] for x in xs} | {dur})
        R["maxdrought_" + arm].append(max(q - p for p, q in zip(ts, ts[1:])))
print("session  n  LAALtext(V3/B) s   firstwrite(V3/B) s  writes/win(V3/B)  words(V3/B)  V3 queue s/word  maxdrought(V3/B) s  win_dur s")
out = {}
for sid in PG:
    R = res[sid]; m = lambda k: st.mean(R[k])
    q = sum(R["queue_V3_wsum"]) / sum(R["words_V3"]) / 1000
    row = dict(n=len(R["dur"]), laal_V3=m("laal_V3") / 1000, laal_B=m("laal_B") / 1000, first_V3=m("first_V3") / 1000,
               first_B=m("first_B") / 1000, nw_V3=m("nwrite_V3"), nw_B=m("nwrite_B"), words_V3=sum(R["words_V3"]),
               words_B=sum(R["words_B"]), queue_V3=q, drought_V3=m("maxdrought_V3") / 1000, drought_B=m("maxdrought_B") / 1000,
               dur=m("dur") / 1000, missing=R["missing"])
    out[sid] = row
    print(f"{sid}  {row['n']:3d}  {row['laal_V3']:.2f}/{row['laal_B']:.2f}   {row['first_V3']:.2f}/{row['first_B']:.2f}   "
          f"{row['nw_V3']:.1f}/{row['nw_B']:.1f}   {row['words_V3']}/{row['words_B']}   {q:.2f}   "
          f"{row['drought_V3']:.2f}/{row['drought_B']:.2f}   {row['dur']:.1f}  miss={R['missing']}")
print("errors:", sorted(ERR))
json.dump(out, open("/tmp/claude-0/-opt-dlami-nvme-jasonleeeli-projects-UniSS/e9a62284-fa55-4a9e-abbc-6ec6e36ef328/scratchpad/win_lag.json", "w"), indent=1)

# ---- part 2: B queue delay, window-end share, gap attribution
print("\nsession  queue s/word(V3/B)  words@winEnd(V3/B)  gaps>3s: n(V3/B)  sec(V3/B)  V3gap_sec_src_active  V3gap_sec_B_speaking  speech_sec(V3/B)")
def cover(iv):
    iv = sorted(iv); out = []
    for a, b in iv:
        if out and a <= out[-1][1]: out[-1][1] = max(out[-1][1], b)
        else: out.append([a, b])
    return out
def overlap(a, b, iv):
    return sum(max(0, min(b, y) - max(a, x)) for x, y in iv)
for sid, s in PG.items():
    zh = s["dir"] == "en2zh"
    ws = [w for w in WIN.values() if w["meeting"] == sid]
    q = {"V3": [0, 0], "B": [0, 0]}; endw = {"V3": [0, 0], "B": [0, 0]}
    for arm in ("V3", "B"):
        tg = s["arms"][arm]["tgt"]; pool = defaultdict(list)
        for x in tg: pool[x["text"]].append(x["t"])
        for w in ws:
            for wm, k, t, _ in A[w["id"]][arm]:
                n = len(toks(t, zh)); endw[arm][1] += n; endw[arm][0] += n if wm >= w["dur_ms"] - 640 else 0
                c = [tt for tt in pool.get(t, []) if tt >= w["start_s"] + wm / 1000 - 0.05]
                if c:
                    tt = min(c); pool[t].remove(tt); q[arm][0] += (tt - w["start_s"] - wm / 1000) * n; q[arm][1] += n
    src = s["src"] if isinstance(s["src"], list) else ast.literal_eval(s["src"])
    S = cover([(x["t"], x["e"]) for x in src])
    T0, T1 = s.get("t0", 0) * 0, 300.0
    gi = {}
    for arm in ("V3", "B"):
        cv = cover([(x["t"], x["e"]) for x in s["arms"][arm]["tgt"]]); prev = 0.0; g = []
        for a, b in cv:
            if a > prev + 3 and a < T1: g.append((prev, min(a, T1)))
            prev = max(prev, b)
        if prev + 3 < T1: g.append((prev, T1))
        gi[arm] = (g, cv)
    gv, cvv = gi["V3"]; gb, cvb = gi["B"]
    sec = lambda g: sum(b - a for a, b in g)
    print(f"{sid}   {q['V3'][0]/max(1,q['V3'][1]):.2f}/{q['B'][0]/max(1,q['B'][1]):.2f}   "
          f"{endw['V3'][0]/endw['V3'][1]:.2f}/{endw['B'][0]/endw['B'][1]:.2f}   {len(gv)}/{len(gb)}   {sec(gv):.0f}/{sec(gb):.0f}   "
          f"{sum(overlap(a,b,S) for a,b in gv):.0f}   {sum(overlap(a,b,cvb) for a,b in gv):.0f}   "
          f"{sum(min(b,T1)-a for a,b in cvv if a<T1):.0f}/{sum(min(b,T1)-a for a,b in cvb if a<T1):.0f}")
