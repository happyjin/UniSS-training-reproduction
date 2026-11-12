"""Check that every number fill_paper_v13.py put into paper/bayes_simul_s2st_v13_baselines matches its source JSON.
Independent of fill_paper_v13.py: the tier cells of tab_realsi_main are checked against the executors' own tiers.json
(by_D1), the seed-0 tiers of tab_tiers / tab_tiers_main are recomputed from the per-point JSONs, and the build log is
checked for undefined references / citations. Exit code 1 on any mismatch.

    /opt/dlami/nvme/jasonleeeli/venvs/vllm_omni_030/bin/python experiments/s2st_baselines_v1/check_paper_v13.py
"""
from __future__ import annotations

import glob
import json
import re
import sys
from pathlib import Path

R = Path(__file__).resolve().parents[2]
P = R / "paper/bayes_simul_s2st_v13_baselines"
RES = R / "paper/baselines_results"
BUDGETS = {"en2zh": [1.18, 1.98, 2.46, 2.98, 3.48, 4.04], "zh2en": [1.18, 1.92, 2.40, 2.98, 3.62, 4.24]}
DN = {"en2zh": r"En$\rightarrow$Zh", "zh2en": r"Zh$\rightarrow$En"}
bad, n_ok = [], 0


def ok(cond, msg):
    global n_ok
    if cond:
        n_ok += 1
    else:
        bad.append(msg)


def tex(name):
    return (P / name).read_text(encoding="utf-8")


def cell(a, d1):
    return f"{a:.2f}\\,{{\\scriptsize({d1:.2f})}}"


def unbold(c):
    return re.sub(r"^\\textbf\{([\d.]+)\}", r"\1", c.strip())


def rows_of(s):
    """{(dir, tier): cells} for the RealSI rows of a tier table."""
    out, sec = {}, None
    for line in s.split("\n"):
        if "RealSI En$\\rightarrow$Zh" in line or line.startswith("En$\\rightarrow$Zh &"):
            sec = "en2zh"
        elif "RealSI Zh$\\rightarrow$En" in line or line.startswith("Zh$\\rightarrow$En &"):
            sec = "zh2en"
        elif "ACL60/60" in line:
            sec = None
        m = re.search(r"(?:^| )m(\d)(?: \(| &)", line)
        if sec and m and line.rstrip().endswith("\\\\"):
            out[(sec, int(m.group(1)))] = [unbold(c) for c in line.rstrip()[:-2].split(" & ")]
    return out


def header(s):
    line = next(l for l in s.split("\n") if "LA2" in l and "&" in l and "Tier" in l)
    return [c.strip() for c in line.rstrip()[:-2].split(" & ")]


# ---------------------------------------------------------------- sources
xt = {(j["point"], j["direction"]): j for j in (json.load(open(f)) for f in glob.glob(str(RES / "xtranslator/xt-*_*.json")))}
hq = {(j["point"], j["direction"]): j for j in
      (json.load(open(f)) for f in glob.glob(str(RES / "hysimulmt_qwen3tts/hy-*_*.json")))}
xt_t = json.load(open(RES / "xtranslator/tiers.json"))["by_D1"]
hq_t = json.load(open(RES / "hysimulmt_qwen3tts/tiers.json"))["by_D1"]
om = json.load(open(RES / "qabc_omnitalker/summary.json"))["points"]
ok(len(xt) == 8 and len(hq) == 4 and len(om) == 8, "source counts")


def best(pts, budget):     # pts: list of (asr, d1, payload)
    fit = [p for p in pts if p[1] <= budget + 1e-9]
    return max(fit, key=lambda p: p[0]) if fit else None


# ---------------------------------------------------------------- tab_baselines (all points)
s = tex("generated/tab_baselines.tex")
half = {"en2zh": s[:s.index("RealSI Zh$")], "zh2en": s[s.index("RealSI Zh$"):]}
for (pt, d), j in xt.items():
    row = f"& {j['asr_bleu']:.2f} & {j['text_bleu']:.2f} & {j['laal_d1']:.2f} & {j['laal_d0w']:.2f} \\\\"
    ok(half[d].count(row) == 1, f"tab_baselines XT {pt} {d}: {row}")
for (pt, d), j in hq.items():
    row = (f"& {j['asr_bleu']:.2f}$\\pm${j['asr_bleu_sd']:.2f} & {j['text_bleu']:.2f} & {j['laal_d1']:.2f} & "
           f"{j['laal_d0w']:.2f} \\\\")
    ok(half[d].count(row) == 1, f"tab_baselines HyQ {pt} {d}: {row}")

# ---------------------------------------------------------------- tab_realsi_main (means; executors' tiers.json)
s = tex("generated/tab_realsi_main.tex")
h, rows = header(s), rows_of(s)
ix, iq = h.index("X-Translator$^\\P$"), h.index("Hy+Qwen3-TTS$^\\P$")
for d in BUDGETS:
    for k in range(1, 5):
        for t, i, name in ((xt_t, ix, "XT"), (hq_t, iq, "HyQ")):
            e = t[f"{d}|m{k}"]
            want = "--" if e["point"] is None else cell(e["asr_bleu"], e["laal_d1"])
            ok(rows[(d, k)][i] == want, f"tab_realsi_main {name} {d} m{k}: {rows[(d, k)][i]} != {want}")
            ok(abs(e["budget"] - BUDGETS[d][k - 1]) < 1e-9, f"budget {d} m{k}")

# ---------------------------------------------------------------- tab_tiers / tab_tiers_main (seed 0, m1-m6)
xpts = {d: [(j["asr_bleu"], j["laal_d1"], j) for (p, dd), j in xt.items() if dd == d] for d in BUDGETS}
qpts = {d: [(j["seeds"]["0"]["asr_bleu"], j["seeds"]["0"]["laal_d1"], j) for (p, dd), j in hq.items() if dd == d]
        for d in BUDGETS}
for fname, cols in (("generated/tab_tiers.tex", (("X-Translator$^\\P$", xpts, True), ("Hy+Qwen3-TTS$^\\P$", qpts, True))),
                    ("generated/tab_tiers_main.tex", (("XT$^\\P$", xpts, False),))):
    s = tex(fname)
    h, rows = header(s), rows_of(s)
    ok(len(rows) == 12, f"{fname} rows {len(rows)}")
    for col, pts, text in cols:
        i = h.index(col)
        for d in BUDGETS:
            for k in range(1, 7):
                b = best(pts[d], BUDGETS[d][k - 1])
                if b is None:
                    want = "--"
                else:
                    want = (f"{b[0]:.2f}/{b[2]['text_bleu']:.1f}\\,{{\\scriptsize({b[1]:.2f})}}" if text
                            else cell(b[0], b[1]))
                ok(rows[(d, k)][i] == want, f"{fname} {col} {d} m{k}: {rows[(d, k)][i]} != {want}")
    for line in s.split("\n"):           # every row has as many cells as the header
        if line.rstrip().endswith("\\\\") and "&" in line and "multicolumn" not in line:
            ok(line.count(" & ") == len(h) - 1, f"{fname} cell count: {line[:60]}")

# ---------------------------------------------------------------- tab_omnitalker
s = tex("generated/tab_omnitalker.tex")
for key, v in om.items():
    d, t = key.split("_")
    p, o = v["paper"], v["omni"]
    sub = s[s.index("RealSI " + DN[d]):]
    row = (f"{t} & \\textbf{{{p['asr_bleu']:.2f}}}$\\pm${p['asr_bleu_sd']:.2f} & {o['asr_bleu']:.2f}$\\pm${o['asr_bleu_sd']:.2f} & "
           f"{p['asr_bleu'] - o['asr_bleu']:+.2f} & {p['laal_d1']:.2f} & {o['laal_d1']:.2f} & {p['talker_wer']:.2f} & "
           f"{o['talker_wer']:.2f} \\\\")
    ok(sub.split("\\midrule")[0].count(row) == 1 if d == "en2zh" else sub.count(row) == 1, f"tab_omnitalker {key}: {row}")
# cross-check the deltas against the executor's own summary_tables.md
md = (RES / "qabc_omnitalker/summary_tables.md").read_text(encoding="utf-8")
for key, v in om.items():
    dlt = v["omni"]["asr_bleu"] - v["paper"]["asr_bleu"]
    ok(f"| {dlt:.2f} |" in md, f"summary_tables delta {key} {dlt:.2f}")
ex = [v["omni"]["thinker_exact_share_s0"] for v in om.values()]
ok(f"{100 * min(ex):.1f}--{100 * max(ex):.1f}\\%" in s, "thinker exact share range")

# ---------------------------------------------------------------- prose
deltas = [v["paper"]["asr_bleu"] - v["omni"]["asr_bleu"] for v in om.values()]
res, app, setup = tex("sections/results.tex"), tex("sections/appendix.tex"), tex("sections/setup.tex")
ok(f"+{min(deltas):.1f}--{max(deltas):.1f} ASR-BLEU" in res, "results omni range")
ok(f"+{min(deltas):.2f} to +{max(deltas):.2f} ASR-BLEU" in app, "appendix omni range")
wer = sum(v["paper"]["talker_wer"] < v["omni"]["talker_wer"] for v in om.values())
d1l = sum(round(v["paper"]["laal_d1"], 2) < round(v["omni"]["laal_d1"], 2) for v in om.values())
d1e = sum(round(v["paper"]["laal_d1"], 2) == round(v["omni"]["laal_d1"], 2) for v in om.values())
ok(f"rate at {wer} of eight\nand a lower D1 at {d1l} (equal at {d1e})" in app or
   f"rate at {wer} of eight and a lower D1 at {d1l} (equal at {d1e})" in app.replace("\n", " "), "appendix wer/d1 counts")
gaps = []
for d in BUDGETS:
    for k in range(1, 5):
        e = xt_t[f"{d}|m{k}"]
        if e["point"]:
            gaps.append(round(om[f"{d}_m{k}"]["paper"]["asr_bleu"], 2) - e["asr_bleu"])
ok(f"{min(gaps):.1f}--{max(gaps):.1f} below" in res.replace("\n", " "), f"results XT gap {min(gaps):.2f}-{max(gaps):.2f}")
ok("En$\\rightarrow$Zh m4 and Zh$\\rightarrow$En m3--m4" in res.replace("\n", " ")
   and [k for k in xt_t if xt_t[k]["point"]] == ["en2zh|m4", "zh2en|m3", "zh2en|m4"], "results XT tiers")
caps = [j["n_hit_cap_3seeds"] / sum(x["n_tts_blocks"] for x in j["seeds"].values()) for j in hq.values()]
ok(f"{100 * min(caps):.1f}--{100 * max(caps):.1f}\\% of blocks reach the cap" in app.replace("\n", " "), "cap share")
ok("min(1800,\\max(180,18\\times" in app, "cap formula")
tts_fail = max(j["n_tts_fail_segments"] for j in xt.values())
ok(tts_fail <= 2 and "at most two per point" in app.replace("\n", " "), f"XT tts fails max {tts_fail}")

# ---------------------------------------------------------------- citations / build
bib = tex("references.bib")
for k in ("zhao2026xtranslator", "qwen2026qwen3tts", "hy2026simulmt", "seamless2023"):
    ok(bib.count("{" + k + ",") == 1, f"bib {k}")
    ok(any(k in tex(f) for f in ("sections/setup.tex", "sections/appendix.tex")), f"cited {k}")
log = tex("main.log")
ok(not re.search(r"undefined|Citation .* undefined|There were undefined", log, re.I), "undefined refs/cites in main.log")
ok("Overfull" not in log, "overfull boxes")
ok((P / "main.pdf").stat().st_mtime >= max((P / f).stat().st_mtime for f in
                                             ("generated/tab_omnitalker.tex", "sections/appendix.tex", "references.bib")),
   "main.pdf is newer than the filled sources")

print(f"{n_ok} checks passed, {len(bad)} failed")
for b in bad:
    print("FAIL", b)
sys.exit(1 if bad else 0)
