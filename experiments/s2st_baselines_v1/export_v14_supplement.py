"""Supplementary CSV files of the single-speaker QABC paper v14: every operating point that v13 listed in its
appendix tables (v13 Tables 14-16: tab_appendix_full, tab_appendix_acl, tab_baselines). The v14 PDF no longer prints
them; Figure 2 and Table 2 summarise them.

Output: paper/bayes_simul_s2st_v14/supplementary/{realsi_operating_points,acl6060_operating_points,
external_systems_operating_points}.csv. Existing files are left untouched unless --force is given.
"""
import csv
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
V13 = REPO / "paper/bayes_simul_s2st_v13_baselines/generated"
OUT = REPO / "paper/bayes_simul_s2st_v14/supplementary"
FORCE = "--force" in sys.argv


def plain(s):
    s = s.replace(r"\qabcc", "QABC-Delta (constant delay)").replace(r"\qabc", "QABC").replace(r"\ttb", "TTB")
    s = s.replace(r"\rightarrow", "->").replace(r"\times", "x").replace(r"\tau", "tau").replace(r"\gamma", "gamma")
    s = s.replace("{=}", "=").replace("{-}", "-").replace(r"\,", " ").replace("$", "").replace("{}", "")
    s = re.sub(r"\\[a-zA-Z]+", "", s)
    return re.sub(r"\s+", " ", s.replace("{", "").replace("}", "")).strip()


def points(cell):
    body = re.search(r"\\raggedright\s*(.*)\}\s*$", cell.strip()).group(1)
    out = []
    for p in body.split(","):
        p = p.strip()
        if not p:
            continue
        q, lat = p.split("/")
        m, _, sd = q.partition("$\\pm$")
        out.append((m.strip(), sd.strip(), lat.strip()))
    return out


def save(name, header, rows):
    OUT.mkdir(exist_ok=True)
    p = OUT / name
    if p.exists() and not FORCE:
        print(f"exists, not overwritten (use --force): {p}")
        return
    with p.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"wrote {p} ({len(rows)} rows)")


rows = []
for line in (V13 / "tab_appendix_full.tex").read_text().splitlines():
    if "\\parbox" not in line:
        continue
    d, pol, cell = line.split(" & ", 2)
    cell = cell.rsplit("\\\\", 1)[0]
    for m, sd, lat in points(cell):
        rows.append([plain(d), plain(pol), m, sd, lat])
save("realsi_operating_points.csv", ["direction", "policy", "asr_bleu_mean_over_talker_seeds", "asr_bleu_sd", "segment_laal_s"], rows)

rows = []
for line in (V13 / "tab_appendix_acl.tex").read_text().splitlines():
    if "\\parbox" not in line:
        continue
    pol, cell = line.split(" & ", 1)
    cell = cell.rsplit("\\\\", 1)[0]
    for m, sd, lat in points(cell):
        rows.append(["En->Zh", plain(pol), m, lat])
save("acl6060_operating_points.csv", ["direction", "policy", "bleu", "streamlaal_s"], rows)

rows, sec, system = [], None, None
for line in (V13 / "tab_baselines.tex").read_text().splitlines():
    m = re.search(r"\\emph\{(.*?)\}\}", line)
    if m:
        sec = plain(m.group(1))
        continue
    if line.count(" & ") != 5 or line.startswith("System"):
        continue
    cells = [c.strip() for c in line.rsplit("\\\\", 1)[0].split(" & ")]
    if cells[0]:
        system = plain(cells[0])
    if system == "QABC":
        continue   # QABC's own points are in realsi_operating_points.csv
    rows.append([sec, system, plain(cells[1])] + cells[2:])
save("external_systems_operating_points.csv",
     ["test_set", "system", "setting", "asr_bleu", "text_bleu", "segment_laal_s", "simuleval_word_laal_s"], rows)
