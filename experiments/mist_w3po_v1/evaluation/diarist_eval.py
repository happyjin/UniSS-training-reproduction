"""DiariST official SAgBLEU / SAtBLEU of our AliMeeting outputs (prereg reports/mist_w3po_v1/
multispk_s2st_baselines_prereg_v2.zh-CN.md §5; CPU only).

DiariST (Yang et al., arXiv:2309.08007; github.com/Mu-Y/DiariST, read-only clone with its commit in
research_sources/diarist_20261116/) scores 3-6 min mini-sessions of the AliMeeting Eval (dev, 76) and Test (test, 195)
sessions. This script:

  refs   runs the official diarist/data/generate_data_json.py (unchanged, as run_prepare_data.sh does) on the DiariST
         human translations data/diarist_alimeeting_refs_v1/{dev,test}.txt (sha256 in SOURCE*.md) and the official
         mini-session lists diarist/data/{dev,test}.json, then writes one reference JSON per mini-session exactly as
         generate_diarist_alimeeting.gen_ihm_mix does (fix_timing, json.dump(indent=4, ensure_ascii=False), file name
         f"{session}-{start}-{end}.json"); only the audio is not written (the score needs none).
  hyps   converts a whole-meeting online run (windows file + hyp.jsonl / hyp_asr.jsonl, segments on window clocks) to
         DiariST TSVs (speaker \\t start \\t end \\t text, one row per segment, mini-session clock):
           - session clock of a segment = window start_s + segment start_s (as cp_asr_bleu / score_ext);
           - rows sorted by session-clock start (ties: original order), speaker label "spk<k>" (meeting-level label);
           - a segment belongs to the mini-session containing its start; consecutive official mini-sessions overlap by
             up to 4.15 s, and inside an overlap the later mini-session wins (rule "later"), i.e. the hypothesis
             partition is [start_i, start_{i+1}); the alternative "earlier" ([end_{i-1}, end_i)) is scored as a
             sensitivity check. No segment is dropped; one starting after the last mini-session end joins the last.
           - every mini-session gets a file (empty if it has no segment), so all 195 / 76 are scored.
  score  runs the official diarist/scoring/eval.py unchanged (venvs/s2st_baselines: sacreBLEU 2.6.0, fire; the repo
         pins 2.3.1, see the report) and parses its printed corpus SAgBLEU / SAtBLEU.

    python -m experiments.mist_w3po_v1.evaluation.diarist_eval all [--xtranslator]
Outputs go to eval_outputs/mist_w3po_v1/diarist_eval/ (refuses to overwrite).
"""
from __future__ import annotations

import argparse
import bisect
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
DIARIST = Path("/opt/dlami/nvme/jasonleeeli/research_sources/diarist_20261116/DiariST")
PY = "/opt/dlami/nvme/jasonleeeli/venvs/s2st_baselines/bin/python"
OUT = REPO / "eval_outputs/mist_w3po_v1/diarist_eval"
REFTXT = REPO / "data/diarist_alimeeting_refs_v1"
DEC = REPO / "eval_outputs/mist_w3po_v1/decode"
WIN = {"test": REPO / "eval_outputs/mist_w3po_v1/data/windows_test_c2g06_v1/windows_test_online_ali.jsonl",
       "dev": REPO / "eval_outputs/mist_w3po_v1/data/windows_alieval_c2g06_v1/windows_test_online_ali.jsonl"}
RB = "joint_v8h6k_20261113T073658Z_s06000"
PRE = {"test": f"jtest_bali_{RB}", "dev": f"jaev_b_{RB}"}


def systems(sub: str, xtranslator: bool) -> list[dict]:
    """(key, list of runs averaged, hyp file); B and MIST share the Thinker text."""
    p = PRE[sub]
    out = [{"key": "B_MIST:text", "runs": [p], "hyp": "hyp.jsonl"},
           {"key": "MIST:asr", "runs": [f"{p}__rlplem", f"{p}__rlplem_tts2"], "hyp": "hyp_asr.jsonl"},
           {"key": "B:asr", "runs": [f"{p}__t1c", f"{p}__t1c_tts2"], "hyp": "hyp_asr.jsonl"}]
    if sub == "test":
        out.append({"key": "SeamlessStreaming:asr", "runs": ["jtest_ali_seamless_default"], "hyp": "hyp_asr.jsonl"})
        if xtranslator:
            out.append({"key": "X-Translator:asr", "runs": ["jtest_ali_xtranslator_default"], "hyp": "hyp_asr.jsonl"})
    return out


def write_new(p: Path, text: str) -> None:
    if p.exists():
        raise SystemExit(f"refusing to overwrite {p}")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def mini_name(m: dict) -> str:
    return f"{m['session']}-{m['start']}-{m['end']}"   # as generate_diarist_alimeeting.py names its files


def refs() -> dict:
    stats = {}
    for sub in ("dev", "test"):
        js = OUT / "data/DiariST-AliMeeting" / f"{sub}.json"
        if js.exists():
            raise SystemExit(f"refusing to overwrite {js}")
        js.parent.mkdir(parents=True, exist_ok=True)
        r = subprocess.run([PY, "-I", str(DIARIST / "diarist/data/generate_data_json.py"), str(REFTXT / f"{sub}.txt"),
                            str(DIARIST / f"diarist/data/{sub}.json")], capture_output=True, text=True, check=True)
        js.write_text(r.stdout, encoding="utf-8")
        ms = json.loads(r.stdout)
        d = OUT / "data/DiariST-AliMeeting/IHM-MIX" / sub
        for m in ms:
            data = m["data"]
            for e in data:                      # fix_timing(data, start)
                e["start"] -= m["start"]
                e["end"] -= m["start"]
            p = d / f"{mini_name(m)}.json"
            if p.exists():
                raise SystemExit(f"refusing to overwrite {p}")
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "w", encoding="utf-8") as fp:
                json.dump(data, fp, indent=4, ensure_ascii=False)
        utts = [u for m in ms for u in m["data"]]
        rows = sum(1 for _ in open(REFTXT / f"{sub}.txt", encoding="utf-8"))
        stats[sub] = {"mini_sessions": len(ms), "sessions": len({m["session"] for m in ms}), "tsv_rows": rows,
                      "utterances_in_mini_sessions": len(utts),
                      "words_whitespace": sum(len(u["translation"].split()) for u in utts),
                      "hours_mini_sessions": round(sum(m["end"] - m["start"] for m in ms) / 3600, 2),
                      "mini_dur_s_min_max": [round(min(m["end"] - m["start"] for m in ms), 2),
                                             round(max(m["end"] - m["start"] for m in ms), 2)],
                      "speakers_per_mini_session": sorted({len({u["speaker"] for u in m["data"]}) for m in ms})}
    return stats


def load_minis(sub: str) -> dict:
    ms = json.load(open(OUT / "data/DiariST-AliMeeting" / f"{sub}.json", encoding="utf-8"))
    by = {}
    for m in ms:
        by.setdefault(m["session"], []).append(m)
    for L in by.values():
        L.sort(key=lambda m: m["start"])
    return by


def hyps(sub: str, runs: list[str], hyp: str, rule: str) -> list[dict]:
    """one TSV directory per run; returns conversion statistics."""
    minis = load_minis(sub)
    W = [json.loads(l) for l in open(WIN[sub], encoding="utf-8")]
    out = []
    for run in runs:
        H = {h["id"]: h for h in map(json.loads, open(DEC / run / hyp, encoding="utf-8"))}
        missing = [w["id"] for w in W if w["id"] not in H]
        segs = {}
        n = 0
        for w in W:
            h = H.get(w["id"])
            if h is None:
                continue
            for s in h["segments"]:
                t = re.sub(r"\s+", " ", s["text"]).strip()
                if not t:
                    continue
                a = w["start_s"] + s["start_s"]
                b = w["start_s"] + max(s["end_s"], s["start_s"])
                segs.setdefault(w["meeting"], []).append((a, n, b, f"spk{s['spk']}", t))
                n += 1
        if set(segs) - set(minis):
            raise SystemExit(f"meetings without DiariST mini-sessions: {sorted(set(segs) - set(minis))}")
        d = OUT / "hyp" / rule / sub / run
        if d.exists():
            raise SystemExit(f"refusing to overwrite {d}")
        d.mkdir(parents=True)
        in_ovl = after_end = 0
        files = {mini_name(m): [] for L in minis.values() for m in L}
        for meet, L in segs.items():
            M = minis[meet]
            starts = [m["start"] for m in M]
            ends = [m["end"] for m in M]
            for a, _, b, spk, t in sorted(L):
                if rule == "later":
                    i = max(0, bisect.bisect_right(starts, a) - 1)
                else:                                # "earlier": first mini-session whose end is >= a
                    i = min(len(M) - 1, bisect.bisect_left(ends, a))
                if sum(1 for m in M if m["start"] <= a <= m["end"]) > 1:
                    in_ovl += 1
                if a > ends[-1]:
                    after_end += 1
                m = M[i]
                files[mini_name(m)].append(f"{spk}\t{a - m['start']:.2f}\t{b - m['start']:.2f}\t{t}")
        for name, rows in files.items():
            (d / f"{name}.tsv").write_text("".join(r + "\n" for r in rows), encoding="utf-8")
        out.append({"run": run, "hyp": hyp, "windows_missing": len(missing), "segments": n,
                    "segments_in_overlap": in_ovl, "segments_after_last_end": after_end,
                    "empty_mini_sessions": sum(1 for r in files.values() if not r), "dir": str(d.relative_to(REPO))})
    return out


def oracle(sub: str) -> Path:
    """the references themselves as TSV (pipeline check: must score 100 / 100)."""
    d = OUT / "hyp/oracle" / sub
    if d.exists():
        raise SystemExit(f"refusing to overwrite {d}")
    d.mkdir(parents=True)
    for p in sorted((OUT / "data/DiariST-AliMeeting/IHM-MIX" / sub).glob("*.json")):
        rows = [f"{u['speaker']}\t{u['start']:.2f}\t{u['end']:.2f}\t{u['translation']}"
                for u in json.load(open(p, encoding="utf-8"))]
        (d / f"{p.stem}.tsv").write_text("".join(r + "\n" for r in rows), encoding="utf-8")
    return d


def score(sub: str, hyp_dir: Path, log: Path) -> dict:
    r = subprocess.run([PY, "-I", str(DIARIST / "diarist/scoring/eval.py"), "--ref_dir",
                        str(OUT / "data/DiariST-AliMeeting/IHM-MIX" / sub), "--hyp_dir", str(hyp_dir)],
                       capture_output=True, text=True, check=True)
    write_new(log, r.stdout + r.stderr)
    found = int(re.search(r"Found (\d+) files", r.stdout).group(1))
    ag = float(re.search(r"SAgBLEU: ([\d.]+)", r.stdout).group(1))
    at = float(re.search(r"SAtBLEU: ([\d.]+)", r.stdout).group(1))
    return {"n_mini_sessions": found, "SAgBLEU": ag, "SAtBLEU": at}


def score_system(sub: str, runs: list[str], hyp: str) -> dict:
    row = {"runs": runs, "hyp": hyp}
    for rule in ("later", "earlier"):
        conv = hyps(sub, runs, hyp, rule)
        per = [score(sub, REPO / c["dir"], OUT / "logs" / f"{sub}_{rule}_{c['run']}.log") for c in conv]
        row[rule] = {"per_run": [dict(c, **p) for c, p in zip(conv, per)],
                     "SAgBLEU": round(sum(p["SAgBLEU"] for p in per) / len(per), 2),
                     "SAtBLEU": round(sum(p["SAtBLEU"] for p in per) / len(per), 2),
                     "n_mini_sessions": per[0]["n_mini_sessions"]}
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["all", "extra"],
                    help="all: references + every system; extra: one more system after `all` (e.g. X-Translator), "
                         "written to diarist_eval_v1_extra_<run>.json")
    ap.add_argument("--xtranslator", action="store_true")
    ap.add_argument("--sub", choices=["test", "dev"], default="test")
    ap.add_argument("--key", default="X-Translator:asr")
    ap.add_argument("--run", action="append", help="extra: decode run(s), averaged")
    ap.add_argument("--hyp", default="hyp_asr.jsonl")
    a = ap.parse_args()
    if a.cmd == "extra":
        p = OUT / f"diarist_eval_v1_extra_{a.run[0]}.json"
        if p.exists():
            raise SystemExit(f"refusing to overwrite {p}")
        row = {"set": a.sub, "key": a.key, **score_system(a.sub, a.run, a.hyp)}
        write_new(p, json.dumps(row, ensure_ascii=False, indent=1))
        print(json.dumps({r: (row[r]["SAgBLEU"], row[r]["SAtBLEU"], row[r]["n_mini_sessions"])
                          for r in ("later", "earlier")}))
        return
    res_p = OUT / "diarist_eval_v1.json"
    if res_p.exists():
        raise SystemExit(f"refusing to overwrite {res_p}")
    commit = subprocess.run(["git", "-C", str(DIARIST), "rev-parse", "HEAD"], capture_output=True, text=True,
                            check=True).stdout.strip()
    sb = subprocess.run([PY, "-c", "import sacrebleu;print(sacrebleu.__version__)"], capture_output=True, text=True,
                        check=True).stdout.strip()
    res = {"diarist_repo": "https://github.com/Mu-Y/DiariST", "diarist_commit": commit, "sacrebleu": sb,
           "condition": "IHM-MIX (our whole-meeting online runs on the summed headset audio)",
           "assignment_rule": "segment -> mini-session containing its session-clock start; in overlaps the later "
                              "mini-session (partition [start_i, start_{i+1})); 'earlier' = sensitivity",
           "reference_stats": refs(), "sets": {}}
    for sub in ("test", "dev"):
        S = {"oracle": score(sub, oracle(sub), OUT / "logs" / f"{sub}_oracle.log"), "systems": {}}
        for sy in systems(sub, a.xtranslator):
            S["systems"][sy["key"]] = score_system(sub, sy["runs"], sy["hyp"])
        res["sets"][sub] = S
    res["published_diarist_test_ihm_mix"] = {
        "source": "Yang et al. 2023, arXiv:2309.08007v2, Tables (IHM-MIX, test)",
        "Whisper-small offline ST->SD": {"SAgBLEU": 13.68, "SAtBLEU": 12.10},
        "Whisper-small offline SD->ST": {"SAgBLEU": 10.40, "SAtBLEU": 9.31},
        "streaming 1spk ST": {"SAgBLEU": 16.15}, "streaming 1spk ST + d-vec": {"SAtBLEU": 13.39},
        "streaming 1spk ST + t-vec": {"SAtBLEU": 13.22}, "streaming t-SOT ST": {"SAgBLEU": 16.62},
        "streaming t-SOT ST + d-vec": {"SAtBLEU": 13.85}, "DiariST (t-SOT ST + t-vec)": {"SAgBLEU": 16.62, "SAtBLEU": 14.76}}
    write_new(res_p, json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps({s: {"oracle": v["oracle"], **{k: {r: (x[r]["SAgBLEU"], x[r]["SAtBLEU"]) for r in ("later", "earlier")}
                                                     for k, x in v["systems"].items()}} for s, v in res["sets"].items()},
                     indent=1))


if __name__ == "__main__":
    sys.exit(main())
