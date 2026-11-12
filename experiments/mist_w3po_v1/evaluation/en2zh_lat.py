"""speech_latency_lit (and its --text-only mode) on an En->Zh run (reports/mist_w3po_v1/en2zh_full_plan_prereg_v1
.zh-CN.md §5.3), without changing the tool: the zh2en tools count words with str.split(), so every text they read is
first rewritten with zh_units.charify (one token per CJK character / Latin word, punctuation attached):

  <work>/run/hyp.jsonl           the run's hyp.jsonl, segment (and block) texts charified
  <work>/talker/results.jsonl    the talker's results.jsonl, block texts charified (speech runs only)
  references                     metric_headroom.scored_utts wrapped: every reference text charified
  metric_headroom.REF_WIN / DR   --ref-win / "en2zh" (the scored En->Zh reference windows and the en2zh universe =
                                 every annotated AMI utterance of the meeting, as overlap_score)

Timing, windows (--win: the C2 or chunk-protocol windows), the assignment and every formula are speech_latency_lit's.
Per-word delays become per-unit delays; LAAL's |Y| = max(output units, reference units). <work> must be new.

    python -m experiments.mist_w3po_v1.evaluation.en2zh_lat --ref-win <en2zh ref windows> --work <new dir> \\
        --run <decode dir> [--talker <talker dir> | --text-only] --win <windows> --out <json>

Test-only switches right after --work: --dr zh2en|ali (another universe) and --units word (no charify: the copies are
verbatim), with which the wrapper must reproduce a stored zh2en latency json exactly (test_en2zh_lat).
"""
from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path

from experiments.mist_w3po_v1.evaluation.zh_units import charify


def charify_hyp(src: Path, dst: Path, f=charify) -> int:
    n = 0
    with open(dst, "x", encoding="utf-8") as fh:
        for l in open(src, encoding="utf-8"):
            h = json.loads(l)
            for key in ("segments", "blocks"):
                for s in h.get(key) or []:
                    if "text" in s:
                        s["text"] = f(s["text"])
            fh.write(json.dumps(h, ensure_ascii=False) + "\n")
            n += 1
    return n


def charify_results(src: Path, dst: Path, f=charify) -> int:
    n = 0
    with open(dst, "x", encoding="utf-8") as fh:
        for l in open(src, encoding="utf-8"):
            r = json.loads(l)
            for b in r.get("blocks") or []:
                b["text"] = f(b["text"])
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def take(argv: list[str], key: str) -> str | None:
    if key not in argv:
        return None
    i = argv.index(key)
    return argv[i + 1]


def main() -> None:
    argv = sys.argv[1:]
    if len(argv) < 6 or argv[0] != "--ref-win" or argv[2] != "--work":
        raise SystemExit(__doc__)
    ref, work, rest = argv[1], Path(argv[3]), argv[4:]
    dr, unit = "en2zh", "char"
    while rest and rest[0] in ("--dr", "--units"):
        if rest[0] == "--dr":
            dr = rest[1]
        else:
            unit = rest[1]
        rest = rest[2:]
    assert dr in ("en2zh", "zh2en", "ali") and unit in ("char", "word"), (dr, unit)
    f = charify if unit == "char" else (lambda t: t)
    from experiments.mist_w3po_v1.evaluation import metric_headroom as M

    if not (M.REPO / ref).exists():
        raise SystemExit(f"reference windows missing: {ref}")
    if work.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {work}")
    run, talker = take(rest, "--run"), take(rest, "--talker")
    if run is None:
        raise SystemExit("--run required")
    (work / "run").mkdir(parents=True)
    nh = charify_hyp(Path(run) / "hyp.jsonl", work / "run" / "hyp.jsonl", f)
    rest[rest.index("--run") + 1] = str(work / "run")
    nr = 0
    if talker:
        (work / "talker").mkdir()
        nr = charify_results(Path(talker) / "results.jsonl", work / "talker" / "results.jsonl", f)
        rest[rest.index("--talker") + 1] = str(work / "talker")
    (work / "source.json").write_text(json.dumps({"run": run, "talker": talker, "ref_win": ref, "dr": dr,
                                                  "hyp_lines": nh, "results_lines": nr,
                                                  "units": "zh_units.charify" if unit == "char" else "word (verbatim)"},
                                                 indent=1), encoding="utf-8")
    M.REF_WIN, M.DR = ref, dr
    orig = M.scored_utts

    def scored_utts(W, RW):
        sc = orig(W, RW)
        for v in sc.values():
            v["ref"] = f(v["ref"])
        return sc

    M.scored_utts = scored_utts
    print(f"[en2zh_lat] REF_WIN={ref} DR={dr} units {unit}: run {run} -> {work}", file=sys.stderr, flush=True)
    sys.argv = ["experiments.mist_w3po_v1.evaluation.speech_latency_lit"] + rest
    runpy.run_module("experiments.mist_w3po_v1.evaluation.speech_latency_lit", run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
