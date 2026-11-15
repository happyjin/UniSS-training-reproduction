"""speech_latency_lit with a per-unit dump, under either speech-LAAL convention (post-review v10 analysis, M5).

  LATU_MODE=perword     the tool unchanged (the paper's speech LAAL: a block's words spread evenly over its playback)
  LATU_MODE=blockstart  every word of a block is charged at the START of the block's online playback (the single-speaker
                        convention of the paper's block-start table; the same one-line patch as the v10 table's
                        scratchpad m2laal.py, mode "online")

Both modes additionally write <out>.units.jsonl: one row per value the tool appends to its unit lists
([meeting, unit id, region, key, value]), so meeting-level bootstraps can be computed afterwards. Nothing else changes:
the repo's tool is read, patched in a copy next to <out>, never edited; wrappers (on_refset, en2zh_lat) pass through.

    LATU_MODE=blockstart python -m experiments.mist_w3po_v1.evaluation.lat_units_dump <module> <module args ... --out X>
"""
import os
import runpy
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
LIT = "experiments.mist_w3po_v1.evaluation.speech_latency_lit"


def patched_source(mode: str, dump: str) -> str:
    src = (REPO / "experiments/mist_w3po_v1/evaluation/speech_latency_lit.py").read_text()
    reps = []
    if mode == "blockstart":
        reps.append(('            out.append(b[k0_] + (k + 1) / len(bw) * (b[k1_] - b[k0_]) - a)\n',
                     '            out.append(b[k0_] - a)\n'))
    elif mode != "perword":
        raise SystemExit(f"LATU_MODE must be perword or blockstart, got {mode!r}")
    reps += [
        ("from __future__ import annotations\n",
         "from __future__ import annotations\n"
         "_CUR = [None, None, None]\n_DUMP = []\n"
         "class _LogList(list):\n"
         "    def __init__(self, key):\n        super().__init__()\n        self.key = key\n"
         "    def append(self, v):\n        _DUMP.append([_CUR[0], _CUR[1], _CUR[2], self.key, float(v)])\n"
         "        super().append(v)\n"
         "class _KeyedLog(dict):\n"
         "    def __missing__(self, k):\n        self[k] = _LogList(k)\n        return self[k]\n"),
        ("    U_ = defaultdict(lambda: defaultdict(list))\n", "    U_ = defaultdict(_KeyedLog)\n"),
        ('            reg = u["reg"]\n', '            reg = u["reg"]\n            _CUR[:] = [m, uid, reg]\n'),
        ('    print(json.dumps({"check": chk',
         f'    with open({dump!r}, "x", encoding="utf-8") as _fh:\n'
         '        _fh.writelines(json.dumps(r_) + "\\n" for r_ in _DUMP)\n'
         '    print(json.dumps({"check": chk'),
    ]
    for old, new in reps:
        assert src.count(old) == 1, old
        src = src.replace(old, new)
    return src


def main() -> None:
    mode = os.environ["LATU_MODE"]
    argv = sys.argv[1:]
    out = argv[argv.index("--out") + 1]
    dump = out + ".units.jsonl"
    if Path(out).exists() or Path(dump).exists():
        raise SystemExit(f"refusing to overwrite {out}")
    patched = Path(out).parent / (Path(out).stem + f".lit_{mode}.py")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(patched, "x", encoding="utf-8") as fh:
        fh.write(patched_source(mode, dump))
    orig = runpy.run_module

    def rm(name, *a, **k):
        if name == LIT:
            return runpy.run_path(str(patched), run_name="__main__")
        return orig(name, *a, **k)
    runpy.run_module = rm
    target, sys.argv = argv[0], argv
    if target == LIT:
        runpy.run_path(str(patched), run_name="__main__")
    else:
        orig(target, run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
