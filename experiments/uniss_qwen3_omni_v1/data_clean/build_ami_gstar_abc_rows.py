"""ABC read rows for multi-speaker interpreting (docs/multi_speaker_si_plan_v9_bidir_abc.zh-CN.md §3), both directions.

  pick   training windows to label: --n per direction, multi-speaker windows drawn --multi-weight times as often
         (en->zh: e2e_gstar_stream_v1, zh->en: e2e_gstar_zh2en_v1; both carry the low-latency preset's posteriors)
  rows   the oracle dumps (gstar_si_stream_online --policy oracle on those windows, with the offline model O2) -> one
         row per read: the audio heard so far, the posteriors the diarizer has finalised (post_until_ms = t - 1040),
         the committed text as prefix, the stable prefix as target, then <|box_end|> (wait) -- or the rest then
         <|im_end|> at the window's end. Reads with nothing to commit are kept with probability --keep-empty. Mixed in:
         whole-window offline rows (--offline-share of the read rows, from O2's training rows) and single-speaker
         rows (--sd-share), so the offline ability is kept.

    python -m experiments.uniss_qwen3_omni_v1.data_clean.build_ami_gstar_abc_rows pick --out <dir>
    python -m experiments.uniss_qwen3_omni_v1.data_clean.build_ami_gstar_abc_rows rows --out <dir> --dumps <glob> ...
"""

from __future__ import annotations

import argparse
import glob
import json
import random
from pathlib import Path

WIN = {"en2zh": "data/ami_multispk_si_v1/e2e_gstar_stream_v1/windows_train_ref.jsonl",
       "zh2en": "data/ami_multispk_si_v1/e2e_gstar_zh2en_v1/windows_train_ref.jsonl"}
LOOKAHEAD_MS = 1040


def pick(args) -> None:
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    out.mkdir(parents=True)
    rng = random.Random(args.seed)
    for d, f in WIN.items():
        W = [json.loads(l) for l in open(f, encoding="utf-8")]
        talk = lambda w: {u["speaker_id"] for u in w["utts"] if not u.get("hesitation")}   # noqa: E731
        weights = [args.multi_weight if len(talk(w)) > 1 else 1.0 for w in W]
        chosen, idx = set(), list(range(len(W)))
        while len(chosen) < min(args.n, len(W)):
            chosen.update(rng.choices(idx, weights=weights, k=args.n - len(chosen)))
        with open(out / f"windows_{d}.jsonl", "x", encoding="utf-8") as fh:
            for i in sorted(chosen):
                fh.write(json.dumps({**W[i], "dir": d}, ensure_ascii=False) + "\n")
        print(d, len(chosen), "of", len(W))


def rows(args) -> None:
    out = Path(args.out)
    p = out / "rows_train.jsonl"
    if p.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {p}")
    rng = random.Random(args.seed)
    W = {}
    for d in WIN:
        for line in open(out / f"windows_{d}.jsonl", encoding="utf-8"):
            w = json.loads(line)
            W[w["id"]] = w
    rs, stats = [], {"windows": 0, "reads": 0, "kept": 0, "empty_dropped": 0, "nonempty": 0}
    for f in sorted(glob.glob(args.dumps)):
        for line in open(f, encoding="utf-8"):
            h = json.loads(line)
            w = W[h["id"]]
            stats["windows"] += 1
            for r in h["read_log"]:
                stats["reads"] += 1
                if not r["final"] and not r["target"]:
                    if rng.random() >= args.keep_empty:
                        stats["empty_dropped"] += 1
                        continue
                else:
                    stats["nonempty"] += 1
                rs.append({"id": f"{w['id']}|abc|{r['t']}", "audio": w["audio"], "dir": w["dir"], "t_ms": r["t"],
                           "dur_ms": w["dur_ms"], "post": w["post"]["lowlat"],
                           "post_until_ms": r["t"] - LOOKAHEAD_MS, "prefix": r["prefix"], "target": r["target"],
                           "end": "eos" if r["final"] else "wait", "kind": "read", "inst": "gstar", "preset": "lowlat"})
    stats["kept"] = len(rs)
    off = [json.loads(l) for f in args.offline_rows for l in open(f, encoding="utf-8")]
    gstar_off = [r for r in off if r.get("inst") == "gstar"]
    sd = [r for r in off if r.get("inst") == "sd"]
    extra = rng.sample(gstar_off, min(len(gstar_off), round(args.offline_share * len(rs))))
    extra += rng.sample(sd, min(len(sd), round(args.sd_share * len(rs))))
    stats["offline_rows"], stats["sd_rows"] = len(extra) - min(len(sd), round(args.sd_share * len(rs))), \
        min(len(sd), round(args.sd_share * len(rs)))
    allr = rs + extra
    rng.shuffle(allr)
    with open(p, "x", encoding="utf-8") as fh:
        fh.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in allr)
    stats["rows"] = len(allr)
    (out / "summary.json").write_text(json.dumps({"stats": stats, "args": vars(args)}, indent=1), encoding="utf-8")
    print(json.dumps(stats))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("pick")
    a.add_argument("--out", required=True)
    a.add_argument("--n", type=int, default=4000)
    a.add_argument("--multi-weight", type=float, default=1.5)
    a.add_argument("--seed", type=int, default=20261029)
    a = sub.add_parser("rows")
    a.add_argument("--out", required=True)
    a.add_argument("--dumps", required=True, help="glob of the oracle dump jsonl files")
    a.add_argument("--offline-rows", nargs="+", required=True, help="O2's training rows (offline and sd rows)")
    a.add_argument("--keep-empty", type=float, default=0.5)
    a.add_argument("--offline-share", type=float, default=0.2)
    a.add_argument("--sd-share", type=float, default=0.15)
    a.add_argument("--seed", type=int, default=20261029)
    args = ap.parse_args()
    {"pick": pick, "rows": rows}[args.cmd](args)


if __name__ == "__main__":
    main()
