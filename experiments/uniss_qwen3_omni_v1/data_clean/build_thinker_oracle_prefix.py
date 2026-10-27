"""Plan 4b's rows: the Thinker taught to say, at each read, only the part of its own greedy draft that the whole
sentence would not change, then wait -- the oracle commit (si_thinker_stream --policy oracle) amortised into it.

From the oracle dumps (si_thinker_stream --policy head --oracle --dump, data/si_bayes_lr_v1/dump_* and
data/si_belief_v1/shard_*): at every read the state is (the audio heard, the translation committed before it) and the
target is the draft's leading units the oracle -- the same prefix continued with the whole sentence heard -- agrees
with, followed by the wait marker <|box_end|>. The read that hears the end gets the rest of the run's translation and
<|im_end|>, and a share of rows are the whole audio with the Thinker's own greedy whole-sentence translation (the oracle
of the item's first read), so it keeps translating whole sentences. Every target is the Thinker's own greedy text cut
short -- unlike Plan 4A's (a vote of samples at temperature 0.7), which cost en2zh 1.4 BLEU at full latency.

Rows (as Plan 4A's): {"id", "audio", "dir", "t_ms", "dur_ms", "prefix", "target", "end": "wait" | "eos", "kind"}

    python -m experiments.uniss_qwen3_omni_v1.data_clean.build_thinker_oracle_prefix --out data/si_thinker_oracle_prefix_v1

Fluency plan v3 (all off by default): --oracle-rewrites takes accepted interpreter-style rewrites of the whole-sentence
oracle (rewrite_interpreter_style.py, filtered); a read with nothing committed then targets the draft's agreement with
the rewrite, and --rewrite-offline-share of the offline rows say the rewrite. --clean-targets passes every prefix and
target through si_fluency_rules.clean (no fillers, no immediate repeats). --start-boost k repeats the reads with nothing
committed k times.
"""

from __future__ import annotations

import argparse
import glob
import json
import random
import re
import zlib
from pathlib import Path


def lcp(a: list, b: list) -> int:
    k = 0
    while k < min(len(a), len(b)) and a[k] == b[k]:
        k += 1
    return k


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dumps", default="data/si_bayes_lr_v1/dump_*/dumps/*.jsonl,data/si_belief_v1/shard_*/dumps/*.jsonl")
    ap.add_argument("--items", default="data/unist_simul_thinker_v1/items_{d}.jsonl")
    ap.add_argument("--offline-share", type=float, default=0.2)
    ap.add_argument("--max-rows", type=int, default=240000)
    ap.add_argument("--val-every", type=int, default=50, help="1 item in this many held out (stable hash)")
    ap.add_argument("--seed", type=int, default=20261023)
    ap.add_argument("--oracle-rewrites", default="", help="jsonl {id, rewrite, accepted} (fluency plan v3)")
    ap.add_argument("--rewrite-offline-share", type=float, default=0.5)
    ap.add_argument("--clean-targets", action="store_true")
    ap.add_argument("--start-oracle", default="", help="build_start_oracle.py output: a start read's target becomes the "
                                                      "draft's first best_L units when that is longer than the agreement")
    ap.add_argument("--start-boost", type=int, default=1)
    ap.add_argument("--lengthened-boost", type=int, default=1,
                    help="with --start-oracle: repeat only the start reads whose target the start oracle lengthened")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    rng = random.Random(args.seed)
    audio = {}
    for d in ("en2zh", "zh2en"):
        for line in open(args.items.format(d=d), encoding="utf-8"):
            r = json.loads(line)
            audio[r["id"]] = (r["audio"], int(r["dur_ms"]))
    rew = {}
    if args.oracle_rewrites:
        for line in open(args.oracle_rewrites, encoding="utf-8"):
            r = json.loads(line)
            if r.get("accepted"):
                rew[r["id"]] = r["rewrite"]
    reads, finals, offline = [], [], {}
    n_start_rew = [0, 0]
    start_L = {}
    if args.start_oracle:
        for line in open(args.start_oracle, encoding="utf-8"):
            r = json.loads(line)
            start_L[(r["id"], int(r["t"]))] = int(r["best_L"])
    n_start_L = [0, 0]
    for f in sorted({f for pat in args.dumps.split(",") for f in glob.glob(pat)}):
        d = "en2zh" if "_en2zh_" in f else "zh2en"
        zh = d == "en2zh"
        run = Path(f).parent.parent / Path(f).name   # the blocks file beside the dumps directory
        blocks = {json.loads(l)["id"]: json.loads(l) for l in open(run, encoding="utf-8")}
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            if r["id"] not in audio or r["id"] not in blocks or "oracle_units" not in r:
                continue
            bl = blocks[r["id"]]["blocks"]
            t = int(r["t"])
            prefix = "".join(b["text"] for b in bl if b["write_ms"] < t)
            cont = "".join(r["tokens"])
            spans = [m.end() for m in re.finditer(r"\S" if zh else r"\S+", cont)]
            oracle = r["oracle_units"]
            if r["n_committed"] == 0 and r["id"] in rew:   # a sentence start: agree with the interpreter-style rewrite
                oracle = [c for c in rew[r["id"]] if not c.isspace()] if zh else rew[r["id"]].split()
                n_start_rew[0] += 1
                n_start_rew[1] += lcp(r["draft_units"], oracle) > 0
            L = lcp(r["draft_units"], oracle)
            lengthened = False
            if r["n_committed"] == 0 and (r["id"], t) in start_L:   # the prefix-constrained start oracle
                n_start_L[0] += 1
                if start_L[(r["id"], t)] > L:
                    n_start_L[1] += 1
                    L = start_L[(r["id"], t)]
                    lengthened = True
            target = cont[: spans[L - 1]] if L and L <= len(spans) else ""
            row = {"id": r["id"], "audio": audio[r["id"]][0], "dir": d, "t_ms": t, "dur_ms": audio[r["id"]][1],
                   "prefix": prefix, "target": target, "end": "wait", "kind": "read"}
            reads.extend([row] * ((args.start_boost if r["n_committed"] == 0 else 1) * (args.lengthened_boost if lengthened else 1)))
            if r["n_committed"] == 0 and r["id"] not in offline:   # the oracle of an empty prefix: the greedy whole
                ou = r["oracle_units"]
                whole = "".join(ou) if zh else " ".join(ou)
                if r["id"] in rew and zlib.crc32(("off" + r["id"]).encode()) % 1000 < 1000 * args.rewrite_offline_share:
                    whole = rew[r["id"]]
                offline[r["id"]] = {"id": r["id"], "audio": audio[r["id"]][0], "dir": d, "t_ms": audio[r["id"]][1],
                                    "dur_ms": audio[r["id"]][1], "prefix": "", "target": whole,
                                    "end": "eos", "kind": "offline"}
        for i, b in blocks.items():   # the read that hears the end: the rest, then the end of the reply
            if i not in audio or not b["blocks"] or not b["blocks"][-1].get("final"):
                continue
            last = b["blocks"][-1]
            finals.append({"id": i, "audio": audio[i][0], "dir": d, "t_ms": audio[i][1], "dur_ms": audio[i][1],
                           "prefix": "".join(x["text"] for x in b["blocks"][:-1]), "target": last["text"], "end": "eos",
                           "kind": "read"})
    if args.clean_targets:   # the Thinker learns to say neither fillers nor repeats
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
        from experiments.uniss_qwen3_omni_v1.evaluation.si_fluency_rules import clean
        cleaned = []
        for r in reads + finals + list(offline.values()):
            zh = r["dir"] == "en2zh"
            p = clean("", r["prefix"], zh) if r["prefix"] else ""
            t = clean(p, (" " if (not zh and p and r["target"] and not r["target"].startswith(" ")) else "") + r["target"], zh) if r["target"] else ""
            cleaned.append({**r, "prefix": p, "target": t if zh or not p else t})
        n_r, n_f = len(reads), len(finals)
        reads, finals = cleaned[:n_r], cleaned[n_r: n_r + n_f]
        offline = {r["id"]: r for r in cleaned[n_r + n_f:]}
    is_val = lambda r: zlib.crc32(r["id"].encode()) % args.val_every == 0
    n_off = int(args.max_rows * args.offline_share)
    off = list(offline.values())
    rng.shuffle(off)
    off = off[:n_off]
    rest = reads + finals
    rng.shuffle(rest)
    rows = off + rest[: args.max_rows - len(off)]
    rng.shuffle(rows)
    out.mkdir(parents=True)
    tr, va = [r for r in rows if not is_val(r)], [r for r in rows if is_val(r)]
    for name, part in (("rows_train.jsonl", tr), ("rows_val.jsonl", va)):
        with open(out / name, "x", encoding="utf-8") as fh:
            fh.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in part)
    stats = {"reads_all": len(reads), "finals_all": len(finals), "offline_all": len(offline), "train": len(tr), "val": len(va),
             "wait_share": round(sum(r["end"] == "wait" for r in rows) / len(rows), 3),
             "empty_wait_share": round(sum(r["end"] == "wait" and not r["target"] for r in rows) / len(rows), 3),
             "offline_share": round(sum(r["kind"] == "offline" for r in rows) / len(rows), 3),
             "en2zh": sum(r["dir"] == "en2zh" for r in rows), "zh2en": sum(r["dir"] == "zh2en" for r in rows),
             "start_reads_with_start_oracle": n_start_L[0], "start_targets_lengthened": n_start_L[1],
             "start_reads_with_rewrite": n_start_rew[0], "their_agreement": round(n_start_rew[1] / max(1, n_start_rew[0]), 3),
             "args": {k: v for k, v in vars(args).items() if k != "out"}}
    (out / "README.md").write_text("Plan 4b rows (see data_clean/build_thinker_oracle_prefix.py).\n\n"
                                   + json.dumps(stats, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(stats), flush=True)


if __name__ == "__main__":
    main()
