"""Robust wave 16 NCWI helpers (reports/mist_w3po_v1/robust_wave16_ncwi_prereg.zh-CN.md, section 3), CPU only.

  agree     the alpha-0.5 LoRA run (wave16/eval_lora05) against the merged unist_wise05 (wave14/eval_unist_wise05):
            share of items with identical full text and with identical blocks (text + write time), per direction; the
            LoRA path is accepted iff both shares reach the smallest pairwise share among wave 5's 3 base runs (S1,
            base_rerun1, base_rerun2) in both directions, and both NCWI LoRA outputs (wave16/eval_ncwi) are complete.
            Writes --out json; exit 0 = accepted, 3 = rejected.
  groups    merged fallback: per direction and distinct alpha*, the subset items file
            wave16/eval_ncwi_merged/parts/items_<dir>_a<alpha>.jsonl (S1 order) and one line
            "<dir> <alpha> <model> <items file> <n>" (model: alpha 0 the base, 0.5 / 1 wave 14's merged copies, else
            wave16/merged_a<alpha>).
  assemble  merged fallback: the per-alpha blocks files -> wave16/eval_ncwi_merged/thinker_<dir>/blocks_<dir>_w1.jsonl in
            S1 order; every item exactly once, from the part of its own alpha*.
  wisejson  write wave16/adapters_a<alpha>/adapters_final.json (run_unist's json with alpha 32 * alpha).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
M = REPO / "eval_outputs/mist_w3po_v1/wave16"
W14 = REPO / "eval_outputs/mist_w3po_v1/wave14"
S1 = REPO / "eval_outputs/mist_w3po_v1/wave2_S1"
BASE_RUNS = {"S1": S1, "r1": REPO / "eval_outputs/mist_w3po_v1/wave5/base_rerun1",
             "r2": REPO / "eval_outputs/mist_w3po_v1/wave5/base_rerun2"}
QWEN3 = "pretrained_models/Qwen3-Omni-30B-A3B-Instruct"
DIRS = ("en2zh", "zh2en")


def jl(p):
    return [json.loads(x) for x in open(p, encoding="utf-8") if x.strip()]


def blocks(root: Path, d: str) -> dict:
    return {r["id"]: r["blocks"] for r in jl(root / f"thinker_{d}" / f"blocks_{d}_w1.jsonl")}


def shares(x: dict, y: dict, ids) -> tuple:
    txt = sum("".join(b["text"] for b in x[i]) == "".join(b["text"] for b in y[i]) for i in ids) / len(ids)
    blk = sum(x[i] == y[i] for i in ids) / len(ids)
    return round(txt, 4), round(blk, 4)


def agree(a) -> int:
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    res, ok = {}, True
    for d in DIRS:
        ids = [r["id"] for r in jl(S1 / "data" / f"items_{d}.jsonl")]
        r = {}
        try:
            nc = blocks(M / "eval_ncwi", d)
            r["ncwi_complete"] = set(nc) == set(ids)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            r["ncwi_complete"] = False; r["ncwi_error"] = str(e)
        try:
            lo = blocks(M / "eval_lora05", d)
            assert set(lo) == set(ids), f"{len(set(ids) - set(lo))} missing"
            ref = blocks(W14 / "eval_unist_wise05", d)
            assert set(ref) == set(ids)
            r["lora05_vs_unist_wise05_[text,blocks]"] = list(shares(lo, ref, ids))
        except (FileNotFoundError, AssertionError, json.JSONDecodeError) as e:
            r["lora05_error"] = str(e)
        B = {k: blocks(v, d) for k, v in BASE_RUNS.items()}
        pw = {f"{x}-{y}": list(shares(B[x], B[y], ids)) for x, y in (("S1", "r1"), ("S1", "r2"), ("r1", "r2"))}
        r["base_pairwise_[text,blocks]"] = pw
        thr = (min(v[0] for v in pw.values()), min(v[1] for v in pw.values()))
        r["threshold_[text,blocks]"] = list(thr)
        got = r.get("lora05_vs_unist_wise05_[text,blocks]")
        r["pass"] = bool(r["ncwi_complete"] and got and got[0] >= thr[0] and got[1] >= thr[1])
        ok &= r["pass"]
        res[d] = r
    res["lora_path_accepted"] = ok
    with open(out, "x", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)
    print(json.dumps(res), flush=True)
    return 0 if ok else 3


def model_of(al: float) -> str:
    return {0.0: str(REPO / QWEN3), 0.5: str(W14 / "merged_unist_wise05"),
            1.0: str(W14 / "merged_unist")}.get(al, str(M / f"merged_a{al:.4f}"))


def groups(a) -> int:
    P = M / "eval_ncwi_merged" / "parts"
    P.mkdir(parents=True, exist_ok=True)
    for d in DIRS:
        al = {r["id"]: round(float(r["alpha"]), 4) for r in jl(M / "fit" / f"assign_{d}.jsonl")}
        items = jl(S1 / "data" / f"items_{d}.jsonl")
        assert set(al) == {it["id"] for it in items}
        for x in sorted(set(al.values())):
            f = P / f"items_{d}_a{x:.4f}.jsonl"
            sub = [it for it in items if al[it["id"]] == x]
            if f.exists():
                assert jl(f) == sub, f"{f} exists with other content"
            else:
                with open(f, "x", encoding="utf-8") as fh:
                    fh.writelines(json.dumps(it, ensure_ascii=False) + "\n" for it in sub)
            print(d, f"{x:.4f}", model_of(x), f, len(sub))
    return 0


def assemble(a) -> int:
    for d in DIRS:
        al = {r["id"]: round(float(r["alpha"]), 4) for r in jl(M / "fit" / f"assign_{d}.jsonl")}
        items = jl(S1 / "data" / f"items_{d}.jsonl")
        got = {}
        for x in sorted(set(al.values())):
            for r in jl(M / "eval_ncwi_merged" / "parts" / f"thinker_{d}_a{x:.4f}" / f"blocks_{d}_w1.jsonl"):
                assert r["id"] not in got and al[r["id"]] == x, (d, x, r["id"])
                got[r["id"]] = r
        miss = [it["id"] for it in items if it["id"] not in got]
        assert not miss, f"{d}: {len(miss)} items missing, e.g. {miss[:3]}"
        o = M / "eval_ncwi_merged" / f"thinker_{d}"
        o.mkdir(parents=True, exist_ok=True)
        with open(o / f"blocks_{d}_w1.jsonl", "x", encoding="utf-8") as fh:
            fh.writelines(json.dumps(got[it["id"]], ensure_ascii=False) + "\n" for it in items)
        print(d, "assembled", len(items), flush=True)
    return 0


def wisejson(a) -> int:
    x = round(float(a.alpha), 4)
    src = json.load(open(W14 / "run_unist" / "adapters_final.json"))
    assert src["alpha"] == 32.0 and src["rank"] == 16, src
    src.update({"arm": f"ncwi_a{x:.4f}", "alpha": 32.0 * x,
                "wise_ft": f"W0 + {x}*(W_unist - W0): LoRA alpha x {x}, adapters = wave14/run_unist/adapters_final.pt"})
    with open(M / f"adapters_a{x:.4f}" / "adapters_final.json", "x") as fh:
        json.dump(src, fh)
    print(json.dumps(src))
    return 0


def extracards(a) -> int:
    """cards 4 / 5 a wave-16 job may stack on now (coordinator 16:15 CET): no training process on the card (a
    no-checkpoint placeholder, scripts/gpu_load/, does not count -- the launcher stops it by PID) and effective free memory
    = total - used + the placeholders' memory >= --min-gb (nvidia-smi). Cards 6-7 (wave 17 training) are never listed.
    Prints the eligible cards space-separated (maybe nothing) and one diagnostic line per card to stderr."""
    import subprocess
    q = lambda args: [[x.strip() for x in ln.split(",")] for ln in subprocess.run(
        ["nvidia-smi", *args, "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True).stdout.splitlines() if ln.strip()]
    blk = REPO / "eval_outputs/mist_w3po_v1/combo/ctrl_pool_base/DONE"
    if not blk.exists():   # coordinator 18:15 CET: uniss-de's ctrl_pool_base holds cards 4-5 until this flag appears
        print(f"cards 4-5 reserved for ctrl_pool_base until {blk.relative_to(REPO)} exists -> none", file=sys.stderr)
        print("")
        return 0
    gpus = {u: (int(i), float(t), float(us)) for i, u, t, us in q(["--query-gpu=index,uuid,memory.total,memory.used"])}
    apps = q(["--query-compute-apps=gpu_uuid,pid,used_memory"])
    ok = []
    for card in (4, 5):
        u = next(k for k, v in gpus.items() if v[0] == card)
        _, tot, used = gpus[u]
        ph, train, other = 0.0, [], []
        for gu, pid, mem in apps:
            if gu != u:
                continue
            try:
                cmd = open(f"/proc/{pid}/cmdline", "rb").read().replace(b"\0", b" ").decode(errors="replace")
            except OSError:
                cmd = ""
            if "scripts/gpu_load/" in cmd:
                ph += float(mem)
            elif any(t in cmd for t in ("torch.distributed.run", "torchrun", "_train.py", "train_")):
                train.append(pid)
            else:
                other.append(pid)
        free = (tot - used + ph) / 1024
        good = not train and free >= a.min_gb
        print(f"card {card}: free {free:.1f} GB (placeholders {ph / 1024:.1f} GB), training {train}, other {other} -> "
              f"{'eligible' if good else 'no'}", file=sys.stderr)
        if good:
            ok.append(str(card))
    print(" ".join(ok))
    return 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("agree", "groups", "assemble", "wisejson", "extracards"))
    ap.add_argument("--out", default="")
    ap.add_argument("--alpha", default="")
    ap.add_argument("--min-gb", type=float, default=60.0)
    a = ap.parse_args()
    sys.exit({"agree": agree, "groups": groups, "assemble": assemble, "wisejson": wisejson,
              "extracards": extracards}[a.cmd](a))


if __name__ == "__main__":
    main()
