"""Idea 2 prereg v2.2: shrink+Vfb token table = the shrink table with the rows of streams that had no usable window at
first commit (n_win == 0) replaced by Idea 1's V tokens (frozen encoder on the HDemucs vocals of the baseline prompt;
eval_outputs/mist_w3po_v1/voicecond_idea1/tokens/game_<dir>_V.jsonl, sha1-checked). All other rows are copied unchanged.

    python3 experiments/mist_w3po_v1/voicecond/idea2_game_vfb_table.py --dir zh2en|en2zh
Writes eval_outputs/mist_w3po_v1/voicecond_idea2/tokens/game_<dir>_shrinkvfb.jsonl (refuses to overwrite)."""
import argparse, json, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[3]
ap = argparse.ArgumentParser(); ap.add_argument("--dir", required=True, choices=["zh2en", "en2zh"]); a = ap.parse_args()
T = REPO / "eval_outputs/mist_w3po_v1/voicecond_idea2/tokens"
out = T / f"game_{a.dir}_shrinkvfb.jsonl"
if out.exists():
    sys.exit(f"refuse to overwrite {out}")
V = {r["key"]: r for r in map(json.loads, open(REPO / f"eval_outputs/mist_w3po_v1/voicecond_idea1/tokens/game_{a.dir}_V.jsonl"))}
rows, nfb = [], 0
for r in map(json.loads, open(T / f"game_{a.dir}_shrink.jsonl")):
    r = dict(r)
    if r["n_win"] == 0:
        v = V[r["key"]]
        assert v["sha1"] == r["sha1"] and len(v["ids"]) == 32, r["key"]
        r["ids_shrink"], r["ids"], r["mode"], r["fallback"] = r["ids"], v["ids"], "shrink_vfb", True
        nfb += 1
    else:
        r["mode"], r["fallback"] = "shrink_vfb", False
    rows.append(r)
with open(out, "x") as fh:
    for r in rows:
        fh.write(json.dumps(r) + "\n")
print(json.dumps({"out": str(out), "n": len(rows), "fallback_V": nfb}))
