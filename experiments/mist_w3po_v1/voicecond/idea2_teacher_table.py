"""Idea 2 e2e prereg §3/§5: the teacher's voice tokens g* for the cvt teacher items, as an idea1_si_infer_vo table:
one row per speaker prompt file teacher_in/voice/<spk>.wav: {key = file stem, ids = g* (tokens of the centroid of the
speaker's clean 6 s prompts, cvt/refs/ustar*.npz), sha1 of the prompt file}. Refuses to overwrite.
    python3 experiments/mist_w3po_v1/voicecond/idea2_teacher_table.py"""
import hashlib, json, sys
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[3]
CV = REPO / "data/open_voicecond_v1/idea2/cvt"
out = CV / "teacher_in/voice_gstar.jsonl"
if out.exists():
    sys.exit(f"refuse to overwrite {out}")
k, p = 2, CV / "refs/ustar.npz"
while (CV / f"refs/ustar_v{k}.npz").exists():
    p, k = CV / f"refs/ustar_v{k}.npz", k + 1
z = np.load(p, allow_pickle=True)
g = {str(s).replace(":", "_"): [int(x) for x in ids] for s, ids in zip(z["spk"], z["gstar"])}
rows = []
for v in sorted((CV / "teacher_in/voice").glob("*.wav")):
    rows.append({"key": v.stem, "id": v.stem, "ids": g[v.stem], "sha1": hashlib.sha1(v.read_bytes()).hexdigest()})
with open(out, "x") as fh:
    for r in rows:
        fh.write(json.dumps(r) + "\n")
print(json.dumps({"out": str(out), "speakers": len(rows), "refs": str(p)}))
