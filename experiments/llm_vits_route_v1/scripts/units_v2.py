"""Rewrite manifest units to the leading-space convention (text.split_units); output manifest_v2.jsonl."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from text import split_units  # noqa: E402

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
if dst.exists():
  raise FileExistsError(f"refusing to overwrite {dst}")
with open(dst, "w") as f:
  for line in open(src):
    r = json.loads(line)
    old = r["units"]
    r["units"] = split_units(r["clean"])
    assert len(old) == len(r["units"]) and all(a[0] <= b[1] for a, b in zip(old, r["units"]))
    assert [r["clean"][:b[1]].rstrip(" ") for b in r["units"]] == [r["clean"][:a[1]].rstrip(" ") for a in old]
    f.write(json.dumps(r) + "\n")
