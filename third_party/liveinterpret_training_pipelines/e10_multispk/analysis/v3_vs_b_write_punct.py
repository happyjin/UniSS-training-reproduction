import json, glob, re
from collections import Counter, defaultdict
U = "/opt/dlami/nvme/jasonleeeli/projects/UniSS"; E = U + "/third_party/liveinterpret_training_pipelines/e10_multispk/output"
WIN = {}
for d in ("zh2en", "en2zh"):
    for w in map(json.loads, open(f"{U}/eval_outputs/mist_w3po_v1/data/windows_game_{d}_c2g06_v1/windows_test_online_{d}.jsonl")): WIN[w["id"]] = w
def kind(t):
    t = t.strip()
    return "sent" if re.search(r"[.?!。？！…]['\"”’]?$", t) else ("comma" if re.search(r"[,，、;；:：]$", t) else "none")
C = defaultdict(Counter); GAPW = defaultdict(list)
for d in ("zh2en", "en2zh"):
    for f in glob.glob(f"{E}/demo_game_v3/raw_{d}/part_*.jsonl"):
        for r in map(json.loads, open(f)):
            steps = defaultdict(str)
            for g in r.get("segs") or []:
                if g["text"].strip(): steps[g["write_ms"]] += " " + g["text"]
            for t in steps.values(): C[d, "V3"][kind(t)] += 1
    for r in map(json.loads, open(f"{U}/eval_outputs/mist_w3po_v1/speech/game_b_{d}__t1c/talker_in/blocks.jsonl")):
        steps = defaultdict(str)
        for b in r["blocks"]:
            if b["text"].strip(): steps[b["write_ms"]] += " " + b["text"]
        for t in steps.values(): C[d, "B"][kind(t)] += 1
for k, c in sorted(C.items()):
    n = sum(c.values()); print(k, n, {x: round(c[x] / n, 2) for x in ("sent", "comma", "none")})
