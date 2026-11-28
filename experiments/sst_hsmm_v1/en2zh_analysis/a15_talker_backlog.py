"""overnight 2026-11-28 diagnosis (report only): is the talker still speaking when a block is committed?
for every talker block (new/talker_<d>/part_*.jsonl, fields write_ms / start_ms / dur_ms), backlog = unspoken speech already queued at
write time = max(0, end of previous speech - write_ms). Early (non-final) units are split into idle (backlog < 0.25 s) / busy
(0.25-1 s) / long (>= 1 s, i.e. more than one read interval: the unit could have waited for the next read at no speech cost).
for each class: share of early units, and clipped unigram precision of that text vs the offline ceiling (wave17 whole audio).
    python3 experiments/sst_hsmm_v1/en2zh_analysis/a15_talker_backlog.py <sys> [<sys> ...]
"""
import ast, collections, glob, json, re, sys
E = "eval_outputs/sst_hsmm_v1"
for d in ("en2zh", "zh2en"):
    U = (lambda s: [c for c in s if not c.isspace() and c not in "。！？，、,.!?;；:：\"“”"]) if d == "en2zh" else (lambda s: re.findall(r"[\w'’-]+", s.lower()))
    ceil = {}
    for f in glob.glob(f"{E}/ceiling/offline_{d}_*.jsonl"):
        for l in open(f, encoding="utf-8"):
            r = json.loads(l); ceil[r["id"]] = collections.Counter(U(r["translation"]))
    for s in sys.argv[1:]:
        st = collections.Counter()
        for f in glob.glob(f"{E}/{s}/new/talker_{d}/part_*.jsonl"):
            for l in open(f, encoding="utf-8"):
                r = json.loads(l)
                if r["id"] not in ceil: continue
                B = ast.literal_eval(r["blocks"]) if isinstance(r["blocks"], str) else r["blocks"]
                end = 0; bag = {c: collections.Counter() for c in ("idle", "busy", "long")}
                for b in B:
                    u = U(b["text"])
                    if not b["final"] and u:
                        bl = max(0, end - b["write_ms"]) / 1000
                        c = "idle" if bl < 0.25 else ("busy" if bl < 1.0 else "long")
                        bag[c].update(u); st[c] += len(u)
                    st["all"] += len(u)
                    if b.get("start_ms") is not None and b.get("dur_ms") is not None:
                        end = max(end, b["start_ms"] + b["dur_ms"])
                for c, h in bag.items():
                    st[c + "_m"] += sum(min(v, ceil[r["id"]][g]) for g, v in h.items())
        early = st["idle"] + st["busy"] + st["long"]
        print(f"{d} {s:7s} early share {early/st['all']:.3f} | " + " | ".join(
            f"{c} {st[c]/max(1,early):.3f} of early, prec vs offline {100*st[c+'_m']/max(1,st[c]):5.1f}" for c in ("idle", "busy", "long")))
