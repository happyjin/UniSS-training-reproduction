"""Speech transcripts -> hypothesis lines (reports/gstar_si_v7/after_v7.sh): per window and output speaker, the
speech-only ASR of that speaker's stream as one segment spanning its segments. Refuses an existing output.

    python -m experiments.mist_w3po_v1.evaluation.asr_hyp <hyp.jsonl> <asr_speech.jsonl> <hyp_asr.jsonl>
"""
import json
import sys
from collections import defaultdict

hyp, asr, out = sys.argv[1:4]
A = {r["id"]: r.get("asr_text", "") for r in map(json.loads, open(asr, encoding="utf-8"))}
with open(out, "x", encoding="utf-8") as fh:
    for h in map(json.loads, open(hyp, encoding="utf-8")):
        by = defaultdict(list)
        for s in h["segments"]:
            by[s["spk"]].append(s)
        segs = []
        for k, ss in by.items():
            t = A.get(f"{h['id']}__S{k}", "").strip()
            if t:
                segs.append({"spk": k, "start_s": min(s["start_s"] for s in ss), "end_s": max(s["end_s"] for s in ss),
                             "text": t, "write_ms": max(s.get("write_ms", 0) for s in ss)})
        fh.write(json.dumps({"id": h["id"], "segments": segs}, ensure_ascii=False) + "\n")
