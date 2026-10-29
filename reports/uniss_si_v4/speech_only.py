"""results_speechonly.jsonl beside each si_infer results.jsonl given (the run's realsi_<dir> or cvss_<dir> folders): the
speech segments without the silence of the timeline, which the recognisers hallucinate on (v3's p8_prep.py, kept here).

    python reports/uniss_si_v4/speech_only.py <folder with results.jsonl> ...
"""
import json, os, sys

import soundfile as sf

for base in sys.argv[1:]:
    src, out = f"{base}/results.jsonl", f"{base}/results_speechonly.jsonl"
    if not os.path.exists(src):
        parts = sorted(f for f in os.listdir(base) if f.startswith("part_") and f.endswith(".jsonl"))
        with open(src, "x", encoding="utf-8") as fh:
            for p in parts:
                fh.writelines(open(f"{base}/{p}", encoding="utf-8"))
    if os.path.exists(out):
        continue
    rows = []
    for r in map(json.loads, open(src, encoding="utf-8")):
        if r.get("speech_path") and os.path.exists(r["speech_path"]):
            r["audio_path"] = r["speech_path"]
            r["audio_duration_seconds"] = round(sf.info(r["speech_path"]).duration, 3)
        rows.append(r)
    with open(out, "x", encoding="utf-8") as fh:
        fh.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    print(base, len(rows))
