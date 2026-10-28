"""results_speechonly.jsonl for one run directory (argv[1]): every row points at its speech-only wav (no waits), for ASR."""
import json, os, sys
import soundfile as sf
base = sys.argv[1]
src, out = f"{base}/results.jsonl", f"{base}/results_speechonly.jsonl"
if not os.path.exists(out):
    rows = []
    for r in map(json.loads, open(src, encoding="utf-8")):
        if r.get("speech_path") and os.path.exists(r["speech_path"]):
            r["audio_path"] = r["speech_path"]; r["audio_duration_seconds"] = round(sf.info(r["speech_path"]).duration, 3)
        rows.append(r)
    with open(out, "x", encoding="utf-8") as fh:
        fh.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
print(base)
