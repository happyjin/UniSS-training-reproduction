"""UTMOS22 (CPU) on each run's speech-only wavs: name=run_dir ... -> out jsonl (id, run, utmos)."""
import json, os, sys
import numpy as np, soundfile as sf, torch, librosa
os.environ.setdefault("TORCH_HOME", "/opt/dlami/nvme/jasonleeeli/cache/torch")
torch.set_num_threads(int(os.environ.get("UTMOS_THREADS", "16")))
out, specs = sys.argv[1], sys.argv[2:]
limit = int(os.environ.get("UTMOS_LIMIT", "0"))
m = torch.hub.load("/opt/dlami/nvme/jasonleeeli/cache/torch/hub/tarepan_SpeechMOS_v1.2.0", "utmos22_strong", source="local", trust_repo=True).eval()
with open(out, "x") as fh:
    for spec in specs:
        name, d = spec.split("=", 1)
        rows = [json.loads(l) for l in open(os.path.join(d, "results.jsonl"))]
        for r in rows[: limit or None]:
            x, sr = sf.read(r["speech_path"], dtype="float32", always_2d=True)
            x = x.mean(1)
            if sr != 16000:
                x = librosa.resample(x, orig_sr=sr, target_sr=16000)
            if len(x) < 8000:
                continue
            with torch.no_grad():
                s = float(m(torch.from_numpy(x).unsqueeze(0), 16000).item())
            fh.write(json.dumps({"run": name, "id": r["id"], "utmos": s, "dur": len(x) / 16000}) + "\n")
            fh.flush()
print("done", flush=True)
