"""UTMOS on 120 N1 en2zh sentences: speech-only as is / trimmed / WSOLA variants at fixed rates, to find what costs quality."""
import json, os, sys
import numpy as np, soundfile as sf, torch
sys.path.insert(0, "/opt/dlami/nvme/jasonleeeli/projects/UniSS")
from experiments.uniss_qwen3_omni_v1.evaluation.si_talk_timeline import wsola, trim_edges
os.environ.setdefault("TORCH_HOME", "/opt/dlami/nvme/jasonleeeli/cache/torch")
torch.set_num_threads(64)
m = torch.hub.load("/opt/dlami/nvme/jasonleeeli/cache/torch/hub/tarepan_SpeechMOS_v1.2.0", "utmos22_strong", source="local", trust_repo=True).eval()

def wsola2(x, r, L=960, Ha=240, tol=240):   # 60 ms frames, 75% overlap, wider search
    if abs(r - 1) < 1e-3: return x
    w = np.hanning(L); n_out = int(len(x) / r)
    xp = np.pad(x, (tol, L + tol + int(Ha * r) + Ha + 1))
    nf = n_out // Ha + 1; y = np.zeros(nf * Ha + L); ws = np.zeros_like(y); delta = 0
    for k in range(nf):
        a = int(k * Ha * r) + tol + delta; seg = xp[a:a + L]
        if len(seg) < L: break
        y[k*Ha:k*Ha+L] += seg * w; ws[k*Ha:k*Ha+L] += w
        nat = xp[a + Ha:a + Ha + L]; nxt = int((k + 1) * Ha * r) + tol; region = xp[nxt - tol:nxt + tol + L]
        if len(nat) < L or len(region) < 2 * tol + L: break
        delta = int(np.argmax(np.correlate(region, nat, "valid"))) - tol
    return (y / np.maximum(ws, 1e-6))[:n_out].astype(np.float32)

E = "/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_qwen3_omni_v1/si_t1c_20261021T181440Z/full_iter0001500/system_flu-hc3-b0p5_m0.5/realsi_en2zh/results.jsonl"
rows = [json.loads(l) for l in open(E)][::3][:120]
def score(x): return float(m(torch.from_numpy(x).unsqueeze(0), 16000).item())
res = {}
for r in rows:
    x, _ = sf.read(r["audio_path"], dtype="float32")
    blocks = [x[int(b["start_ms"]*16):int((b["start_ms"]+b["dur_ms"])*16)] for b in r["blocks"]]
    variants = {"orig": np.concatenate(blocks), "trim": np.concatenate([trim_edges(b) for b in blocks])}
    for rate in (0.92, 1.1, 1.15):
        variants[f"w_{rate}"] = np.concatenate([wsola(b, rate) for b in blocks])
        variants[f"w2_{rate}"] = np.concatenate([wsola2(b, rate) for b in blocks])
    for k, v in variants.items():
        if len(v) >= 8000: res.setdefault(k, []).append(score(v))
for k, v in res.items(): print(k, len(v), round(float(np.mean(v)), 3))
