import json, os, sys
import numpy as np, soundfile as sf, torch
os.environ.setdefault("TORCH_HOME", "/opt/dlami/nvme/jasonleeeli/cache/torch")
torch.set_num_threads(64)
m = torch.hub.load("/opt/dlami/nvme/jasonleeeli/cache/torch/hub/tarepan_SpeechMOS_v1.2.0", "utmos22_strong", source="local", trust_repo=True).eval()

def wsola(x, r, L, Ha, tol, ncc):
    if len(x) < 2000: return x
    w = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(L) / L)   # periodic Hann: sums to a constant at 50%/75% overlap
    n_out = int(round(len(x) / r))
    xp = np.pad(x, (tol, L + tol + int(Ha * r) + Ha + 1))
    nf = n_out // Ha + 1; y = np.zeros(nf * Ha + L); ws = np.zeros_like(y); delta = 0
    for k in range(nf):
        a = int(round(k * Ha * r)) + tol + delta; seg = xp[a:a + L]
        if len(seg) < L: break
        y[k*Ha:k*Ha+L] += seg * w; ws[k*Ha:k*Ha+L] += w
        nat = xp[a + Ha:a + Ha + L]; nxt = int(round((k + 1) * Ha * r)) + tol; region = xp[nxt - tol:nxt + tol + L]
        if len(nat) < L or len(region) < 2 * tol + L: break
        c = np.correlate(region, nat, "valid")
        if ncc:
            e = np.sqrt(np.convolve(region ** 2, np.ones(L), "valid") + 1e-9); c = c / e
        delta = int(np.argmax(c)) - tol
    return (y / np.maximum(ws, 1e-3))[:n_out].astype(np.float32)

E = "/opt/dlami/nvme/jasonleeeli/projects/UniSS/eval_outputs/uniss_qwen3_omni_v1/si_t1c_20261021T181440Z/full_iter0001500/system_flu-hc3-b0p5_m0.5/realsi_en2zh/results.jsonl"
rows = [json.loads(l) for l in open(E)][::3][:60]
def score(x): return float(m(torch.from_numpy(x).unsqueeze(0), 16000).item())
res = {}
cfgs = {"A20ms50": (320, 160, 80, True), "B40ms75": (640, 160, 160, True), "C30ms50ncc": (480, 240, 80, True), "D20ms75": (320, 80, 80, True)}
for r in rows:
    x, _ = sf.read(r["audio_path"], dtype="float32")
    blocks = [x[int(b["start_ms"]*16):int((b["start_ms"]+b["dur_ms"])*16)] for b in r["blocks"]]
    v = {"orig": np.concatenate(blocks)}
    for name, (L, Ha, tol, ncc) in cfgs.items():
        for rate in (1.0, 1.1):
            v[f"{name}_{rate}"] = np.concatenate([wsola(b, rate, L, Ha, tol, ncc) for b in blocks])
    for k, a in v.items():
        if len(a) >= 8000: res.setdefault(k, []).append(score(a))
for k, a in res.items(): print(k, round(float(np.mean(a)), 3))
