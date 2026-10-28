"""Render one acoustic scheme behind identical committed plans (tts_schemes environment).

The planner is fixed (S6-future, policy fixed:K, common random numbers), so every backend receives the same committed
durations, prior statistics and encoder states; only the acoustic side differs. Streaming rules hold for all of them:
a backend sees committed units only and never revises emitted audio.

  vits  E0 coupling flow over committed frames + HiFi-GAN with 48 frames of left context (the current system)
  s5    as vits with the decoder trained against the WavLM discriminator
  t4    causal flow + causal HiFi-GAN (prefix invariant; 48 frames of left context exceed the receptive field)
  s2    CFM vector field: each unit's latents generated after up to 96 committed context frames (Euler, --nfe)
  s3    25 Hz AudioVAE latents for low-rate frames fully inside committed time (dual clock), causal prior + flow,
        stateful streaming AudioVAE decoder at 48 kHz (resampled to 22.05 kHz for the shared evaluation)
  s4    12.5 Hz Qwen3-TTS tokens generated autoregressively for token frames that start inside committed time
        (conditions from the committed part of the frame), adapter-corrected prior, then the E0 flow + decoder
Writes <out>/commits.jsonl (durations, for the audio metrics) and <out>/wavs/.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "vits"))
sys.path.insert(0, str(ROOT / "scripts"))
from acoustic_common import CausalFlow, FRAME_S, clock_bounds, interval_overlap, token_bounds  # noqa: E402
from run_stream_eval import load_stream  # noqa: E402
from simulator import Simulator, unit_rng  # noqa: E402
from stream_data import load_cands  # noqa: E402

HOP, LEFT = 256, 48


def expand(t, dur):
  rep = torch.from_numpy(np.repeat(np.arange(len(dur)), dur)).to(t.device)
  return t[:, rep]


class Backend:
  def __init__(self, kind, sim_model, ck, dev, nfe, temp):
    self.kind, self.dev, self.nfe, self.temp = kind, dev, nfe, temp
    self.flow, self.dec = sim_model.frozen["flow"], sim_model.frozen["dec"]
    if kind == "s5":
      self.dec.load_state_dict(torch.load(ck["s5"], map_location="cpu")["dec"])
    if kind == "t4":
      from train_t4_causal import make_causal
      self.cflow = CausalFlow().to(dev).eval()
      self.cflow.load_state_dict(torch.load(ck["t4flow"], map_location="cpu")["model"])
      self.cdec = make_causal(self.dec).to(dev).eval()   # in place: the t4 backend never uses the E0 decoder
      self.cdec.load_state_dict(torch.load(ck["t4dec"], map_location="cpu")["model"])
    if kind == "s2":
      from cfm import CFMField
      self.field = CFMField().to(dev).eval()
      self.field.load_state_dict(torch.load(ck["s2"], map_location="cpu")["model"])
    if kind == "s3":
      from audiovae import load_audiovae
      from codec_priors import S3Prior
      self.vae = load_audiovae(dev)
      self.prior = S3Prior().to(dev).eval()
      self.prior.load_state_dict(torch.load(ck["s3"], map_location="cpu")["prior"])
    if kind == "s4":
      from codec_priors import TokenAdapter, TokenPredictor
      st = torch.load(ck["s4"], map_location="cpu")
      self.pred, self.adapt = TokenPredictor().to(dev).eval(), TokenAdapter().to(dev).eval()
      self.pred.load_state_dict(st["pred"])
      self.adapt.load_state_dict(st["adapt"])

  @torch.no_grad()
  def render(self, sim, plans):
    k = self.kind
    if k in ("vits", "s5", "t4"):
      return self._flow_dec(sim, [None] * len(plans), plans)
    if k == "s2":
      return self._s2(sim, plans)
    if k == "s3":
      return self._s3(sim, plans)
    return self._s4(sim, plans)

  def _decode_new(self, dec, z_done, n_new):
    s = max(0, z_done.size(1) - n_new - LEFT)
    o = dec(z_done[None, :, s:])[0, 0]
    return o[(z_done.size(1) - n_new - s) * HOP:]

  def _flow_dec(self, sim, stats, plans):
    """Prior sample per unit (eps fixed per unit as in Simulator.commit) -> flow inverse over committed frames."""
    causal = self.kind == "t4"
    flow = self.cflow if causal else self.flow
    dec = self.cdec if causal else self.dec
    zp_all = torch.zeros(192, 0, device=self.dev)
    z_done = torch.zeros(192, 0, device=self.dev)
    pieces = []
    for i, (p, st) in enumerate(zip(plans, stats)):
      m, ls = st if st is not None else (expand(p["m_p"], p["dur"]), expand(p["logs_p"], p["dur"]))
      if m.size(1) == 0:
        continue
      eps = torch.randn(m.shape, generator=unit_rng(sim.seed, sim.uid, i, 2)).to(self.dev)
      zp_all = torch.cat([zp_all, m + eps * torch.exp(ls) * sim.noise_z], 1)
      mask = torch.ones(1, 1, zp_all.size(1), device=self.dev)
      z = flow(zp_all[None], mask, reverse=True)[0]
      z_new = z[:, z_done.size(1):]
      z_done = torch.cat([z_done, z_new], 1)
      pieces.append(self._decode_new(dec, z_done, z_new.size(1)))
    return torch.cat(pieces).float().cpu().numpy()

  def _s2(self, sim, plans):
    z_done = torch.zeros(192, 0, device=self.dev)
    c_done = torch.zeros(384, 0, device=self.dev)
    pieces = []
    for i, p in enumerate(plans):
      c = torch.cat([expand(p["h"], p["dur"]), expand(p["m_p"], p["dur"])], 0)
      if c.size(1) == 0:
        continue
      z0 = torch.randn(c.size(1), 192, generator=unit_rng(sim.seed, sim.uid, i, 2)).to(self.dev)
      ctx = slice(max(0, z_done.size(1) - 96), z_done.size(1))
      z = self.field.generate(z_done[:, ctx].T, c_done[:, ctx].T, c.T, z0, self.nfe).T
      z_done = torch.cat([z_done, z], 1)
      c_done = torch.cat([c_done, c], 1)
      pieces.append(self._decode_new(self.dec, z_done, z.size(1)))
    return torch.cat(pieces).float().cpu().numpy()

  def _s3(self, sim, plans):
    import torchaudio
    from audiovae import LAT_HZ, OUT_SR
    tok_c, durs = [], []
    u_done = torch.zeros(64, 0, device=self.dev)
    out = []
    with self.vae.streaming_decode() as sd:
      for i, p in enumerate(plans):
        tok_c.append(torch.cat([p["h"], p["m_p"]], 0))
        durs.extend(p["dur"].tolist())
        t_end = sum(durs) * FRAME_S
        last = i == len(plans) - 1
        n_have = u_done.size(1)
        n_need = int(np.ceil(t_end * LAT_HZ - 1e-9)) if last else int(np.floor(t_end * LAT_HZ + 1e-9))
        if n_need <= n_have:
          continue
        o = torch.from_numpy(interval_overlap(clock_bounds(n_need, LAT_HZ), token_bounds(np.array(durs)))).to(self.dev)
        c = (o @ torch.cat(tok_c, 1).T).T[None]                         # [1, 384, n_need]
        mask = torch.ones(1, 1, n_need, device=self.dev)
        eps = torch.stack([torch.randn(64, generator=unit_rng(sim.seed, sim.uid, f, 6)) for f in range(n_need)], 1)
        u = self.prior.sample(c, mask, eps[None].to(self.dev), self.temp)[0]
        u_new = u[:, n_have:]                                           # causal prior: earlier frames unchanged
        u_done = torch.cat([u_done, u_new], 1)
        out.append(sd.decode_chunk(u_new[None])[0, 0])
    y = torch.cat(out).float()
    return torchaudio.functional.resample(y, OUT_SR, 22050).cpu().numpy()

  def _s4(self, sim, plans):
    from codec_priors import TokenPredictor
    tok_c, durs, a_done = [], [], torch.zeros(0, 16, dtype=torch.long, device=self.dev)
    stats = []
    f_done = 0
    for i, p in enumerate(plans):
      tok_c.append(torch.cat([p["h"], p["m_p"]], 0))
      durs.extend(p["dur"].tolist())
      tb = token_bounds(np.array(durs))
      t_end = tb[-1]
      n_tok = int(np.ceil(t_end * 12.5 - 1e-9))                      # token frames starting inside committed time
      c_all = torch.cat(tok_c, 1).T
      o_tok = torch.from_numpy(interval_overlap(clock_bounds(n_tok, 12.5), tb)).to(self.dev)
      cond = (o_tok @ c_all)[None]                                     # committed part of each token frame
      for t in range(a_done.size(0), n_tok):
        prev = TokenPredictor.shift(torch.cat([a_done, a_done.new_zeros(1, 16)], 0)[None], self.pred.vocab)
        logits = self.pred(cond[:, :t + 1], prev[:, :t + 1])[0, t] / self.temp
        g = unit_rng(sim.seed, sim.uid, t, 7)
        a_t = torch.multinomial(torch.softmax(logits.float(), -1).cpu(), 1, generator=g)[:, 0].to(self.dev)
        a_done = torch.cat([a_done, a_t[None]], 0)
      n_f = int(sum(durs))
      o_frm = torch.from_numpy(interval_overlap(clock_bounds(n_f, 1 / FRAME_S), clock_bounds(n_tok, 12.5))).to(self.dev)
      h_f = expand(torch.cat([q["h"] for q in plans[:i + 1]], 1), np.array(durs))
      m_f = expand(torch.cat([q["m_p"] for q in plans[:i + 1]], 1), np.array(durs))
      l_f = expand(torch.cat([q["logs_p"] for q in plans[:i + 1]], 1), np.array(durs))
      m2, l2 = self.adapt(a_done[None], o_frm[None], h_f[None], m_f[None], l_f[None])
      stats.append((m2[0, :, f_done:n_f], l2[0, :, f_done:n_f]))
      f_done = n_f
    return self._flow_dec(sim, stats, plans)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--backend", required=True, choices=["vits", "s5", "t4", "s2", "s3", "s4"])
  ap.add_argument("--model", required=True, help="planner (S6-future) checkpoint")
  ap.add_argument("--ckpt", default="{}", help='json: {"s5": path, "t4flow": path, "t4dec": path, "s2": .., "s3": .., "s4": ..}')
  ap.add_argument("--backbone-config", required=True)
  ap.add_argument("--backbone-ckpt", required=True)
  ap.add_argument("--k", type=int, default=1)
  ap.add_argument("--split", default="test")
  ap.add_argument("--limit", type=int, default=0)
  ap.add_argument("--shard", type=int, default=0)
  ap.add_argument("--num-shards", type=int, default=1)
  ap.add_argument("--noise-w", type=float, default=0.8)
  ap.add_argument("--nfe", type=int, default=16)
  ap.add_argument("--temp", type=float, default=0.667)
  ap.add_argument("--out", required=True)
  args = ap.parse_args()
  dev = torch.device("cuda")
  out = Path(args.out)
  (out / "wavs").mkdir(parents=True, exist_ok=True)
  res = out / (f"commits.shard{args.shard}.jsonl" if args.num_shards > 1 else "commits.jsonl")
  if res.exists():
    raise FileExistsError(res)
  model = load_stream(args.model, args.backbone_config, args.backbone_ckpt, dev)
  be = Backend(args.backend, model, json.loads(args.ckpt), dev, args.nfe, args.temp)
  cands = load_cands(sorted(Path(ROOT / "data/future_cands").glob("shard*.jsonl")))
  rows = [r for r in map(json.loads, open(ROOT / "data/manifest_v2.jsonl")) if r["split"] == args.split]
  rows = (rows[:args.limit] if args.limit else rows)[args.shard::args.num_shards]
  with open(res, "w") as f:
    for r in rows:
      qz = np.load(ROOT / f"data/qwen_prefix/{r['utt_id']}.npz")
      sim = Simulator(model, r, dev, qwen=qz if model.use_qwen else None, cands=cands, noise_w=args.noise_w)
      plans, units = [], []
      t0 = time.time()
      for i in range(sim.U):
        j = min(i + args.k, sim.U - 1)
        p = sim.plan(i, j)
        sim.commit(i, j, p)
        plans.append(p)
        units.append({"i": i, "j": j, "y": p["y"].tolist(), "dur": p["dur"].tolist()})
      t_plan = time.time() - t0
      wav = be.render(sim, plans)
      torch.cuda.synchronize()
      t_ac = time.time() - t0 - t_plan
      sf.write(out / "wavs" / f"{r['utt_id']}.wav", wav, 22050)
      f.write(json.dumps({"utt_id": r["utt_id"], "units": units, "plan_s": t_plan, "acoustic_s": t_ac,
                          "audio_s": len(wav) / 22050}) + "\n")
      f.flush()
  print(f"done {len(rows)} -> {res}", flush=True)


if __name__ == "__main__":
  main()
