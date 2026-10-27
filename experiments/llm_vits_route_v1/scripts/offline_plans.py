"""E0 offline reference: full-text VITS durations (SDP over the whole sentence), recorded per unit like a commit log."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "vits"))
from simulator import unit_rng  # noqa: E402
from stream_common import load_backbone, prosody_y, token_is_word, unit_token_spans  # noqa: E402
from text import intersperse, text_to_ids  # noqa: E402


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--backbone-config", required=True)
  ap.add_argument("--backbone-ckpt", required=True)
  ap.add_argument("--manifest", default=str(ROOT / "data/manifest_v2.jsonl"))
  ap.add_argument("--split", default="test")
  ap.add_argument("--noise-w", type=float, default=0.0)
  ap.add_argument("--noise-z", type=float, default=0.667)
  ap.add_argument("--render", action="store_true")
  ap.add_argument("--out", required=True)
  args = ap.parse_args()
  out = Path(args.out)
  out.mkdir(parents=True, exist_ok=True)
  if (out / "commits.jsonl").exists():
    raise FileExistsError(out / "commits.jsonl")
  if args.render:
    (out / "wavs").mkdir(exist_ok=True)
  net = load_backbone(args.backbone_config, args.backbone_ckpt).cuda().eval()
  rows = [json.loads(l) for l in open(args.manifest) if json.loads(l)["split"] == args.split]
  with open(out / "commits.jsonl", "w") as f, torch.no_grad():
    for r in rows:
      ids = intersperse(text_to_ids(r["clean"]))
      x = torch.LongTensor(ids)[None].cuda()
      h, m_p, logs_p, x_mask = net.enc_p(x, torch.LongTensor([len(ids)]).cuda())
      noise = torch.randn(1, 2, len(ids), generator=unit_rng(1234, r["utt_id"], -1, 1)).cuda() * args.noise_w
      z = noise
      flows = list(reversed(net.dp.flows))
      flows = flows[:-2] + [flows[-1]]
      xd = net.dp.proj(net.dp.convs(net.dp.pre(h), x_mask)) * x_mask
      for fl in flows:
        z = fl(z, x_mask, g=xd, reverse=True)
      dur = torch.ceil(torch.exp(z[:, :1]) * x_mask)[0, 0].long().cpu().numpy()
      isw = token_is_word(r["clean"])
      units = []
      for i, (a, b) in enumerate(unit_token_spans(r)):
        w = dur[a:b]
        y = prosody_y([[float(w[isw[a:b]].sum()), float(w[~isw[a:b]].sum())]])[0]
        units.append({"i": i, "j": len(unit_token_spans(r)) - 1, "y": y.tolist(), "dur": w.tolist()})
      f.write(json.dumps({"utt_id": r["utt_id"], "units": units, "offline": True}) + "\n")
      if args.render:
        rep = torch.from_numpy(np.repeat(np.arange(len(dur)), dur)).cuda()
        mf, lf = m_p[0][:, rep], logs_p[0][:, rep]
        eps = torch.randn(mf.shape, generator=unit_rng(1234, r["utt_id"], -1, 2)).cuda()
        zp = (mf + eps * torch.exp(lf) * args.noise_z)[None]
        z = net.flow(zp, torch.ones(1, 1, zp.size(-1), device=zp.device), reverse=True)
        sf.write(out / "wavs" / f"{r['utt_id']}.wav", net.dec(z)[0, 0].float().cpu().numpy(), 22050)
  print("done", len(rows))


if __name__ == "__main__":
  main()
