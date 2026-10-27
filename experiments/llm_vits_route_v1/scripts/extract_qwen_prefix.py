"""Frozen Qwen3-1.7B prefix states for every utterance (handbook G2, E2).

Qwen's pre-tokenizer never merges across a word boundary, so the state of a token that ends inside unit j is
identical whether the model saw the full sentence or only units 1..j: one causal forward pass serves every prefix.
A unit's trailing space is the leading space of the next word's token, so a space char that starts a token is
mapped to the previous token (never to a future word).

Per utterance (npz): h_mid/h_last [n_tok, 2048] fp16, char_tok [n_chars] int, unit_last_tok [n_units] int,
unit_entropy [n_units] (next-token entropy after unit j, nats), unit_p_punct [n_units] (next token is punctuation).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--manifest", required=True)
  ap.add_argument("--model", required=True)
  ap.add_argument("--out", required=True)
  ap.add_argument("--mid-layer", type=int, default=14)
  ap.add_argument("--batch", type=int, default=64)
  args = ap.parse_args()
  out = Path(args.out)
  out.mkdir(parents=True, exist_ok=True)
  tok = AutoTokenizer.from_pretrained(args.model)
  model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).cuda().eval()
  vocab = tok.convert_ids_to_tokens(list(range(len(tok))))
  punct = torch.zeros(model.config.vocab_size, dtype=torch.bool)
  for i, t in enumerate(vocab):
    s = tok.convert_tokens_to_string([t]).strip() if t is not None else ""
    if s and all(c in ",.;:!?" for c in s):
      punct[i] = True
  punct = punct.cuda()
  rows = [json.loads(l) for l in open(args.manifest)]
  rows = [r for r in rows if not (out / f"{r['utt_id']}.npz").exists()]
  rows.sort(key=lambda r: len(r["clean"]))
  for b in range(0, len(rows), args.batch):
    chunk = rows[b:b + args.batch]
    enc = tok([r["clean"] for r in chunk], return_offsets_mapping=True, padding=True, return_tensors="pt")
    with torch.no_grad():
      o = model(input_ids=enc["input_ids"].cuda(), attention_mask=enc["attention_mask"].cuda(), output_hidden_states=True)
    logp = torch.log_softmax(o.logits.float(), -1)
    ent = -(logp.exp() * logp).sum(-1)
    p_punct = (logp.exp() * punct.float()).sum(-1)
    for k, r in enumerate(chunk):
      m = enc["attention_mask"][k].bool()
      idx = torch.nonzero(m).squeeze(1)   # padding side may be left or right
      offs = enc["offset_mapping"][k][idx].tolist()
      n_chars = len(r["clean"])
      char_tok = np.full(n_chars, -1, dtype=np.int32)
      for t, (s, e) in enumerate(offs):
        for c in range(s, e):
          char_tok[c] = t
      for c in range(n_chars):
        t = char_tok[c]
        if r["clean"][c] == " " and t > 0 and offs[t][0] == c:
          char_tok[c] = t - 1
      assert (char_tok >= 0).all(), r["utt_id"]
      unit_last = np.array([char_tok[e - 1 if r["clean"][e - 1] != " " else e - 2] for s, e in r["units"]], dtype=np.int32)
      for j, (s, e) in enumerate(r["units"]):   # causality: a unit's last token must not extend past the unit
        assert offs[unit_last[j]][1] <= e, (r["utt_id"], j)
      np.savez(out / f"{r['utt_id']}.npz",
               h_mid=o.hidden_states[args.mid_layer][k][idx].to(torch.float16).cpu().numpy(),
               h_last=o.hidden_states[-1][k][idx].to(torch.float16).cpu().numpy(),
               char_tok=char_tok, unit_last_tok=unit_last,
               unit_entropy=ent[k][idx][unit_last].cpu().numpy().astype(np.float32),
               unit_p_punct=p_punct[k][idx][unit_last].cpu().numpy().astype(np.float32))
    print(f"{b + len(chunk)}/{len(rows)}", flush=True)


if __name__ == "__main__":
  main()
