"""Sample M future continuations of every unit prefix from frozen Qwen3-1.7B (handbook 1.2, S1/S6 future r, T2).

The decoding kernel is explicit (temperature, top_p, max_tokens, seed are written into the output), so the
candidates are independent draws from a declared deployment process, weighted 1/M. Candidates are for planning
only and are never spoken. Output jsonl rows: {utt_id, unit, cands: [M strings]}.
"""
import argparse
import json
from pathlib import Path

from vllm import LLM, SamplingParams


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--manifest", required=True)
  ap.add_argument("--model", required=True)
  ap.add_argument("--out", required=True)
  ap.add_argument("--splits", default="train,val,calib,test")
  ap.add_argument("--m", type=int, default=8)
  ap.add_argument("--max-tokens", type=int, default=12)
  ap.add_argument("--temperature", type=float, default=1.0)
  ap.add_argument("--top-p", type=float, default=1.0)
  ap.add_argument("--seed", type=int, default=17)
  ap.add_argument("--shard", type=int, default=0)
  ap.add_argument("--num-shards", type=int, default=1)
  ap.add_argument("--gpu-mem", type=float, default=0.35)
  args = ap.parse_args()
  out = Path(args.out)
  if out.exists():
    raise FileExistsError(f"refusing to overwrite {out}")
  splits = set(args.splits.split(","))
  rows = [json.loads(l) for l in open(args.manifest)]
  rows = [r for i, r in enumerate(rows) if r["split"] in splits and i % args.num_shards == args.shard]
  keys, prompts = [], []
  for r in rows:
    for j, (s, e) in enumerate(r["units"]):
      keys.append((r["utt_id"], j))
      prompts.append(r["clean"][:e].rstrip(" "))
  llm = LLM(model=args.model, dtype="bfloat16", gpu_memory_utilization=args.gpu_mem, seed=args.seed,
            enable_prefix_caching=True, max_model_len=512)
  sp = SamplingParams(n=args.m, temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens,
                      seed=args.seed)
  res = llm.generate(prompts, sp)
  with open(out, "w") as f:
    f.write(json.dumps({"meta": {"model": args.model, "m": args.m, "temperature": args.temperature, "top_p": args.top_p,
                                 "max_tokens": args.max_tokens, "seed": args.seed}}) + "\n")
    for (uid, j), r in zip(keys, res):
      f.write(json.dumps({"utt_id": uid, "unit": j, "cands": [c.text for c in r.outputs]}) + "\n")
  print(f"wrote {len(keys)} prefixes to {out}", flush=True)


if __name__ == "__main__":
  main()
