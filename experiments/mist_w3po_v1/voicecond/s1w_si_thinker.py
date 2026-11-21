"""Robust S1 "witness algebra" (reports/mist_w3po_v1/robust_s1_witness_prereg.zh-CN.md): commit rules for the
streaming Thinker, inference only. A copy of combo_si_thinker.py's la2 / w1 read loop (same prompts, same reads every
1000 ms, same unit split, greedy, base Thinker); combo_si_thinker.py is not modified. Per non-final read t, with v0's
draft d_t (the committed prefix + the greedy continuation), n committed units, and (two-view rules) v1's draft d_t^{v1}
on the DeepFilterNet3 view of the same prefix (wave15_enh_views.py, same vLLM batch, same prompt):

  la1     cut = |d_t|                                          (commit the whole current draft)
  la2     cut = LCP(d_{t-1}, d_t)                              (= si_thinker_stream la2 w1; 0 on the first read)
  la3     cut = LCP(d_{t-2}, d_{t-1}, d_t)                     (0 until three drafts exist)
  aov     cut = LCP(d_t, d_t^{v1})                             (agreement over views: no temporal witness)
  either  cut = max(LCP(d_{t-1}, d_t), LCP(d_t, d_t^{v1}))     (either witness; first read: the view witness only)

and commits d_t[n:cut] (nothing if cut <= n). The final read commits v0's whole draft (as LA2) and needs no v1 prompt.
Drafts d_{t-1}, d_{t-2} are this rule's own earlier drafts (each rule has its own committed trajectory).

Several jobs (rule:direction) run one after another on one vLLM engine:
    <vllm_omni_030 python> -m experiments.mist_w3po_v1.voicecond.s1w_si_thinker --jobs aov:zh2en,aov:en2zh \
        --out-root eval_outputs/mist_w3po_v1/s1w --data eval_outputs/mist_w3po_v1/wave2_S1/data \
        --views-root eval_outputs/mist_w3po_v1/wave15/views
Output <out-root>/<rule>/thinker_<d>/blocks_<d>_w1.jsonl (+ .s1w.json stats); every output is checked before the
engine starts and none is ever overwritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

import soundfile as sf

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from experiments.uniss_qwen3_omni_v1.evaluation.si_thinker_stream import QWEN3, THINKER_EOS, join, units  # noqa: E402
from experiments.mist_w3po_v1.voicecond.wave15_si_thinker_cvla import lcp, load_views  # noqa: E402

RULES = ("la1", "la2", "la3", "aov", "either")
TWO = ("aov", "either")


def load_items(path: Path):
    items = [{"id": r["id"], "wav": r["audio"], "dur_ms": int(r["dur_ms"])} for r in map(json.loads, open(path, encoding="utf-8"))]
    cache = {}
    for it in items:
        if it["wav"] not in cache:
            x, sr = sf.read(it["wav"], dtype="float32", always_2d=True)
            assert sr == 16000, it["wav"]
            cache = {it["wav"]: x.mean(axis=1)}
        it["audio"] = cache[it["wav"]][: it["dur_ms"] * 16]
        it["dur_ms"] = len(it["audio"]) // 16
    return items


def cont_units(committed, cont, zh):
    sep = "" if zh or not committed or cont[:1].isspace() or cont[:1] in ",.;:!?)" else " "
    return units(join(committed, zh) + sep + cont, zh)


def run_job(llm, params, head_of, rule, d, items, views_root: Path, out: Path, seed_note: dict) -> None:
    zh = d.endswith("zh")
    head = head_of[d]
    two = rule in TWO
    if two:
        load_views(views_root / d, items)
    window = 1000
    for it in items:
        it.update(committed=[], hist=[], blocks=[], drafts=[], done=False)
    st = Counter()
    t0, r = time.time(), 0
    tag = f"s1w {rule} {d}"
    while any(not it["done"] for it in items):
        live = [it for it in items if not it["done"]]
        prompts, vk = [], []
        for it in live:
            it["t"] = min((r + 1) * window, it["dur_ms"])
            prompts.append({"prompt": head + join(it["committed"], zh),
                            "multi_modal_data": {"audio": (it["audio"][: it["t"] * 16], 16000)}})
        if two:
            for k, it in enumerate(live):
                if it["t"] < it["dur_ms"]:
                    a1 = it["v1"][it["t"]]
                    assert len(a1) == it["t"] * 16
                    prompts.append({"prompt": prompts[k]["prompt"], "multi_modal_data": {"audio": (a1, 16000)}})
                    vk.append(k)
        results = llm.generate(prompts, params, use_tqdm=False)
        res1 = {k: results[len(live) + j] for j, k in enumerate(vk)}
        for k, (it, res) in enumerate(zip(live, results[: len(live)])):
            draft = cont_units(it["committed"], res.outputs[0].text, zh)
            rec = {"t": it["t"], "draft": join(draft, zh)}
            n = len(it["committed"])
            if it["t"] >= it["dur_ms"]:
                new, it["done"] = draft[n:], True
            else:
                h = it["hist"]
                l_prev = lcp(draft, h[-1]) if h else None
                if rule == "la1":
                    cut = len(draft)
                elif rule == "la2":
                    cut = l_prev if h else 0
                elif rule == "la3":
                    cut = min(l_prev, lcp(draft, h[-2])) if len(h) >= 2 else 0
                else:
                    d1 = cont_units(it["committed"], res1[k].outputs[0].text, zh)
                    l_v1 = lcp(draft, d1)
                    rec.update(draft_v1=join(d1, zh), lcp_v1=l_v1)
                    st["v1_differs"] += int(d1 != draft)
                    if rule == "aov":
                        cut = l_v1
                        if d1 == draft:
                            assert cut == len(draft)
                    else:
                        cut = max(l_v1, l_prev if h else 0)
                        st["view_witness_longer"] += int(l_v1 > (l_prev if h else 0))
                        st["time_witness_longer"] += int(bool(h) and l_prev > l_v1)
                new = draft[n: cut] if cut > n else []
                rec.update(n=n, lcp_prev=l_prev, cut=cut, new=len(new), la2_new=max(0, (l_prev or 0) - n))
                st["reads_nonfinal"] += 1
                st["committed_units"] += len(new)
                st["la2_units_same_traj"] += max(0, (l_prev or 0) - n)
                st["reads_with_commit"] += int(bool(new))
                h.append(draft)
                if len(h) > 2:
                    h.pop(0)
            it["drafts"].append(rec)
            if new or it["done"]:
                text = ("" if zh or not n else " ") + join(new, zh)
                it["blocks"].append({"write_ms": it["t"], "text": text, "final": it["done"]})
                it["committed"] += new
        r += 1
        print(f"{tag} round {r}: {len(live)} sentences, {len(vk)} v1 prompts, {time.time() - t0:.0f}s", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "x", encoding="utf-8") as fh:
        for it in items:
            blocks = [b for b in it["blocks"] if b["text"].strip() or b["final"]]
            if blocks and not blocks[-1]["text"].strip() and len(blocks) > 1:   # an empty last commit: the one before is the last
                blocks = blocks[:-1]
                blocks[-1]["final"] = True
            fh.write(json.dumps({"id": it["id"], "translation": join(it["committed"], zh), "blocks": blocks,
                                 "drafts": it["drafts"]}, ensure_ascii=False) + "\n")
    stats = dict(st)
    stats.update(rule=rule, direction=d, items=len(items), seconds=round(time.time() - t0, 1), **seed_note)
    out.with_suffix(".s1w.json").write_text(json.dumps(stats), encoding="utf-8")
    print(f"{tag}: {len(items)} sentences, {time.time() - t0:.0f}s -> {out}", flush=True)
    print("s1w stats:", json.dumps(stats), flush=True)
    for it in items:   # free the per-job state (views, drafts)
        for key in ("v1", "drafts", "blocks", "hist", "committed"):
            it.pop(key, None)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", required=True, help="comma list of rule:direction")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--views-root", required=True)
    ap.add_argument("--thinker", default=QWEN3)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=20261021)
    args = ap.parse_args()
    jobs = []
    for j in args.jobs.split(","):
        rule, d = j.split(":")
        if rule not in RULES or d not in ("zh2en", "en2zh"):
            raise SystemExit(f"bad job {j}")
        jobs.append((rule, d))
    outs = {(rule, d): Path(args.out_root) / rule / f"thinker_{d}" / f"blocks_{d}_w1.jsonl" for rule, d in jobs}
    for o in outs.values():
        if o.exists() or o.with_suffix(".s1w.json").exists():
            raise SystemExit(f"拒绝覆盖已有输出: {o}")
    data = Path(args.data)

    from transformers import Qwen3OmniMoeProcessor
    from vllm import LLM, SamplingParams
    from experiments.uniss_qwen3_omni_v1.training.e2_batch import inference_instruction

    processor = Qwen3OmniMoeProcessor.from_pretrained(QWEN3)
    head_of = {}
    for d in ("zh2en", "en2zh"):
        instruction = inference_instruction()["cmn" if d.endswith("zh") else "eng"]
        conv = [{"role": "user", "content": [{"type": "audio", "audio": "x.wav"}, {"type": "text", "text": instruction}]}]
        head_of[d] = processor.apply_chat_template([conv], add_generation_prompt=True, tokenize=False)[0]
    llm = LLM(model=args.thinker, gpu_memory_utilization=args.gpu_mem, max_model_len=8192, limit_mm_per_prompt={"audio": 1},
              seed=args.seed, tensor_parallel_size=1)
    params = SamplingParams(temperature=0.0, max_tokens=160, stop_token_ids=[THINKER_EOS])
    for rule, d in jobs:
        f = data / f"items_{d}.jsonl"
        items = load_items(f)
        note = {"thinker": args.thinker, "seed": args.seed, "views": str(Path(args.views_root) / d) if rule in TWO else "",
                "items_sha1": hashlib.sha1(open(f, "rb").read()).hexdigest()}
        run_job(llm, params, head_of, rule, d, items, Path(args.views_root), outs[(rule, d)], note)
    print("s1w all jobs done:", args.jobs, flush=True)


if __name__ == "__main__":
    main()
