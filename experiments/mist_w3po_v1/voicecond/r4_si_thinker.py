"""Robust R4: invariance-witness routing between the base Thinker and a robust adapter (reports/mist_w3po_v1/
robust_r4_route_prereg.zh-CN.md; plan robust_analysis_and_ideas_v3_20261120.zh-CN.md section 3 R4).

LA2 w1 commit rule unchanged (si_thinker_stream la2 w1 / combo_si_thinker --second-view none): commit LCP(d_{t-1}, d_t)
of the raw-view drafts. Per item and read t, the draft model is chosen by the routing signal of read t-1:
  read t-1: the chosen model also drafted the DeepFilterNet3 view v1 (wave15_enh_views, causal, same batch);
            v1 draft == v0 draft (views agree)  -> read t drafts with the BASE model;
            views disagree                       -> read t drafts with the ADAPTER model.
  read 1: no signal -> BASE (pre-registered: protect clean inputs; reads 1 never commit under LA2).
The final read (t = duration) drafts with the model chosen by read t-1's signal; no v1 draft there.
Models: two vLLM engines in two spawned worker processes, one per card (--cards A,B: A = base, B = adapter merged copy);
each worker sets CUDA_VISIBLE_DEVICES before importing vllm. --route base|adapter forces one model (self-check: base
must reproduce LA2 base up to vLLM nondeterminism; no v1 prompts then).
Output <out>/blocks_<dir>_w1.jsonl (rows: id, translation, blocks, drafts with per-read model/agree) + .r4.json stats.

    <vllm_omni_030 python> -m experiments.mist_w3po_v1.voicecond.r4_si_thinker --adapter <merged dir> --cards 1,7 \
        --views <dir> --direction en2zh --items-file <jsonl> --out <dir> [--route witness|base|adapter]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import sys
import time
from collections import Counter
from pathlib import Path

import soundfile as sf

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))


def worker(card: str, model: str, gpu_mem: float, seed: int, conn) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = card
    from vllm import LLM, SamplingParams
    from experiments.uniss_qwen3_omni_v1.evaluation.si_thinker_stream import THINKER_EOS
    llm = LLM(model=model, gpu_memory_utilization=gpu_mem, max_model_len=8192, limit_mm_per_prompt={"audio": 1},
              seed=seed, tensor_parallel_size=1)
    params = SamplingParams(temperature=0.0, max_tokens=160, stop_token_ids=[THINKER_EOS])
    conn.send("ready")
    while True:
        prompts = conn.recv()
        if prompts is None:
            break
        res = llm.generate(prompts, params, use_tqdm=False) if prompts else []
        conn.send([r.outputs[0].text for r in res])


def main() -> None:
    from experiments.uniss_qwen3_omni_v1.evaluation.si_thinker_stream import QWEN3, join, units
    from experiments.mist_w3po_v1.voicecond.wave15_si_thinker_cvla import lcp, load_views

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", required=True, help="merged robust model dir")
    ap.add_argument("--cards", required=True, help="A,B: base card, adapter card")
    ap.add_argument("--route", choices=("witness", "base", "adapter"), default="witness")
    ap.add_argument("--views", default="")
    ap.add_argument("--direction", choices=("zh2en", "en2zh"), required=True)
    ap.add_argument("--items-file", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=20261021)
    args = ap.parse_args()
    out = Path(args.out) / f"blocks_{args.direction}_w1.jsonl"
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    if args.route == "witness" and not args.views:
        raise SystemExit("--route witness needs --views")
    Path(args.out).mkdir(parents=True, exist_ok=True)
    zh = args.direction.endswith("zh")
    tgt = "cmn" if zh else "eng"
    items = [{"id": r["id"], "wav": r["audio"], "dur_ms": int(r["dur_ms"])}
             for r in map(json.loads, open(args.items_file, encoding="utf-8"))]
    cache = {}
    for it in items:
        if it["wav"] not in cache:
            x, sr = sf.read(it["wav"], dtype="float32", always_2d=True)
            assert sr == 16000, it["wav"]
            cache = {it["wav"]: x.mean(axis=1)}
        it["audio"] = cache[it["wav"]][: it["dur_ms"] * 16]
        it["dur_ms"] = len(it["audio"]) // 16
    if args.route == "witness":       # --views may list several view roots (comma); each item must be in exactly one
        left = list(items)
        for vd in args.views.split(","):
            ids = {json.loads(l)["id"] for l in open(Path(vd) / "index.jsonl", encoding="utf-8")}
            sub = [it for it in left if it["id"] in ids]
            load_views(Path(vd), sub)
            left = [it for it in left if it["id"] not in ids]
        assert not left, f"{len(left)} items without views, e.g. {left[0]['id']}"

    from transformers import Qwen3OmniMoeProcessor
    from experiments.uniss_qwen3_omni_v1.training.e2_batch import inference_instruction
    processor = Qwen3OmniMoeProcessor.from_pretrained(QWEN3)
    instruction = inference_instruction()[tgt]
    conv = [{"role": "user", "content": [{"type": "audio", "audio": "x.wav"}, {"type": "text", "text": instruction}]}]
    head = processor.apply_chat_template([conv], add_generation_prompt=True, tokenize=False)[0]

    ca, cb = args.cards.split(",")
    need = {"base": ["base"], "adapter": ["adapter"], "witness": ["base", "adapter"]}[args.route]
    ctx = mp.get_context("spawn")
    W = {}
    for name in need:
        parent, child = ctx.Pipe()
        p = ctx.Process(target=worker, args=(ca if name == "base" else cb, QWEN3 if name == "base" else args.adapter,
                                             args.gpu_mem, args.seed, child), daemon=True)
        p.start()
        W[name] = (p, parent)
    for name, (p, conn) in W.items():
        assert conn.recv() == "ready", name
    print(f"r4 workers ready: {list(W)}", flush=True)

    window = 1000
    for it in items:
        it.update(committed=[], prev=None, blocks=[], drafts=[], done=False,
                  model="base" if args.route in ("witness", "base") else "adapter")
    st = Counter()
    t0, r = time.time(), 0
    while any(not it["done"] for it in items):
        live = [it for it in items if not it["done"]]
        batch = {k: [] for k in W}            # model -> list of (item, view, prompt)
        for it in live:
            t = min((r + 1) * window, it["dur_ms"])
            it["t"] = t
            pr = head + join(it["committed"], zh)
            batch[it["model"]].append((it, 0, {"prompt": pr, "multi_modal_data": {"audio": (it["audio"][: t * 16], 16000)}}))
            if args.route == "witness" and t < it["dur_ms"]:
                a1 = it["v1"][t]
                assert len(a1) == t * 16
                batch[it["model"]].append((it, 1, {"prompt": pr, "multi_modal_data": {"audio": (a1, 16000)}}))
        for k, (p, conn) in W.items():
            conn.send([x[2] for x in batch[k]])
        outs = {}
        for k, (p, conn) in W.items():
            for (it, v, _), text in zip(batch[k], conn.recv()):
                outs[(it["id"], v)] = text
        for it in live:
            def draft_of(v):
                cont = outs[(it["id"], v)]
                sep = "" if zh or not it["committed"] or cont[:1].isspace() or cont[:1] in ",.;:!?)" else " "
                return units(join(it["committed"], zh) + sep + cont, zh)
            draft = draft_of(0)
            rec = {"t": it["t"], "model": it["model"], "draft": join(draft, zh)}
            st[f"reads_{it['model']}"] += 1
            n = len(it["committed"])
            if it["t"] >= it["dur_ms"]:
                new, it["done"] = draft[n:], True
            else:
                agreed = lcp(draft, it["prev"]) if it["prev"] is not None else 0
                new = draft[n: agreed] if agreed > n else []
                it["prev"] = draft
                if args.route == "witness":
                    d1 = draft_of(1)
                    agree = d1 == draft
                    rec.update(draft_v1=join(d1, zh), agree=agree)
                    it["model"] = "base" if agree else "adapter"
                    st["agree_reads"] += int(agree)
                st["reads_nonfinal"] += 1
            it["drafts"].append(rec)
            if new or it["done"]:
                text = ("" if zh or not n else " ") + join(new, zh)
                it["blocks"].append({"write_ms": it["t"], "text": text, "final": it["done"]})
                it["committed"] += new
        r += 1
        print(f"{args.direction} r4-{args.route} round {r}: {len(live)} sentences, "
              f"{ {k: len(v) for k, v in batch.items()} } prompts, {time.time() - t0:.0f}s", flush=True)
    for k, (p, conn) in W.items():
        conn.send(None)
    with open(out, "x", encoding="utf-8") as fh:
        for it in items:
            blocks = [b for b in it["blocks"] if b["text"].strip() or b["final"]]
            if blocks and not blocks[-1]["text"].strip() and len(blocks) > 1:
                blocks = blocks[:-1]
                blocks[-1]["final"] = True
            fh.write(json.dumps({"id": it["id"], "translation": join(it["committed"], zh), "blocks": blocks,
                                 "drafts": it["drafts"]}, ensure_ascii=False) + "\n")
    st = dict(st)
    st.update(route=args.route, adapter=args.adapter, views=args.views, items=len(items), seconds=round(time.time() - t0, 1),
              items_sha1=hashlib.sha1(open(args.items_file, "rb").read()).hexdigest())
    out.with_suffix(".r4.json").write_text(json.dumps(st), encoding="utf-8")
    print("r4 thinker:", json.dumps(st), flush=True)
    for k, (p, conn) in W.items():
        p.join(timeout=120)


if __name__ == "__main__":
    main()
