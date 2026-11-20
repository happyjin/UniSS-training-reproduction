"""Robust wave 15: CV-LA, cross-view LocalAgreement (reports/mist_w3po_v1/robust_wave15_cvla_prereg.zh-CN.md; method and
rule: robust_mainline_plan_20261120.zh-CN.md section 4.1).

A copy of si_thinker_stream's la2 / --windows 1 path (the shared file is not modified; units, join, QWEN3, THINKER_EOS are
imported from it; item loading, prompt head, LLM(...) / SamplingParams(...) arguments, draft building, the final read and
the output post-processing are copied line by line; no snap / hygiene / ban-wait / other options; seed 20261021).
Each round: the v0 prompts of all live items, then the v1 prompts (same head + committed text, other audio) of the live
items not at their final read, in ONE llm.generate call. With d0 / d1 the drafts of v0 / v1 (LA2's own concatenation),
agreed = LCP(prev, d0) (prev = v0's previous draft, as in LA2), cv = min(agreed, LCP(d0, d1)) = LCP(A_t, d1); d0[n:cv] is
committed. The final read commits v0's whole draft, as LA2.
  --view1 enh   v1 = DeepFilterNet3 of the heard prefix, from wave15_enh_views.py (--views <dir>; sha1 and length checked)
  --view1 same  v1 = v0's own audio (self-check: must reproduce LA2 up to vLLM's own nondeterminism)
Every read record carries draft_v1, la2_new (what LA2 would commit in this state) and cv_new; asserted: d1 == d0 implies
cv_new == la2_new.
    <vllm_omni_030 python> -m experiments.mist_w3po_v1.voicecond.wave15_si_thinker_cvla --view1 enh --views <dir> \
        --direction en2zh --thinker <dir> --out <dir> --items-file <jsonl> [--gpu-mem 0.85]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from experiments.uniss_qwen3_omni_v1.evaluation.si_thinker_stream import QWEN3, THINKER_EOS, join, units  # noqa: E402


def lcp(a, b) -> int:
    k = 0
    while k < min(len(a), len(b)) and a[k] == b[k]:
        k += 1
    return k


def load_views(vdir: Path, items) -> None:
    idx = {r["id"]: r for r in map(json.loads, open(vdir / "index.jsonl", encoding="utf-8"))}
    assert set(idx) >= {it["id"] for it in items}, "views: missing items"
    for it in items:
        r = idx[it["id"]]
        assert r["dur_ms"] == it["dur_ms"], (it["id"], r["dur_ms"], it["dur_ms"])
        assert r["sha1"] == hashlib.sha1(np.ascontiguousarray(it["audio"]).tobytes()).hexdigest(), it["id"]
        assert r["reads"] == list(range(1000, it["dur_ms"], 1000)), it["id"]
        x = np.load(vdir / r["file"])
        assert len(x) == sum(t * 16 for t in r["reads"]), it["id"]
        v, o = {}, 0
        for t in r["reads"]:
            v[t] = x[o: o + t * 16]
            o += t * 16
        it["v1"] = v


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--view1", choices=("enh", "same"), required=True)
    ap.add_argument("--views", default="")
    ap.add_argument("--direction", choices=("zh2en", "en2zh"), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--thinker", default=QWEN3)
    ap.add_argument("--items-file", required=True)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=20261021)
    args = ap.parse_args()
    if args.view1 == "enh" and not args.views:
        raise SystemExit("--view1 enh needs --views")
    out = Path(args.out) / f"blocks_{args.direction}_w1.jsonl"
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    Path(args.out).mkdir(parents=True, exist_ok=True)

    from transformers import Qwen3OmniMoeProcessor
    from vllm import LLM, SamplingParams
    from experiments.uniss_qwen3_omni_v1.training.e2_batch import inference_instruction

    zh = args.direction.endswith("zh")
    tgt = "cmn" if zh else "eng"
    items = [{"id": r["id"], "wav": r["audio"], "start": 0, "end": int(r["dur_ms"]), "trg_text": r.get("translation", "")}
             for r in map(json.loads, open(args.items_file, encoding="utf-8"))]
    cache = {}
    for it in items:
        if it["wav"] not in cache:
            x, sr = sf.read(it["wav"], dtype="float32", always_2d=True)
            assert sr == 16000, it["wav"]
            cache = {it["wav"]: x.mean(axis=1)}
        it["audio"] = cache[it["wav"]][it["start"] * 16: it["end"] * 16]
        it["dur_ms"] = len(it["audio"]) // 16
    if args.view1 == "enh":
        load_views(Path(args.views), items)
    processor = Qwen3OmniMoeProcessor.from_pretrained(QWEN3)
    instruction = inference_instruction()[tgt]
    conv = [{"role": "user", "content": [{"type": "audio", "audio": "x.wav"},
                                         {"type": "text", "text": instruction}]}]
    head = processor.apply_chat_template([conv], add_generation_prompt=True, tokenize=False)[0]
    llm = LLM(model=args.thinker, gpu_memory_utilization=args.gpu_mem, max_model_len=8192, limit_mm_per_prompt={"audio": 1},
              seed=args.seed, tensor_parallel_size=1)
    params = SamplingParams(temperature=0.0, max_tokens=160, stop_token_ids=[THINKER_EOS])

    window = 1000
    tag = f"{args.direction} {out.stem} cvla-{args.view1}"
    for it in items:
        it.update(committed=[], prev=None, blocks=[], drafts=[], done=False)
    st = {"reads_nonfinal": 0, "v1_differs": 0, "cut_reads": 0, "la2_units": 0, "cv_units": 0}
    t0, r = time.time(), 0
    while any(not it["done"] for it in items):
        live = [it for it in items if not it["done"]]
        prompts, vk = [], []
        for it in live:
            t = min((r + 1) * window, it["dur_ms"])
            it["t"] = t
            prompts.append({"prompt": head + join(it["committed"], zh),
                            "multi_modal_data": {"audio": (it["audio"][: t * 16], 16000)}})
        for k, it in enumerate(live):
            if it["t"] < it["dur_ms"]:
                a1 = it["audio"][: it["t"] * 16] if args.view1 == "same" else it["v1"][it["t"]]
                assert len(a1) == it["t"] * 16
                prompts.append({"prompt": prompts[k]["prompt"], "multi_modal_data": {"audio": (a1, 16000)}})
                vk.append(k)
        results = llm.generate(prompts, params, use_tqdm=False)
        res1 = {k: results[len(live) + j] for j, k in enumerate(vk)}
        for k, (it, res) in enumerate(zip(live, results[: len(live)])):
            cont = res.outputs[0].text
            sep = "" if zh or not it["committed"] or cont[:1].isspace() or cont[:1] in ",.;:!?)" else " "
            draft = units(join(it["committed"], zh) + sep + cont, zh)
            rec = {"t": it["t"], "draft": join(draft, zh)}
            n = len(it["committed"])
            if it["t"] >= it["dur_ms"]:
                new, it["done"] = draft[n:], True
            else:
                c1 = res1[k].outputs[0].text
                sep1 = "" if zh or not it["committed"] or c1[:1].isspace() or c1[:1] in ",.;:!?)" else " "
                d1 = units(join(it["committed"], zh) + sep1 + c1, zh)
                agreed = lcp(draft, it["prev"]) if it["prev"] is not None else 0
                cv = min(agreed, lcp(draft, d1))
                la2_new = max(0, agreed - n)
                new = draft[n: cv] if cv > n else []
                if d1 == draft:
                    assert len(new) == la2_new, (it["id"], it["t"])
                it["prev"] = draft
                rec.update(draft_v1=join(d1, zh), la2_new=la2_new, cv_new=len(new))
                st["reads_nonfinal"] += 1
                st["v1_differs"] += int(d1 != draft)
                st["cut_reads"] += int(len(new) < la2_new)
                st["la2_units"] += la2_new
                st["cv_units"] += len(new)
            it["drafts"].append(rec)
            if new or it["done"]:
                text = ("" if zh or not n else " ") + join(new, zh)
                it["blocks"].append({"write_ms": it["t"], "text": text, "final": it["done"]})
                it["committed"] += new
        r += 1
        print(f"{tag} round {r}: {len(live)} sentences, {len(vk)} v1 prompts, {time.time() - t0:.0f}s", flush=True)
    with open(out, "x", encoding="utf-8") as fh:
        for it in items:
            blocks = [b for b in it["blocks"] if b["text"].strip() or b["final"]]
            if blocks and not blocks[-1]["text"].strip() and len(blocks) > 1:   # an empty last commit: the one before is the last
                blocks = blocks[:-1]
                blocks[-1]["final"] = True
            fh.write(json.dumps({"id": it["id"], "translation": join(it["committed"], zh), "blocks": blocks,
                                 "drafts": it["drafts"]}, ensure_ascii=False) + "\n")
    st.update(view1=args.view1, thinker=args.thinker, items=len(items), seconds=round(time.time() - t0, 1))
    out.with_suffix(".cvla.json").write_text(json.dumps(st), encoding="utf-8")
    print(f"{tag}: {len(items)} sentences, {time.time() - t0:.0f}s -> {out}", flush=True)
    print("wave15 cvla:", json.dumps(st), flush=True)


if __name__ == "__main__":
    main()
