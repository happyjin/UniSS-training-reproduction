"""sst_hsmm_v1 prereg change 26: SVC-JM (joint-majority support-verified commit). = svc_si_thinker.py (change 25) with one
change: the verifier's same-read certificate is cut at the first token where EITHER the translator N (raw view d0) or the
SAP verifier gives its own greedy token probability <= 0.5 (vLLM logprobs=1, raw log-probs of the chosen token; the stop
token excluded). conf(x) = number of leading units of x's draft that are complete before that token. sv = min(LCP(d0, d_sap),
conf(d0), conf(d_sap)); commit cut = min(max(agreed, sv), LCP(d0, d1)) as in change 25. Fixed constant 0.5 (both models
more-than-half sure), no tuning. Each read record also carries svj (sv after the cut) and sv0 (change 25's sv).
sst_hsmm_v1 prereg change 25: Support-Verified Commit (SVC). A copy of mist_w3po_v1/voicecond/combo_si_thinker.py
(not modified) with --verifier <PEFT dir>: the SAP LoRA (change 24, t/train/adapters_final.pt -> wave16_make_lora alpha 1) on
the SAME Thinker (wave17 merged_enc_wise05) is served as a vLLM LoRA request. At every non-final read, besides N's two prompts
(raw view v0, enhanced view v1), a third prompt = v0's prompt with the verifier adapter; its greedy continuation stops at
<|im_end|> where the verifier judges the audio no longer supports the reference (SAP training). sv = LCP(d0, d_sap).
Commit cut = min(max(agreed, sv), LCP(d0, d1))  (N: min(agreed, LCP(d0, d1))): the verifier can only ADD a same-read
certificate next to LA2's previous-read one; the cross-view constraint stays; the content is always N's draft d0;
the final read is N's. Each read record also carries n_new (= N's cut) and sv. Asserted: new >= n_new.
Original docstring follows.
Robust line, full-system combination Thinker (reports/mist_w3po_v1/robust_combo_plan_20261120.zh-CN.md).

A copy of wave15_si_thinker_cvla.py (itself a line-by-line copy of si_thinker_stream's la2 / --windows 1 path; neither
file is modified) with two orthogonal switches added:

  --second-view none     plain LA2: no second prompt, commit d0[n:agreed]                 (= si_thinker_stream la2 w1)
  --second-view enh      CV-LA: v1 = DeepFilterNet3 prefix (--views, wave15_enh_views.py),
                         commit d0[n:min(agreed, LCP(d0, d1))]                             (= wave15 --view1 enh)
  --second-view same     v1 = v0's own audio (self-check: reproduces LA2)                  (= wave15 --view1 same)
  --second-view budget   equal-latency control (no second prompt): commit d0[n:min(agreed, B_t)], B_t = the cumulative
                         number of units the reference CV-LA run (--budget-from, a blocks file written by wave 15 or by
                         this script with --second-view enh) had committed by its read t (sum of its drafts' cv_new for
                         reads <= t); the final read commits v0's whole draft, as LA2. Same Thinker, same reads, at most
                         as many committed units per read as CV-LA; the only difference is WHERE the cut falls (CV-LA:
                         at a cross-view disagreement; budget: wherever the count runs out).
  --budget-pool-meta <meta_<dir>.jsonl>  with budget: noise-blind equal-latency control. The clean and noisy copies of
                         a sentence (same meta "set" and "pair", same reads) both get B_t = ceil((B_t^clean + B_t^noisy)/2),
                         so the total commit volume per sentence pair equals CV-LA's (up to the ceil) but no longer
                         depends on the noise condition.

  NCWI (wave 16) per-request LoRA, optional:
  --assign <assign_<dir>.jsonl> --lora-root <dir>  vLLM multi-LoRA (enable_lora, max_loras 8, rank 16, qkv_proj/o_proj,
                         as wave16_si_thinker_ncwi.py); the adapter is chosen by the ITEM ID's alpha* (not by the audio
                         key, so the enhanced view of an item gets the same adapter as its raw view); alpha* = 0 -> no
                         LoRA. --const-alpha A gives every item alpha A.
  Without --assign the Thinker is --thinker as is (the base, or a merged copy such as wave14/merged_unist_wise05).

Each read record carries la2_new (what LA2 would commit), new (what was committed), and per mode draft_v1 / budget;
alpha per item is written to the output row. Asserted: d1 == d0 implies new == la2_new; budget >= la2_new implies
new == la2_new. Output <out>/blocks_<dir>_w1.jsonl (+ .combo.json stats). Refuses to overwrite.

    <vllm_omni_030 python> -m experiments.mist_w3po_v1.voicecond.combo_si_thinker --second-view enh --views <dir> \
        [--assign <jsonl> --lora-root <dir>] [--thinker <dir>] --direction en2zh --out <dir> --items-file <jsonl>
"""
from __future__ import annotations

import argparse
import math
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.uniss_qwen3_omni_v1.evaluation.si_thinker_stream import QWEN3, THINKER_EOS, join, units  # noqa: E402
from experiments.mist_w3po_v1.voicecond.wave15_si_thinker_cvla import lcp, load_views  # noqa: E402


def load_budget(path: Path, items) -> None:
    """per item {t: cumulative committed units after read t} from a CV-LA run's drafts (non-final reads only)"""
    ref = {r["id"]: r for r in map(json.loads, open(path, encoding="utf-8"))}
    assert set(ref) >= {it["id"] for it in items}, "budget: missing items"
    for it in items:
        cum, b = 0, {}
        for rec in ref[it["id"]]["drafts"]:
            if "cv_new" in rec:
                cum += int(rec["cv_new"])
                b[int(rec["t"])] = cum
        reads = list(range(1000, it["dur_ms"], 1000))
        assert sorted(b) == reads, (it["id"], sorted(b)[:5], reads[:5])
        it["budget"] = b


def pool_budget(meta_path: Path, items) -> None:
    """noise-blind budget: both copies (clean, noisy) of a sentence get ceil of the mean of their two CV-LA budgets"""
    meta = {m["id"]: m for m in map(json.loads, open(meta_path, encoding="utf-8"))}
    groups = {}
    for it in items:
        m = meta[it["id"]]
        groups.setdefault((m["set"], m["pair"]), []).append(it)
    for k, g in groups.items():
        assert len(g) == 2 and {meta[x["id"]]["cond"] for x in g} == {"clean", "noisy"}, k
        assert sorted(g[0]["budget"]) == sorted(g[1]["budget"]), k
        pooled = {t: -(-(g[0]["budget"][t] + g[1]["budget"][t]) // 2) for t in g[0]["budget"]}
        for x in g:
            x["budget_cvla"], x["budget"] = x["budget"], dict(pooled)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--second-view", choices=("none", "enh", "same", "budget"), required=True)
    ap.add_argument("--views", default="")
    ap.add_argument("--budget-from", default="")
    ap.add_argument("--budget-pool-meta", default="")
    ap.add_argument("--assign", default="")
    ap.add_argument("--lora-root", default="")
    ap.add_argument("--const-alpha", type=float, default=None)
    ap.add_argument("--direction", choices=("zh2en", "en2zh"), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--thinker", default=QWEN3)
    ap.add_argument("--items-file", required=True)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=20261021)
    ap.add_argument("--verifier", required=True, help="PEFT adapter dir of the SAP LoRA (alpha 1)")
    args = ap.parse_args()
    sv = args.second_view
    if sv == "enh" and not args.views:
        raise SystemExit("--second-view enh needs --views")
    if sv == "budget" and not args.budget_from:
        raise SystemExit("--second-view budget needs --budget-from")
    if bool(args.assign) != bool(args.lora_root):
        raise SystemExit("--assign and --lora-root go together")
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
    if sv == "enh":
        load_views(Path(args.views), items)
    if sv == "budget":
        load_budget(Path(args.budget_from), items)
        if args.budget_pool_meta:
            pool_budget(Path(args.budget_pool_meta), items)

    # NCWI: per-item alpha* -> LoRARequest (same adapter for both views of an item)
    alpha_of, reqs = {}, {}
    assert sv == "enh" and not args.assign, "SVC: --second-view enh, no NCWI"
    from vllm.lora.request import LoRARequest
    vreq = LoRARequest("sap_verifier", 1, str(Path(args.verifier).resolve()))
    lora_kw = dict(enable_lora=True, max_loras=1, max_lora_rank=16, lora_target_modules=["qkv_proj", "o_proj"],
                   hf_overrides={"architectures": ["Qwen3OmniMoeThinkerForConditionalGeneration"]})   # = k1_si_thinker
    if args.assign:
        for r in map(json.loads, open(args.assign, encoding="utf-8")):
            al = round(float(args.const_alpha if args.const_alpha is not None else r["alpha"]), 4)
            assert alpha_of.setdefault(r["id"], al) == al, r["id"]
        missing = [it["id"] for it in items if it["id"] not in alpha_of]
        if missing:
            raise SystemExit(f"{len(missing)} items without alpha* in {args.assign}, e.g. {missing[:3]}")
        alphas = sorted({alpha_of[it["id"]] for it in items} - {0.0})
        assert len(alphas) <= 8, alphas
        for k, al in enumerate(alphas, start=1):
            p = Path(args.lora_root) / f"a{al:.4f}"
            cfg = json.load(open(p / "adapter_config.json"))
            assert abs(cfg["wave16"]["wise_alpha"] - al) < 1e-9, (p, cfg["wave16"])
            reqs[al] = LoRARequest(f"ncwi_a{al:.4f}", k, str(p.resolve()))
        lora_kw = dict(enable_lora=True, max_loras=8, max_lora_rank=16, lora_target_modules=["qkv_proj", "o_proj"])
        print("combo ncwi:", json.dumps({"adapters": {str(x): v.lora_path for x, v in reqs.items()},
                                         "items_per_alpha": dict(Counter(str(alpha_of[it["id"]]) for it in items)),
                                         "const_alpha": args.const_alpha}), flush=True)

    processor = Qwen3OmniMoeProcessor.from_pretrained(QWEN3)
    instruction = inference_instruction()[tgt]
    conv = [{"role": "user", "content": [{"type": "audio", "audio": "x.wav"},
                                         {"type": "text", "text": instruction}]}]
    head = processor.apply_chat_template([conv], add_generation_prompt=True, tokenize=False)[0]
    llm = LLM(model=args.thinker, gpu_memory_utilization=args.gpu_mem, max_model_len=8192, limit_mm_per_prompt={"audio": 1},
              seed=args.seed, tensor_parallel_size=1, **lora_kw)
    params = SamplingParams(temperature=0.0, max_tokens=160, stop_token_ids=[THINKER_EOS], logprobs=1)
    tok = processor.tokenizer
    LHALF = math.log(0.5)

    def conf(base_units, cont, out):
        """leading units of units(join(base) + sep + cont) complete before the first token with p(chosen) <= 0.5"""
        ids = [t for t in out.token_ids if t != THINKER_EOS]
        j = next((i for i, t in enumerate(ids) if out.logprobs[i][t].logprob <= LHALF), len(ids))
        sep_ = "" if zh or not base_units or cont[:1].isspace() or cont[:1] in ",.;:!?)" else " "
        full = units(join(base_units, zh) + sep_ + cont, zh)
        if j == len(ids):
            return len(full)
        pre = units(join(base_units, zh) + sep_ + tok.decode(ids[:j]).replace("\ufffd", ""), zh)
        k = 0
        while k < len(pre) and k < len(full) and pre[k] == full[k]:
            k += 1
        return k

    window = 1000
    two = sv in ("enh", "same")
    tag = f"{args.direction} {out.stem} svcjm-{sv}{'-ncwi' if args.assign else ''}"
    for it in items:
        it.update(committed=[], prev=None, blocks=[], drafts=[], done=False)
    st = Counter()
    t0, r = time.time(), 0
    while any(not it["done"] for it in items):
        live = [it for it in items if not it["done"]]
        prompts, vk, owners = [], [], []
        for it in live:
            t = min((r + 1) * window, it["dur_ms"])
            it["t"] = t
            prompts.append({"prompt": head + join(it["committed"], zh),
                            "multi_modal_data": {"audio": (it["audio"][: t * 16], 16000)}})
            owners.append(it)
        if two:
            for k, it in enumerate(live):
                if it["t"] < it["dur_ms"]:
                    a1 = it["audio"][: it["t"] * 16] if sv == "same" else it["v1"][it["t"]]
                    assert len(a1) == it["t"] * 16
                    prompts.append({"prompt": prompts[k]["prompt"], "multi_modal_data": {"audio": (a1, 16000)}})
                    vk.append(k)
                    owners.append(it)
            sk = []
            for k, it in enumerate(live):
                if it["t"] < it["dur_ms"]:
                    prompts.append({"prompt": prompts[k]["prompt"], "multi_modal_data": prompts[k]["multi_modal_data"]})
                    sk.append(k)
                    owners.append(it)
        if args.assign:
            lr = [reqs[alpha_of[o["id"]]] if alpha_of[o["id"]] > 0 else None for o in owners]
            results = llm.generate(prompts, params, use_tqdm=False, lora_request=lr)
        else:
            nv = len(live) + len(vk)
            results = llm.generate(prompts, params, use_tqdm=False, lora_request=[None] * nv + [vreq] * len(sk))
        res1 = {k: results[len(live) + j] for j, k in enumerate(vk)}
        ress = {k: results[len(live) + len(vk) + j] for j, k in enumerate(sk)}
        for k, (it, res) in enumerate(zip(live, results[: len(live)])):
            cont = res.outputs[0].text
            sep = "" if zh or not it["committed"] or cont[:1].isspace() or cont[:1] in ",.;:!?)" else " "
            draft = units(join(it["committed"], zh) + sep + cont, zh)
            rec = {"t": it["t"], "draft": join(draft, zh)}
            n = len(it["committed"])
            if it["t"] >= it["dur_ms"]:
                new, it["done"] = draft[n:], True
            else:
                agreed = lcp(draft, it["prev"]) if it["prev"] is not None else 0
                la2_new = max(0, agreed - n)
                if sv == "none":
                    cut = agreed
                elif sv == "budget":
                    b = it["budget"][it["t"]]
                    cut = min(agreed, b)
                    rec["budget"] = b
                    if b >= agreed:
                        assert cut == agreed
                else:
                    c1 = res1[k].outputs[0].text
                    sep1 = "" if zh or not it["committed"] or c1[:1].isspace() or c1[:1] in ",.;:!?)" else " "
                    d1 = units(join(it["committed"], zh) + sep1 + c1, zh)
                    cv = lcp(draft, d1)
                    n_cut = min(agreed, cv)
                    cs = ress[k].outputs[0].text
                    seps = "" if zh or not it["committed"] or cs[:1].isspace() or cs[:1] in ",.;:!?)" else " "
                    ds = units(join(it["committed"], zh) + seps + cs, zh)
                    sv0 = lcp(draft, ds)
                    svl = min(sv0, conf(it["committed"], cont, res.outputs[0]), conf(it["committed"], cs, ress[k].outputs[0]))
                    rec.update(sv0=sv0, svj=svl)
                    cut = min(max(agreed, svl), cv)
                    assert cut >= n_cut
                    rec.update(draft_sap=join(ds, zh), sv=svl, n_new=max(0, n_cut - n))
                    st["svc_extra_units"] += max(0, cut - max(n_cut, n))
                    st["svc_extra_reads"] += int(cut > max(n_cut, n))
                    rec["draft_v1"] = join(d1, zh)
                    st["v1_differs"] += int(d1 != draft)
                    if d1 == draft:
                        assert n_cut == agreed, (it["id"], it["t"])
                new = draft[n: cut] if cut > n else []
                it["prev"] = draft
                rec.update(la2_new=la2_new, cv_new=len(new))
                st["reads_nonfinal"] += 1
                st["cut_reads"] += int(len(new) < la2_new)
                st["la2_units"] += la2_new
                st["committed_units"] += len(new)
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
            row = {"id": it["id"], "translation": join(it["committed"], zh), "blocks": blocks, "drafts": it["drafts"]}
            if args.assign:
                row["alpha"] = alpha_of[it["id"]]
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    st = dict(st)
    st.update(method="SVC-JM (change 26)", verifier=args.verifier, second_view=sv, thinker=args.thinker, assign=args.assign, lora_root=args.lora_root, const_alpha=args.const_alpha,
              views=args.views, budget_from=args.budget_from, budget_pool_meta=args.budget_pool_meta, items=len(items), seconds=round(time.time() - t0, 1),
              items_sha1=hashlib.sha1(open(args.items_file, "rb").read()).hexdigest())
    out.with_suffix(".combo.json").write_text(json.dumps(st), encoding="utf-8")
    print(f"{tag}: {len(items)} sentences, {time.time() - t0:.0f}s -> {out}", flush=True)
    print("combo thinker:", json.dumps(st), flush=True)


if __name__ == "__main__":
    main()
