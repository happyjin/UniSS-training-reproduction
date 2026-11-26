"""sst_hsmm_v1 change 50 (S4, cross-model agreement cap = CMA): a copy of hsc17_si_thinker.py (not modified) with --cma: at every
non-final read one more prompt = the BASE Thinker (no LoRA) on the raw audio with the same (healed) prompt; after the posterior cut,
cut = min(cut, max(n, LCP(d0, d_base))) where d0 is the --draft-lora (anticipatory) draft. Records rec["cma"] = LCP - n.
Original header follows.
sst_hsmm_v1 change 48 (S2, faithful finalisation): a copy of hsc16_si_thinker.py (not modified) with --draft-lora <PEFT dir>:
the drafter prompts (raw view d0 and the enhanced view d1) of every NON-FINAL read are served with that LoRA (e.g. the D3
source-future adapter) on the same engine as the SAP verifier (max_loras 2); the FINAL read's drafts use the base Thinker.
Without --draft-lora it is hsc16. Original header follows.
sst_hsmm_v1 prereg change 45: a copy of hsc15_si_thinker.py (not modified). --variants takes specs mode@theta or pandora@theta@kappa
(mode in base, heal, cec, pandora; e.g. "base@0.75,heal@0.70,cec@0.75,pandora@0.75@0.15"); every spec gets its own item copies in one
engine, rows -> <out>/blocks_<dir>_<spec with @ -> _>.jsonl. theta replaces --theta per copy.
  cec      heal + character-exact evidence: for a healed request (d0 and SAP) the first token's log-prob is renormalised by
           Z = sum of the top-20 candidates whose text is consistent with the backed-off suffix (startswith either way); greedy drafts = heal.
  pandora  heal + blocking-aware threshold: after the n2 cut, unit j (P_j < theta) is committed together with the b_j consecutive
           units behind it that have P >= theta iff P_j >= max(0.5, 1 - kappa (1 + b_j)); repeated while it holds.
Was: sst_hsmm_v1 prereg change 44 (P0-K): a copy of hsc11_si_thinker.py (not modified) that runs commit-boundary variants in one
vLLM engine: --variants base,snap,heal (every item replicated once per variant; rows -> <out>/blocks_<dir>_<variant>.jsonl).
Token boundaries = character end offsets of the Qwen tokenizer's tokenisation of the read's draft text join(d0) (zh units only;
en words never end inside a token).
  base  p3h8 unchanged; each non-final read record gets mid = 1 if the commit ends inside a token.
  snap  the non-final commit cut is moved back to the last token boundary at or before it (not below the committed length n);
        rec["snap"] = [cut, snapped cut].
  heal  commits as base; when the committed text ends inside a token (judged on the commit read's draft), the next prompts
        (d0, d1, SAP) back off to the last token boundary k0 and the continuation is constrained (vLLM structured outputs, regex
        re.escape(suffix) + "[\\s\\S]*") to start with the backed-off suffix, which is then stripped; drafts and log-probs are
        computed exactly as before from the healed base. A violated constraint is counted (heal_fail) and the read keeps no new units.
Was: sst_hsmm_v1 prereg change 40-P0: a copy of hsc10_si_thinker.py (not modified) plus read-level counterfactual forks.
--fork-reads F --fork-k K: per item, F distinct non-final reads are drawn (numpy rng seeded by sha1(id), fixed); when the
greedy (base) stream reaches such a read t, K clones of its state (committed, prev, H, qend, blocks, drafts) are made; each
clone's raw-view draft d0 at read t is SAMPLED (temperature --fork-temp, top_p --fork-top-p, logprobs=1, seed sha1(id|t|k)),
everything else (d1, SAP, posterior, commit rule) is unchanged, and from read t + 1 on the clone runs greedily to the end of
the sentence. Base rows go to blocks_<dir>_w1.jsonl as before; clones go to forks_<dir>.jsonl with fork_t / fork_k.
Clones run in the same round as their base read, so the global round counter stays valid. Stats count base and clones.
Was: sst_hsmm_v1 prereg change 38: a copy of hsc9_si_thinker.py (not modified) that, with --samples K > 0, adds per non-final read one wave17
request (raw view, same prompt and committed prefix, no LoRA) with n = K sampled continuations (--samp-temp, top_p 0.95, --samp-max
tokens, seed from item id / read) and stores rec["cons"] (used as an HSMM observation when --params holds hsmm_cons params, change 38-2): for every uncommitted draft position i, the share of samples whose units agree
with the draft through i (lcp > i), counted over the samples that reach i (an EOS-ended sample always counts; a truncated sample's last
unit does not), -1 if none reaches i. The commit decision itself is unchanged (posterior from --params).
Was: sst_hsmm_v1 prereg change 36: a copy of hsc8_si_thinker.py (not modified) that also stores lp0 / lps (from n) on every non-final read in
--mode hsc, so its own policy rollouts on the UniST training items can be walked as training data (on-policy relabelling).
Was: sst_hsmm_v1 prereg change 35: a copy of hsc6_si_thinker.py (not modified) plus --variant r1 (risk-budget prefix): from n commit the
longest prefix c with P_i >= 0.5 for all i < c and sum_{i<c} (1 - P_i) <= --theta (= epsilon, expected revised units per read).
Was: sst_hsmm_v1 prereg change 33: a copy of hsc5_si_thinker.py (not modified) whose posterior module follows the params file:
hsmm_dep (HSC5 params, keys A_U ...) or hsmm_joint (HSC4 h9 / HSC6 anchored params). Variants as change 32:
no theta - 0.1 pause rule; --variant:
  n1  commit = max(min(agreed, LCP(d0, d1)), n + kH), kH = leading positions with P_i >= theta      (N base kept)
  n2  commit = n + kH                                                                              (pure posterior)
  s1  N base, then beyond it position by position while P_i >= lam / (lam + b_i), lam = theta / (1 - theta) (theta = the
      threshold when the playback queue is empty); b_i = clip(t + 1 - max(q_i, t), 0, 1) s = how much earlier unit i is spoken
      if committed now rather than at the next read (q_i = queue end before unit i, 0.25 s / zh char, 0.35 s / en word);
      b_i = 0 -> stop (waiting one read costs no speech latency)
  s2  the same slack rule from n (pure posterior).
Was: sst_hsmm_v1 prereg change 31 (HSC4, hsmm_joint, n1 / n2 with change 30-C's pause-aware threshold). Inputs per read: inputs_vec(r, vad, age, t) from this read's new 1 s (raw vs
enhanced view). Was:
sst_hsmm_v1 prereg change 30: HSC2-soft / HSC3 (reliability clock) / HSC3 + pause-aware threshold. A copy of
hsc2_si_thinker.py (not modified) with --variant:
  soft        hsmm_commit2 posterior (h3 params); commit = max(min(agreed, LCP(d0, d1)), n + kH)  (no cross-view cap on kH)
  clock       hsmm_commit3 posterior (reliability clock; r_t from this read's new 1 s, raw vs enhanced view); same commit rule
  clockpause  clock + theta_t = max(0.5, theta - 0.1) when the estimated playback queue empties before the next read
              (queue_end = max(queue_end, write time) + units x 0.25 s (zh) / 0.35 s (en); idle iff queue_end <= t + 0.3 s)
Was:
sst_hsmm_v1 prereg change 29: HSC2 = hsc_si_thinker.py with hsmm_commit2 (corrected model). Was: prereg change 28: HSC rollout / inference. A copy of sdc_si_thinker.py (change 27; not modified):
--mode rollout: commit = N's cut min(agreed, LCP(d0, d1)); every non-final read record also stores lp0 / lps (per-unit minimum
                token log-probs of d0 / d_sap for the uncommitted units) for hsmm_commit.py fit (unsupervised EM).
--mode hsc:     per-position observation histories (hsmm_commit.observations), P_i = hsmm_commit.posterior_stable;
                commit = min(max(agreed, n + #leading P_i >= --theta), LCP(d0, d1)); final read = N.
Original docstring (sdc) follows.
sst_hsmm_v1 prereg change 27: SDC rollout / inference. A copy of svc2_si_thinker.py (change 26; not modified) with
--mode rollout: commit = N's cut min(agreed, LCP(d0, d1)) (= combo enh), and every non-final read's SDC features
                (sdc.feats) of the uncommitted units of d0 are saved with label 1(d0[:i+1] == final[:i+1]) to <out>/feats_<dir>.npz
--mode sdc:     commit = n + sdc.commit_len(S, --theta), S = sigmoid(head(feats)) from --head (train_sdc.py); final read = N.
Original docstring (svc2) follows.
sst_hsmm_v1 prereg change 26: SVC-JM (joint-majority support-verified commit). = svc_si_thinker.py (change 25) with one
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
    ap.add_argument("--mode", choices=("rollout", "hsc"), required=True)
    ap.add_argument("--params", default="")
    ap.add_argument("--theta", type=float, default=None)
    ap.add_argument("--samples", type=int, default=0)
    ap.add_argument("--samp-temp", type=float, default=0.8)
    ap.add_argument("--samp-max", type=int, default=24)
    ap.add_argument("--variants", default="base,snap,heal")
    ap.add_argument("--fork-reads", type=int, default=0)
    ap.add_argument("--fork-k", type=int, default=4)
    ap.add_argument("--fork-temp", type=float, default=1.0)
    ap.add_argument("--fork-top-p", type=float, default=1.0)
    ap.add_argument("--variant", choices=("n1", "n2", "s1", "s2", "r1"), required=True)
    ap.add_argument("--verifier", required=True, help="PEFT adapter dir of the SAP LoRA (alpha 1)")
    ap.add_argument("--cma", action="store_true", help="change 50: cap commits by LCP with the base Thinker's draft")
    ap.add_argument("--lora-final", action="store_true", help="change 50: serve --draft-lora on the final read too (no faithful finalisation)")
    ap.add_argument("--draft-lora", default="", help="change 48: PEFT dir served for non-final drafter prompts")
    args = ap.parse_args()
    sv = args.second_view
    if sv == "enh" and not args.views:
        raise SystemExit("--second-view enh needs --views")
    if sv == "budget" and not args.budget_from:
        raise SystemExit("--second-view budget needs --budget-from")
    if bool(args.assign) != bool(args.lora_root):
        raise SystemExit("--assign and --lora-root go together")
    VARS = [x.strip() for x in args.variants.split(",")]
    SPEC = {}
    for v_ in VARS:
        p_ = v_.split("@")
        assert p_[0] in ("base", "heal", "cec", "pandora") and len(p_) == (3 if p_[0] == "pandora" else 2), v_
        SPEC[v_] = {"mode": p_[0], "theta": float(p_[1]), "kappa": float(p_[2]) if p_[0] == "pandora" else None}
    assert args.fork_reads == 0
    outs = {v_: Path(args.out) / f"blocks_{args.direction}_{v_.replace('@', '_')}.jsonl" for v_ in VARS}
    for o_ in outs.values():
        if o_.exists():
            raise SystemExit(f"拒绝覆盖已有输出: {o_}")
    out = outs[VARS[0]]
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
    dreq = LoRARequest("draft_lora", 2, str(Path(args.draft_lora).resolve())) if args.draft_lora else None
    lora_kw = dict(enable_lora=True, max_loras=2 if dreq else 1, max_lora_rank=16, lora_target_modules=["qkv_proj", "o_proj"],
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
    from experiments.sst_hsmm_v1 import sdc as SD
    from experiments.sst_hsmm_v1 import hsmm_commit2 as HC
    if args.mode == "hsc":
        assert args.params and args.theta is not None, "--mode hsc needs --params and --theta"
        prm_ = json.load(open(args.params))["dirs"][args.direction]["params"]
        if "A_U" in prm_:
            from experiments.sst_hsmm_v1 import hsmm_dep as HN
            HP = HN.DParams(prm_)
        else:
            from experiments.sst_hsmm_v1 import hsmm_joint as HN
            HP = HN.NParams(prm_)
            if "cb_U" in prm_["E"]:      # change 38-2: HSC6 + K-sample consistency emission (hsmm_cons)
                from experiments.sst_hsmm_v1 import hsmm_cons as HCO
                HP = HCO.CParams(prm_)
                assert args.samples > 0, "consistency params need --samples > 0"
        print("posterior module:", HN.__name__, flush=True)
    UNIT_S = 0.25 if zh else 0.35

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
    tag = f"{args.direction} {len(VARS)} variants hsc16-{args.variant}-{sv}{'-ncwi' if args.assign else ''}"
    import re as _re
    from functools import lru_cache
    from vllm.sampling_params import StructuredOutputsParams

    @lru_cache(maxsize=200000)
    def tends(text):
        return frozenset(e for _, e in tok(text, return_offsets_mapping=True, add_special_tokens=False)["offset_mapping"])

    def boundary(us, k, dtext):        # is the end of us[:k] a token end of dtext's tokenisation (zh only)?
        return (not zh) or k == 0 or len(join(us[:k], zh)) in tends(dtext)

    def hb(it):                        # the prompt's committed base (healed: backed off to the last token boundary)
        return it["committed"][: it["heal_k"]] if it.get("heal_k") is not None else it["committed"]

    def suf(it):
        return join(it["committed"][it["heal_k"]:], zh) if it.get("heal_k") is not None else ""

    def pp(o):
        if o.get("heal_k") is None:
            return params
        return SamplingParams(temperature=0.0, max_tokens=160, stop_token_ids=[THINKER_EOS], logprobs=20 if SPEC[o["v"]]["mode"] == "cec" else 1,
                              structured_outputs=StructuredOutputsParams(regex=_re.escape(suf(o)) + r"[\s\S]*"))

    def strip_(it, text):
        s_ = suf(it)
        if not s_:
            return text, True
        if text.startswith(s_):
            return text[len(s_):], True
        return "", False

    import types as _types

    def cec_out(it, o):            # change 45: renormalise the first healed token over the suffix-consistent candidates
        sfx = suf(it)
        if SPEC[it["v"]]["mode"] != "cec" or not sfx or not o.token_ids or not o.logprobs:
            return o
        lp0d = o.logprobs[0]
        Z = 0.0
        for v_, L_ in lp0d.items():
            s_ = tok.decode([v_])
            if s_ and (s_.startswith(sfx) or sfx.startswith(s_)):
                Z += math.exp(L_.logprob)
        t0_ = o.token_ids[0]
        if Z <= 0.0 or t0_ not in lp0d:
            return o
        d0_ = dict(lp0d)
        d0_[t0_] = _types.SimpleNamespace(logprob=lp0d[t0_].logprob - math.log(Z))
        st["cec_renorm"] += 1
        return _types.SimpleNamespace(token_ids=o.token_ids, logprobs=[d0_] + list(o.logprobs[1:]))

    items = [dict(it, v=v_) for v_ in VARS for it in items]
    for it in items:
        it.update(committed=[], prev=None, blocks=[], drafts=[], done=False, hist=[], H={}, qend=0.0, fork=None, sample_now=False, heal_k=None)
        nf = list(range(1000, it["dur_ms"], 1000))     # the non-final reads (t < dur)
        rng_ = np.random.default_rng(int(hashlib.sha1(it["id"].encode()).hexdigest(), 16) % (2 ** 32))
        it["fork_at"] = set(int(x) for x in rng_.choice(nf, size=min(args.fork_reads, len(nf)), replace=False)) if nf and args.fork_reads > 0 else set()
    import copy
    forks = []
    STATE = ("committed", "prev", "blocks", "drafts", "hist", "H", "qend")
    st = Counter()
    t0, r = time.time(), 0
    while any(not it["done"] for it in items + forks):
        for it in [x for x in items if not x["done"] and x["fork"] is None]:
            t_next = min((r + 1) * window, it["dur_ms"])
            if t_next in it["fork_at"]:
                for kf in range(args.fork_k):
                    c_ = {k_: v_ for k_, v_ in it.items() if k_ not in STATE}
                    c_.update(copy.deepcopy({k_: it[k_] for k_ in STATE}))
                    c_.update(fork=(t_next, kf), sample_now=True, fork_at=set())
                    forks.append(c_)
        live = [it for it in items + forks if not it["done"]]
        prompts, vk, owners = [], [], []
        for it in live:
            t = min((r + 1) * window, it["dur_ms"])
            it["t"] = t
            prompts.append({"prompt": head + join(hb(it), zh),
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
            ck = []
            if args.samples > 0:
                for k, it in enumerate(live):
                    if it["t"] < it["dur_ms"]:
                        prompts.append({"prompt": prompts[k]["prompt"], "multi_modal_data": prompts[k]["multi_modal_data"]})
                        ck.append(k)
            bk = []
            if args.cma:
                for k, it in enumerate(live):
                    if it["t"] < it["dur_ms"]:
                        prompts.append({"prompt": prompts[k]["prompt"], "multi_modal_data": prompts[k]["multi_modal_data"]})
                        bk.append(k)
                        owners.append(it)
                        owners.append(it)
        if args.assign:
            lr = [reqs[alpha_of[o["id"]]] if alpha_of[o["id"]] > 0 else None for o in owners]
            results = llm.generate(prompts, params, use_tqdm=False, lora_request=lr)
        else:
            nv = len(live) + len(vk)
            def p0_(o):
                if not o["sample_now"]:
                    return params
                return SamplingParams(temperature=args.fork_temp, top_p=args.fork_top_p, max_tokens=160, stop_token_ids=[THINKER_EOS], logprobs=1,
                                      seed=int(hashlib.sha1(f"{o['id']}|{o['fork'][0]}|{o['fork'][1]}".encode()).hexdigest(), 16) % (2 ** 31))
            sps = [pp(o) for o in live] + [pp(live[k_]) for k_ in vk] + [pp(live[k_]) for k_ in sk] + [SamplingParams(n=args.samples, temperature=args.samp_temp, top_p=0.95, max_tokens=args.samp_max,
                                                               stop_token_ids=[THINKER_EOS],
                                                               seed=int(hashlib.sha1(f"{live[k]['id']}|{live[k]['t']}".encode()).hexdigest(), 16) % (2 ** 31))
                                               for k in ck]
            dq = lambda o: dreq if (dreq is not None and (o["t"] < o["dur_ms"] or args.lora_final)) else None   # change 48
            sps = sps + [params] * len(bk)
            lrq = [dq(o) for o in live] + [dq(live[k_]) for k_ in vk] + [vreq] * len(sk) + [None] * len(ck) + [None] * len(bk)
            assert len(lrq) == len(prompts) == len(sps), (len(lrq), len(prompts), len(sps))
            results = llm.generate(prompts, sps, use_tqdm=False, lora_request=lrq)
            resc = {k: results[nv + len(sk) + j] for j, k in enumerate(ck)}
            resb = {k: results[nv + len(sk) + len(ck) + j] for j, k in enumerate(bk)}
        res1 = {k: results[len(live) + j] for j, k in enumerate(vk)}
        ress = {k: results[len(live) + len(vk) + j] for j, k in enumerate(sk)}
        for k, (it, res) in enumerate(zip(live, results[: len(live)])):
            cont_full = res.outputs[0].text
            cont, ok0 = strip_(it, cont_full)
            st["heal_fail"] += int(not ok0)
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
                    c1, ok1 = strip_(it, res1[k].outputs[0].text)
                    st["heal_fail"] += int(not ok1)
                    sep1 = "" if zh or not it["committed"] or c1[:1].isspace() or c1[:1] in ",.;:!?)" else " "
                    d1 = units(join(it["committed"], zh) + sep1 + c1, zh)
                    cv = lcp(draft, d1)
                    n_cut = min(agreed, cv)
                    cs_full = ress[k].outputs[0].text
                    cs, oks = strip_(it, cs_full)
                    st["heal_fail"] += int(not oks)
                    seps = "" if zh or not it["committed"] or cs[:1].isspace() or cs[:1] in ",.;:!?)" else " "
                    ds = units(join(it["committed"], zh) + seps + cs, zh)
                    sv0 = lcp(draft, ds)
                    full0, lp0 = SD.unit_lp(tok, units, join, hb(it) if ok0 else it["committed"], cont_full if ok0 else "", cec_out(it, res.outputs[0]) if ok0 else res.outputs[0], zh)
                    fulls, lps = SD.unit_lp(tok, units, join, hb(it) if oks else it["committed"], cs_full if oks else "", cec_out(it, ress[k].outputs[0]) if oks else ress[k].outputs[0], zh)
                    assert full0 == draft and fulls == ds, (it["id"], it["t"])
                    svl = sv0
                    if args.mode == "rollout":
                        cut = n_cut
                        rec["lp0"] = [round(float(x), 3) for x in lp0[n:]]
                        rec["lps"] = [round(float(x), 3) for x in lps[n:]]
                    if args.samples > 0 and k in resc:
                        cons_ = []
                        samp_ = []
                        for o_ in resc[k].outputs:
                            sp_ = "" if zh or not it["committed"] or o_.text[:1].isspace() or o_.text[:1] in ",.;:!?)" else " "
                            su_ = units(join(it["committed"], zh) + sp_ + o_.text, zh)
                            samp_.append((su_, len(su_) if o_.finish_reason == "stop" else len(su_) - 1))
                        for i_ in range(n, len(draft)):
                            rr_ = [su_ for su_, reach_ in samp_ if reach_ > i_ or reach_ == len(su_)]
                            cons_.append(round(sum(lcp(su_, draft) > i_ for su_ in rr_) / len(rr_), 3) if rr_ else -1)
                        rec["cons"] = cons_
                    if args.mode == "rollout":
                        pass
                    else:
                        rec["lp0"] = [round(float(x), 3) for x in lp0[n:]]
                        rec["lps"] = [round(float(x), 3) for x in lps[n:]]
                        obs = HC.observations(draft, it["prev"], d1, ds, lp0, lps, n)
                        t_ = it["t"]
                        r_t, vad_t = HN.rel_vad(it["audio"][(t_ - 1000) * 16: t_ * 16], it["v1"][t_][-16000:])
                        for i_, o_ in obs.items():
                            if type(HP).__name__ == "CParams":
                                o_ = o_ + (HCO.cbin(cons_[i_ - n]) if 0 <= i_ - n < len(cons_) else HCO.CB_NA,)
                            hh = it["H"].setdefault(i_, ([], []))
                            hh[1].append(HN.inputs_vec(r_t, vad_t, len(hh[0]), t_, zh)); hh[0].append(o_)
                        P_ = [HN.posterior_stable(it["H"][i_][0], it["H"][i_][1], HP) for i_ in range(n, len(draft))]
                        th_ = SPEC[it["v"]]["theta"]
                        base = max(n, min(agreed, cv)) if args.variant in ("n1", "s1") else n
                        if args.variant in ("n1", "n2"):
                            kH = 0
                            while kH < len(P_) and P_[kH] >= th_:
                                kH += 1
                            cut = max(base, n + kH)
                            if SPEC[it["v"]]["mode"] == "pandora":      # change 45: blocking-aware threshold
                                kap_ = SPEC[it["v"]]["kappa"]
                                j_ = cut - n
                                while j_ < len(P_):
                                    b_ = 0
                                    while j_ + 1 + b_ < len(P_) and P_[j_ + 1 + b_] >= th_:
                                        b_ += 1
                                    if P_[j_] >= max(0.5, 1.0 - kap_ * (1 + b_)):
                                        j_ += 1 + b_
                                        st["pandora_extra"] += 1 + b_
                                    else:
                                        break
                                cut = n + j_
                        elif args.variant == "r1":
                            cut, risk_ = n, 0.0
                            while cut - n < len(P_) and P_[cut - n] >= 0.5 and risk_ + 1.0 - P_[cut - n] <= th_:
                                risk_ += 1.0 - P_[cut - n]; cut += 1
                            rec.update(risk=round(risk_, 3))
                        else:
                            lam = th_ / (1.0 - th_)
                            ts = t_ / 1000.0
                            q = max(it["qend"], ts) + (base - n) * UNIT_S        # queue end after the base units
                            cut = base
                            while cut < len(draft):
                                b_ = min(1.0, max(0.0, ts + 1.0 - max(q, ts)))
                                if b_ <= 0.0 or P_[cut - n] < lam / (lam + b_):
                                    break
                                cut += 1; q = max(q, ts) + UNIT_S
                            rec.update(slack0=round(min(1.0, max(0.0, ts + 1.0 - max(it["qend"], ts))), 3))
                        rec.update(P=[round(float(x), 3) for x in P_], r=round(r_t, 3), theta_t=th_)
                        if args.cma:                                        # change 50
                            cb_, okb_ = strip_(it, resb[k].outputs[0].text)
                            st["heal_fail"] += int(not okb_)
                            sepb_ = "" if zh or not it["committed"] or cb_[:1].isspace() or cb_[:1] in ",.;:!?)" else " "
                            db_ = units(join(it["committed"], zh) + sepb_ + cb_, zh)
                            Lb_ = lcp(draft, db_)
                            st["cma_capped"] = st.get("cma_capped", 0) + int(cut > max(n, Lb_))
                            st["cma_units_removed"] = st.get("cma_units_removed", 0) + max(0, cut - max(n, Lb_))
                            cut = min(cut, max(n, Lb_))
                            rec["cma"] = Lb_ - n
                    rec.update(sv0=sv0)
                    rec.update(draft_sap=join(ds, zh), sv=svl, n_new=max(0, n_cut - n))
                    st["hsc_extra_units"] += max(0, cut - max(n_cut, n))
                    st["hsc_fewer_units"] += max(0, max(n_cut, n) - max(cut, n))
                    rec["draft_v1"] = join(d1, zh)
                    st["v1_differs"] += int(d1 != draft)
                    if d1 == draft:
                        assert n_cut == agreed, (it["id"], it["t"])
                if False and zh and cut > n:      # change 44 snap: not used in hsc16
                    dt_ = join(draft, zh)
                    c_ = cut
                    while c_ > n and not boundary(draft, c_, dt_):
                        c_ -= 1
                    rec["snap"] = [int(cut), int(c_)]
                    st["snap_cut_units"] += cut - c_
                    cut = c_
                if zh and cut > n:
                    rec["mid"] = int(not boundary(draft, cut, join(draft, zh)))
                    st["mid_commits"] += rec["mid"]
                    st["commits_nonfinal"] += 1
                new = draft[n: cut] if cut > n else []
                it["prev"] = draft
                it["hist"].append(draft)
                rec.update(la2_new=la2_new, cv_new=len(new))
                st["reads_nonfinal"] += 1
                st["cut_reads"] += int(len(new) < la2_new)
                st["la2_units"] += la2_new
                st["committed_units"] += len(new)
            if it["sample_now"]:      # the sampled action: tokens + their log-probs (old policy, for the RL ratio)
                o_ = res.outputs[0]
                rec["forked"] = True
                rec["fork_prefix"] = join(it["committed"], zh) if not it["done"] else None
                rec["fork_tok"] = list(o_.token_ids)
                rec["fork_lp"] = [round(float(o_.logprobs[i_][t_].logprob), 5) for i_, t_ in enumerate(o_.token_ids)]
                it["sample_now"] = False
            it["drafts"].append(rec)
            if new or it["done"]:
                text = ("" if zh or not n else " ") + join(new, zh)
                it["blocks"].append({"write_ms": it["t"], "text": text, "final": it["done"]})
                it["committed"] += new
                if SPEC[it["v"]]["mode"] != "base" and new and not it["done"]:     # change 44/45: heal state (heal, cec, pandora)
                    dt_ = join(draft, zh)
                    L_ = len(it["committed"])
                    if boundary(it["committed"], L_, dt_):
                        it["heal_k"] = None
                    else:
                        k0_ = L_ - 1
                        while k0_ > 0 and not boundary(it["committed"], k0_, dt_):
                            k0_ -= 1
                        it["heal_k"] = k0_
                        st["heal_set"] += 1
                if new:
                    it["qend"] = max(it["qend"], it["t"] / 1000.0) + len(new) * UNIT_S
        r += 1
        print(f"{tag} round {r}: {len(live)} streams ({len(forks)} forks so far), {len(vk)} v1 prompts, {time.time() - t0:.0f}s", flush=True)
    def row_(it):
        blocks = [b for b in it["blocks"] if b["text"].strip() or b["final"]]
        if blocks and not blocks[-1]["text"].strip() and len(blocks) > 1:
            blocks = blocks[:-1]
            blocks[-1]["final"] = True
        return {"id": it["id"], "fork_t": it["fork"][0], "fork_k": it["fork"][1], "translation": join(it["committed"], zh),
                "blocks": blocks, "drafts": [d_ for d_ in it["drafts"] if d_["t"] >= it["fork"][0]]}
    if forks:
        with open(out.parent / f"forks_{args.direction}.jsonl", "x", encoding="utf-8") as fh:
            for it in forks:
                fh.write(json.dumps(row_(it), ensure_ascii=False) + "\n")
    fhs = {v_: open(o_, "x", encoding="utf-8") for v_, o_ in outs.items()}
    for it in items:
        fh = fhs[it["v"]]
        if True:
            blocks = [b for b in it["blocks"] if b["text"].strip() or b["final"]]
            if blocks and not blocks[-1]["text"].strip() and len(blocks) > 1:   # an empty last commit: the one before is the last
                blocks = blocks[:-1]
                blocks[-1]["final"] = True
            row = {"id": it["id"], "translation": join(it["committed"], zh), "blocks": blocks, "drafts": it["drafts"], "fork_at": sorted(it["fork_at"])}
            if args.assign:
                row["alpha"] = alpha_of[it["id"]]
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    for fh in fhs.values():
        fh.close()
    st = dict(st)
    st.update(method=f"HSMM {args.variant} (change 45 variants)", variants=VARS, fork_reads=args.fork_reads, fork_k=args.fork_k, fork_temp=args.fork_temp, fork_top_p=args.fork_top_p, forks=len(forks), variant=args.variant, params=args.params, theta=args.theta, verifier=args.verifier, second_view=sv, thinker=args.thinker, assign=args.assign, lora_root=args.lora_root, const_alpha=args.const_alpha,
              views=args.views, budget_from=args.budget_from, budget_pool_meta=args.budget_pool_meta, items=len(items), seconds=round(time.time() - t0, 1),
              items_sha1=hashlib.sha1(open(args.items_file, "rb").read()).hexdigest())
    (Path(args.out) / f"job_{args.direction}.combo.json").write_text(json.dumps(st), encoding="utf-8")
    print(f"{tag}: {len(items)} sentences, {time.time() - t0:.0f}s -> {out}", flush=True)
    print("combo thinker:", json.dumps(st), flush=True)


if __name__ == "__main__":
    main()
