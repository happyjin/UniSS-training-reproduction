"""Robust wave 9: si_thinker_stream (shared, NOT modified) with each item's 8 anchor tokens after <|audio_end|>.

A wrapper, not a copy: vllm.LLM.generate is patched in this process before si_thinker_stream.main() runs, so the LA2 /
w1 code path, items, reads, sampling and outputs are si_thinker_stream's own. Every prompt si_thinker_stream sends
({"prompt": head + committed, "multi_modal_data": {"audio": (a[:heard], 16000)}}) is turned into
{"prompt_token_ids": insert_after_audio(tokenizer.encode(prompt), anchor), "multi_modal_data": same} -- vLLM text prompts
cannot name ids the tokenizer does not know. The item's anchor comes from wave9/anchor/table_<dir>.jsonl by the sha1 of
the prompt audio's first second (the anchor is a function of exactly those samples). --anchor none inserts nothing (the
token-id path alone, for the reproduction check). --anchor-check N: on the first generate call the first N prompts are
run twice as text prompts and once as anchor-free token prompts; the engine's prompt ids (after the audio placeholder
expansion) must be identical, else exit; output agreement text-vs-text and text-vs-token is printed (repeated vLLM calls
are not bit-reproducible: prefix cache / batch shapes).

    <vllm_omni_030 python> -m experiments.mist_w3po_v1.voicecond.wave9_si_thinker_anchor --anchor table|none \
        [--anchor-check 64] <si_thinker_stream arguments>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from experiments.mist_w3po_v1.voicecond import wave9_common as C  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--anchor", choices=("table", "none"), required=True)
    ap.add_argument("--anchor-check", type=int, default=0)
    a, rest = ap.parse_known_args()
    direction = rest[rest.index("--direction") + 1]
    table = {}
    if a.anchor == "table":
        for r in map(json.loads, open(C.ANCHOR_DIR / f"table_{direction}.jsonl", encoding="utf-8")):
            assert table.get(r["key"], r["ids"]) == r["ids"]
            table[r["key"]] = r["ids"]
            assert len(r["ids"]) == C.N_DIM and all(C.ANCHOR_LO <= v < C.VOCAB for v in r["ids"])
    import vllm
    orig = vllm.LLM.generate
    st = {"calls": 0, "prompts": 0, "anchored": 0, "shown": False}

    def gen(self, prompts, sampling_params=None, *args, **kw):
        tok = self.get_tokenizer()
        ps = prompts if isinstance(prompts, list) else [prompts]
        new = []
        for p in ps:
            assert isinstance(p, dict) and set(p) == {"prompt", "multi_modal_data"}, type(p)
            ids = tok.encode(p["prompt"])
            anc = table[C.head_key(p["multi_modal_data"]["audio"][0])] if a.anchor == "table" else []
            new.append({"prompt_token_ids": C.insert_after_audio(ids, anc), "multi_modal_data": p["multi_modal_data"]})
            st["anchored"] += bool(anc)
        if not st["shown"] and new:
            ids = new[0]["prompt_token_ids"]
            k = ids.index(C.AUDIO_END)
            print("wave9 anchor: prompt ids around <|audio_end|>:", ids[k - 2: k + C.N_DIM + 4], flush=True)
            st["shown"] = True
        if a.anchor_check and st["calls"] == 0:
            n = min(a.anchor_check, len(ps))
            sp = sampling_params[:n] if isinstance(sampling_params, list) else sampling_params
            r_s = orig(self, ps[:n], sp, use_tqdm=False)
            r_s2 = orig(self, ps[:n], sp, use_tqdm=False)
            r_t = orig(self, [{"prompt_token_ids": tok.encode(p["prompt"]), "multi_modal_data": p["multi_modal_data"]}
                              for p in ps[:n]], sp, use_tqdm=False)
            same_ids = all(list(x.prompt_token_ids) == list(y.prompt_token_ids) for x, y in zip(r_s, r_t))
            agree_tt = sum(x.outputs[0].text == y.outputs[0].text for x, y in zip(r_s, r_s2))
            agree_tk = sum(x.outputs[0].text == y.outputs[0].text for x, y in zip(r_s, r_t))
            print("wave9 anchor check:", json.dumps({"n": n, "same_prompt_ids": same_ids, "agree_text_text": agree_tt,
                                                     "agree_text_token": agree_tk}), flush=True)
            if not same_ids:
                raise SystemExit("wave9 anchor check FAILED: the token-id path gives other prompt ids than the text path")
        st["calls"] += 1
        st["prompts"] += len(ps)
        out = orig(self, new, sampling_params, *args, **kw)
        return out if isinstance(prompts, list) else out[:1]

    vllm.LLM.generate = gen
    sys.argv = [sys.argv[0]] + rest
    from experiments.uniss_qwen3_omni_v1.evaluation import si_thinker_stream as S
    S.main()
    print("wave9 anchor:", json.dumps({"mode": a.anchor, **{k: v for k, v in st.items() if k != "shown"}}), flush=True)


if __name__ == "__main__":
    main()
