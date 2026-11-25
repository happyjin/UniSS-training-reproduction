"""sst_hsmm_v1 prereg change 23: PC-MBR = commitment-constrained Bayes decision at the final read (inference only).
gen:    per item of a streaming blocks file (committed prefix P = the non-final blocks), WHOLE audio, wave17:
        E = unconstrained translations (head, greedy + 7 samples T 0.7), H = P (+) continuations (head + P, greedy + 7 samples);
        P empty -> H = E (not generated twice). -> {"id", "P", "E": [...], "H": [...]} (H as full translations, unit-joined)
select: B (main) = argmax_{h in H} mean_{e in E} chrF(h, e) (self pair excluded when H is E); A (ablation, plain MBR) =
        argmax_{h in H} mean_{e in H, e != h} chrF(h, e); ties -> greedy (index 0). Writes blocks files with the final
        block replaced (non-final blocks untouched).
    <vllm python> -m experiments.sst_hsmm_v1.prmbr gen --blocks B --items I --direction d --shard k --num-shards n --out O
    <qwen3omni python> -m experiments.sst_hsmm_v1.prmbr select --blocks B --gens 'dir/gen_*.jsonl' --direction d --out-b O1 --out-a O2
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
QWEN3 = str(REPO / "pretrained_models/Qwen3-Omni-30B-A3B-Instruct")
WAVE17 = str(REPO / "eval_outputs/mist_w3po_v1/wave17/merged_enc_wise05")
K, TEMP, SEED = 8, 0.7, 20261125


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def prefix_of(r, zh):
    from experiments.uniss_qwen3_omni_v1.evaluation.si_thinker_stream import join, units
    k = sum(len(units(b["text"], zh)) for b in r["blocks"] if not b["final"])
    return join(units(r["translation"], zh)[:k], zh), k


def gen(a):
    if Path(a.out).exists():
        raise SystemExit(f"拒绝覆盖已有输出: {a.out}")
    import soundfile as sf
    from transformers import Qwen3OmniMoeProcessor
    from vllm import LLM, SamplingParams
    from experiments.uniss_qwen3_omni_v1.training.e2_batch import inference_instruction
    from experiments.uniss_qwen3_omni_v1.evaluation.si_thinker_stream import join, units
    zh = a.direction == "en2zh"
    items = {r["id"]: r for r in jl(a.items)}
    rows = [r for i, r in enumerate(jl(a.blocks)) if i % a.num_shards == a.shard and r["id"] in items]
    proc = Qwen3OmniMoeProcessor.from_pretrained(QWEN3)
    conv = [{"role": "user", "content": [{"type": "audio", "audio": "x.wav"},
                                         {"type": "text", "text": inference_instruction()["cmn" if zh else "eng"]}]}]
    head = proc.apply_chat_template([conv], add_generation_prompt=True, tokenize=False)[0]
    llm = LLM(model=a.thinker, gpu_memory_utilization=a.gpu_mem, max_model_len=8192, limit_mm_per_prompt={"audio": 1},
              seed=SEED, tensor_parallel_size=1)
    g = SamplingParams(temperature=0.0, max_tokens=256, stop_token_ids=[151645])
    s = SamplingParams(n=K - 1, temperature=TEMP, max_tokens=256, stop_token_ids=[151645], seed=SEED)
    prompts, kinds = [], []
    for r in rows:
        x, sr = sf.read(items[r["id"]]["audio"], dtype="float32", always_2d=True)
        assert sr == 16000
        au = x.mean(axis=1)
        P, _ = prefix_of(r, zh)
        r["P"] = P
        for pre in ([""] + ([P] if P else [])):
            for sp in (g, s):
                prompts.append({"prompt": head + pre, "multi_modal_data": {"audio": (au, 16000)}}); kinds.append(sp)
    outs = llm.generate(prompts, kinds, use_tqdm=False)
    it = iter(outs)

    def full(pre, cont):
        sep = "" if zh or not pre or cont[:1].isspace() or cont[:1] in ",.;:!?)" else " "
        return join(units((pre + sep + cont).strip(), zh), zh)
    with open(a.out + ".tmp", "x", encoding="utf-8") as fh:
        for r in rows:
            E = [full("", o.text) for o in next(it).outputs] + [full("", o.text) for o in next(it).outputs]
            H = ([full(r["P"], o.text) for o in next(it).outputs] + [full(r["P"], o.text) for o in next(it).outputs]) if r["P"] else None
            fh.write(json.dumps({"id": r["id"], "P": r["P"], "E": E, "H": H}, ensure_ascii=False) + "\n")
    Path(a.out + ".tmp").rename(a.out)
    print("done", len(rows), flush=True)


def select(a):
    for o in (a.out_b, a.out_a):
        if Path(o).exists():
            raise SystemExit(f"拒绝覆盖已有输出: {o}")
    import sacrebleu
    from experiments.uniss_qwen3_omni_v1.evaluation.si_thinker_stream import join, units
    zh = a.direction == "en2zh"
    G = {r["id"]: r for p in sorted(glob.glob(a.gens)) for r in jl(p)}
    ch = sacrebleu.metrics.CHRF()

    def pick(H, E, same):
        best, bi = -1.0, 0
        for i, h in enumerate(H):
            sc = [ch.sentence_score(h, [e]).score for j, e in enumerate(E) if not (same and j == i)]
            v = sum(sc) / max(1, len(sc))
            if v > best + 1e-9:
                best, bi = v, i
        return H[bi]
    nb, nchg_b, nchg_a = 0, 0, 0
    with open(a.out_b, "x", encoding="utf-8") as fb, open(a.out_a, "x", encoding="utf-8") as fa:
        for r in jl(a.blocks):
            g = G[r["id"]]
            P, k = prefix_of(r, zh)
            assert P == g["P"], r["id"]
            H, same = (g["H"], False) if g["H"] else (g["E"], True)
            for sel, fh, tag in ((pick(H, g["E"], same), fb, "b"), (pick(H, H, True), fa, "a")):
                u = units(sel, zh)
                if u[:k] != units(P, zh):                       # joining artefact: keep the original final read
                    u = units(r["translation"], zh)
                bl = [b for b in r["blocks"] if not b["final"]]
                fin = [b for b in r["blocks"] if b["final"]]
                last = dict(fin[-1]) if fin else {"write_ms": r["blocks"][-1]["write_ms"], "final": True}
                last["text"] = join(u[k:], zh)
                out = {**r, "translation": join(u, zh), "blocks": bl + [last]}
                if tag == "b":
                    nchg_b += out["translation"] != r["translation"]
                else:
                    nchg_a += out["translation"] != r["translation"]
                fh.write(json.dumps(out, ensure_ascii=False) + "\n")
            nb += 1
    print(a.direction, "items", nb, "changed B", nchg_b, "A", nchg_a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("gen", "select"))
    ap.add_argument("--blocks", required=True)
    ap.add_argument("--items", default="")
    ap.add_argument("--direction", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--out", default="")
    ap.add_argument("--thinker", default=WAVE17)
    ap.add_argument("--gpu-mem", type=float, default=0.6)
    ap.add_argument("--gens", default="")
    ap.add_argument("--out-b", default="")
    ap.add_argument("--out-a", default="")
    a = ap.parse_args()
    gen(a) if a.mode == "gen" else select(a)


if __name__ == "__main__":
    main()
