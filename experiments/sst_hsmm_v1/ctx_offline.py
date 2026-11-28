"""change 68 no-harm precheck: whole-audio greedy translation (wave17, hsc30's prompt) of a dev set WITH vs WITHOUT session
context. Pseudo-sessions = 10 consecutive items in file order (unrelated sentences, as the OBSI test sessions); the context of item j
= the previous N outputs of the context arm in its session (zh joined without space, en with a space), appended to the instruction
exactly as hsc30 --ctx. Round j decodes item j of every session (both arms in one engine).
gen:   <vllm python> -m experiments.sst_hsmm_v1.ctx_offline gen --items <items_d.jsonl> --direction d --n 3 --out <dir>
score: <qwen3omni python> -m experiments.sst_hsmm_v1.ctx_offline score --root <dir with <ds>_<d>/> --out <json>
       GO iff for both directions the mean over the dev sets of BLEU(ctx) - BLEU(plain) >= -0.3 (official Bleu). Refuses overwrite."""
from __future__ import annotations

import argparse, glob, json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
QWEN3 = str(REPO / "pretrained_models/Qwen3-Omni-30B-A3B-Instruct")
WAVE17 = str(REPO / "eval_outputs/mist_w3po_v1/wave17/merged_enc_wise05")
CTX_TXT = "\nFor reference, the translation of the preceding speech in this conversation: "


def gen(a):
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"refuse to overwrite {out}")
    import soundfile as sf
    from transformers import Qwen3OmniMoeProcessor
    from vllm import LLM, SamplingParams
    from experiments.uniss_qwen3_omni_v1.training.e2_batch import inference_instruction
    rows = [json.loads(l) for l in open(a.items, encoding="utf-8") if l.strip()]
    zh = a.direction == "en2zh"
    proc = Qwen3OmniMoeProcessor.from_pretrained(QWEN3)
    instr = inference_instruction()["cmn" if zh else "eng"]

    def head(ctx):
        conv = [{"role": "user", "content": [{"type": "audio", "audio": "x.wav"}, {"type": "text", "text": instr + (CTX_TXT + ctx if ctx else "")}]}]
        return proc.apply_chat_template([conv], add_generation_prompt=True, tokenize=False)[0]
    llm = LLM(model=WAVE17, gpu_memory_utilization=a.gpu_mem, max_model_len=8192, limit_mm_per_prompt={"audio": 1},
              seed=20261128, tensor_parallel_size=1)
    sp = SamplingParams(temperature=0.0, max_tokens=256, stop_token_ids=[151645])
    sess = [rows[i:i + 10] for i in range(0, len(rows), 10)]
    hist = [[] for _ in sess]
    outp, outc = {}, {}
    for j in range(10):
        live = [(s, ss[j]) for s, ss in enumerate(sess) if j < len(ss)]
        if not live:
            break
        prompts, owner = [], []
        for s, r in live:
            x, sr = sf.read(r["audio"], dtype="float32", always_2d=True)
            assert sr == 16000, r["audio"]
            au = (x.mean(axis=1), 16000)
            ctx = ("" if zh else " ").join(h for h in hist[s][-a.n:] if h)
            prompts += [{"prompt": head(""), "multi_modal_data": {"audio": au}}, {"prompt": head(ctx), "multi_modal_data": {"audio": au}}]
            owner.append((s, r))
        res = llm.generate(prompts, sp, use_tqdm=False)
        for q, (s, r) in enumerate(owner):
            outp[r["id"]] = res[2 * q].outputs[0].text.strip()
            outc[r["id"]] = res[2 * q + 1].outputs[0].text.strip()
            hist[s].append(outc[r["id"]])
        print("round", j, len(live), flush=True)
    out.mkdir(parents=True)
    for name, O in (("plain", outp), ("ctx", outc)):
        with open(out / f"{name}.jsonl", "x", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps({"id": r["id"], "translation": O[r["id"]]}, ensure_ascii=False) + "\n")
    print("done", len(rows), flush=True)


def score(a):
    if Path(a.out).exists():
        raise SystemExit(f"refuse to overwrite {a.out}")
    from experiments.mist_w3po_v1.voicecond.wave2_s1_score import Bleu
    D0 = REPO / "eval_outputs/sst_hsmm_v1/pfb0/dev"
    res = {}
    for d in ("en2zh", "zh2en"):
        bl = Bleu("cmn" if d == "en2zh" else "eng"); dl = []
        for ds in ("cvdev", "edev"):
            ref = {r["id"]: r["translation"] for r in map(json.loads, open(D0 / ds / "data" / f"items_{d}.jsonl", encoding="utf-8"))}
            sc = {}
            for arm in ("plain", "ctx"):
                O = {r["id"]: r["translation"] for r in map(json.loads, open(f"{a.root}/{ds}_{d}/{arm}.jsonl", encoding="utf-8"))}
                ids = [i for i in ref if i in O]
                sc[arm] = round(bl.score(bl.stats([O[i] for i in ids], [ref[i] for i in ids])), 2)
            sc["delta"] = round(sc["ctx"] - sc["plain"], 2); sc["n"] = len(ids)
            res[f"{d}/{ds}"] = sc; dl.append(sc["delta"])
        res[d] = {"mean_delta": round(sum(dl) / len(dl), 3), "go": sum(dl) / len(dl) >= -0.3}
        print(d, res[d], {k: v for k, v in res.items() if k.startswith(d + "/")}, flush=True)
    res["go"] = res["en2zh"]["go"] and res["zh2en"]["go"]
    json.dump(res, open(a.out, "x"), indent=1)
    print("GO" if res["go"] else "NOGO")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("gen", "score"))
    ap.add_argument("--items"); ap.add_argument("--direction", choices=("en2zh", "zh2en"))
    ap.add_argument("--n", type=int, default=3); ap.add_argument("--gpu-mem", type=float, default=0.6)
    ap.add_argument("--root"); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    gen(a) if a.mode == "gen" else score(a)


if __name__ == "__main__":
    main()
