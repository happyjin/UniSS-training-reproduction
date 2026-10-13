"""Matched listening set: the same utterance spoken both ways.

The two arms share a talker and a translation, so anything audible
between them is the 3.5% state difference and nothing else.
"""
import json, shutil, pathlib, sacrebleu

ROOT = pathlib.Path("/opt/dlami/nvme/jasonleeeli/projects/UniSS")
A = ROOT / "eval_outputs/uniss_qwen3_omni_v1/e2e_iter4000"
B = ROOT / "eval_outputs/uniss_qwen3_omni_v1/e2e_iter4000_sp"
OUT = ROOT / "reports/uniss_qwen3_omni_v1/LISTEN_E2E_ITER4000"

def load(d):
    return {r["id"] + "|" + r["tgt_lang"]: r
            for r in (json.loads(l) for l in open(d / "asr.jsonl"))}

two, one = load(A), load(B)

def sbleu(hyp, ref, lang):
    if not hyp or not hyp.strip():
        return 0.0
    tok = "zh" if lang == "cmn" else "13a"
    return sacrebleu.sentence_bleu(hyp, [ref], tokenize=tok).score

rows = []
for k, r in two.items():
    if k not in one:
        continue
    lang = r["tgt_lang"]
    ref = r.get("translation_ref") or ""
    rows.append({
        "k": k, "lang": lang, "id": r["id"],
        "ref": ref,
        "mt": r.get("translated_text") or r.get("generated_text_clean") or "",
        "asr2": two[k].get("asr_text") or "",
        "asr1": one[k].get("asr_text") or "",
        "b2": sbleu(two[k].get("asr_text") or "", ref, lang),
        "b1": sbleu(one[k].get("asr_text") or "", ref, lang),
        "src": r.get("source_audio_path"), "refwav": r.get("reference_audio_path"),
    })

OUT.mkdir(parents=True, exist_ok=True)
picked, lines = [], []
for lang, label in (("eng", "zh→en"), ("cmn", "en→zh")):
    g = sorted([r for r in rows if r["lang"] == lang], key=lambda r: -r["b2"])
    if not g:
        continue
    # best, median, worst by the two-pass score: the range, not the highlights
    sel = [("好", g[0]), ("中位", g[len(g) // 2]), ("差", g[-1])]
    for tag, r in sel:
        n = len(picked) + 1
        stem = f"{n:02d}_{label.replace('→','2')}_{tag}"
        for src, dst in ((r["src"], f"{stem}_源.wav"),
                         (r["refwav"], f"{stem}_参考.wav")):
            if src and pathlib.Path(src).exists():
                shutil.copy(src, OUT / dst)
        # the record already carries the path of the wav it was scored on
        for rec, suffix in ((two[r["k"]], "两次"), (one[r["k"]], "一次")):
            w = rec.get("audio_path")
            if w and pathlib.Path(w).exists():
                shutil.copy(w, OUT / f"{stem}_{suffix}.wav")
            else:
                print(f"  缺 {stem}_{suffix}: {w}")
        picked.append((stem, tag, label, r))

L = ["# iter 4,000 端到端试听：两次调用 vs 一次调用", "",
     "同一个 Talker（checkpoint 相同）、同一句译文（两臂 Text-BLEU 完全相同，",
     "38.34 / 39.24）。听到的任何差别，只可能来自隐状态那 3.5% 的数值扰动。", "",
     "每组四个文件：`_源` 输入音频，`_参考` 目标语参考音频，",
     "`_两次` 现行两次调用的输出，`_一次` 单次调用的输出。", ""]
for stem, tag, label, r in picked:
    L += [f"## {stem}  （{label}，{tag}）", "",
          f"- 参考译文：{r['ref']}",
          f"- 模型译文（两臂相同）：{r['mt']}",
          f"- 两次调用 ASR：{r['asr2']}   — 句级 BLEU {r['b2']:.1f}",
          f"- 一次调用 ASR：{r['asr1']}   — 句级 BLEU {r['b1']:.1f}", ""]
(OUT / "README.zh-CN.md").write_text("\n".join(L), encoding="utf-8")
print(f"{len(picked)} 组 -> {OUT}")
for stem, tag, label, r in picked:
    print(f"  {stem}: 两次 {r['b2']:.1f} / 一次 {r['b1']:.1f}")
