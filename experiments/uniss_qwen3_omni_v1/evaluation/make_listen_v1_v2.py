"""v1 against v2 on the same utterances, the format fix made audible.

v1 iter 8000 read states whose reply carried a transcript prefix it was
trained on but inference never writes. v2 iter 2000 reads the reply
inference actually produces. Same talker lineage, same 64 utterances,
same frozen thinker -- the translations are character-identical, so all
of the difference is in the speaking.
"""
import json, shutil, pathlib, sacrebleu

ROOT = pathlib.Path("/opt/dlami/nvme/jasonleeeli/projects/UniSS")
V1 = ROOT / "eval_outputs/uniss_qwen3_omni_v1/v1_e2e_iter8000"
V2 = ROOT / "eval_outputs/uniss_qwen3_omni_v1/e2e_iter2000"
OUT = ROOT / "reports/uniss_qwen3_omni_v1/LISTEN_V1_vs_V2"

def load(d):
    return {r["id"] + "|" + r["tgt_lang"]: r
            for r in (json.loads(l) for l in open(d / "asr.jsonl"))}

a, b = load(V1), load(V2)
def sb(h, r, lang):
    if not h or not h.strip(): return 0.0
    return sacrebleu.sentence_bleu(h, [r], tokenize="zh" if lang == "cmn" else "13a").score

rows = []
for k, r1 in a.items():
    r2 = b.get(k)
    if r2 is None: continue
    lang = r1["tgt_lang"]; ref = r1.get("translation_ref") or ""
    rows.append(dict(k=k, lang=lang, ref=ref,
                     mt1=r1.get("translated_text") or "", mt2=r2.get("translated_text") or "",
                     a1=r1.get("asr_text") or "", a2=r2.get("asr_text") or "",
                     b1=sb(r1.get("asr_text") or "", ref, lang),
                     b2=sb(r2.get("asr_text") or "", ref, lang),
                     w1=r1.get("audio_path"), w2=r2.get("audio_path"),
                     src=r1.get("source_audio_path"), rw=r1.get("reference_audio_path")))

same_mt = sum(r["mt1"] == r["mt2"] for r in rows)
OUT.mkdir(parents=True, exist_ok=True)
L = ["# v1 vs v2：格式修正的听感对照", "",
     f"同一批 {len(rows)} 条，同一冻结 Thinker。译文逐字相同的 {same_mt}/{len(rows)} 条，",
     "所以听到的差别全部来自 Talker 读到的隐状态格式。", "",
     "    v1 iter 8,000   回复带转写前缀（训练格式），推理给的是单行   mean 12.06",
     "    v2 iter 2,000   训练与推理都是单行译文                      mean 18.53", "",
     "每组：`_源` 输入，`_参考` 目标语参考，`_v1` 旧格式，`_v2` 修正后。", ""]
picked = []
for lang, label in (("eng", "zh2en"), ("cmn", "en2zh")):
    g = [r for r in rows if r["lang"] == lang]
    g.sort(key=lambda r: -(r["b2"] - r["b1"]))          # v2 赢得最多的在前
    sel = [("v2大胜", g[0]), ("v2小胜", g[len(g)//4]),
           ("持平", g[len(g)//2]), ("v1更好", g[-1])]
    for tag, r in sel:
        n = len(picked) + 1
        stem = f"{n:02d}_{label}_{tag}"
        for src, dst in ((r["src"], "源"), (r["rw"], "参考"),
                         (r["w1"], "v1"), (r["w2"], "v2")):
            if src and pathlib.Path(src).exists():
                shutil.copy(src, OUT / f"{stem}_{dst}.wav")
        picked.append((stem, label, tag, r))
        L += [f"## {stem}", "",
              f"- 参考译文：{r['ref']}",
              f"- 模型译文：{r['mt2']}" + ("" if r["mt1"] == r["mt2"] else f"  （v1 生成的不同：{r['mt1']}）"),
              f"- **v1** ASR：{r['a1']}   — 句级 BLEU {r['b1']:.1f}",
              f"- **v2** ASR：{r['a2']}   — 句级 BLEU {r['b2']:.1f}", ""]

wins = sum(r["b2"] > r["b1"] for r in rows)
L += ["## 全量 128 条", "",
      f"    v2 更好 {wins} 条，v1 更好 {sum(r['b1']>r['b2'] for r in rows)} 条，"
      f"持平 {sum(abs(r['b1']-r['b2'])<1e-9 for r in rows)} 条", ""]
(OUT / "README.zh-CN.md").write_text("\n".join(L), encoding="utf-8")
print(f"{len(picked)} 组 -> {OUT}")
print(f"  译文逐字相同 {same_mt}/{len(rows)}；v2 更好 {wins}/{len(rows)} 条")
for stem, label, tag, r in picked:
    print(f"  {stem}: v1 {r['b1']:5.1f} -> v2 {r['b2']:5.1f}")
