"""Every finished RealSI run against the published systems, in one report (markdown + JSON).

Scans eval_outputs/uniss_qwen3_omni_v1 for si_infer runs (realsi_<dir>/metrics.json),
and writes:

  1. the whole system (text from si_thinker_stream, policy from the run's name:
     hold / la2) per read window against SimulS2ST-Omni's Table 3 (sentence level,
     ASR-BLEU, Whisper-large-v3 / Paraformer-zh as here);
  2. matched latency: LiveInterpret 2.0's own sentence-level RealSI S2S LAAL (en-zh
     3.12 s, zh-en 3.79 s; ASR-BLEU 29.09 / 22.19 as SimulS2ST-Omni quotes it), which
     SimulS2ST-Omni's m5 is matched to -- our best run within that LAAL;
  3. our quality-latency points, sorted by LAAL;
  4. the talker alone (the reference translation handed over), per checkpoint;
  5. findings, written from the numbers.

Directions are matched by output language (our en2zh: Chinese speech out). Our LAAL
is on force-aligned output word ends against the source duration, computation not
counted; the others' are as their papers report them.

    python -m experiments.uniss_qwen3_omni_v1.evaluation.si_report [--out eval_outputs/uniss_qwen3_omni_v1/si_report]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path("eval_outputs/uniss_qwen3_omni_v1")
OMNI = {"en2zh": {2: 24.21, 3: 25.01, 4: 25.54, 5: 25.92, 6: 26.32},
        "zh2en": {2: 19.55, 3: 21.07, 4: 22.47, 5: 22.94, 6: 23.47}}
LIVE = {"en2zh": (29.09, 3.12), "zh2en": (22.19, 3.79)}   # ASR-BLEU (SimulS2ST-Omni's quote), LAAL (its paper)
MAIN_TALKER = ("si_t1c", 1500)   # the talker the auto-extension rule kept; other talkers show where it has no run
NAME = {"en2zh": "英→中（输出中文）", "zh2en": "中→英（输出英文）"}


def checkpoint(path: str) -> tuple[str, int]:
    """(run family, step) from where a run sits."""
    fam = next((p.split("_2026")[0] for p in path.split("/") if p.startswith("si_t1")), "R2")
    for pat in (r"full_iter0*(\d+)", r"system(\d+)_", r"talker(\d+)_", r"probe_iter0*(\d+)", r"probe_(\d+)/", r"export_iter0*(\d+)"):
        m = re.search(pat, path)
        if m:
            return fam, int(m.group(1))
    if "probe_si_t1d" in path:
        return re.search(r"probe_(si_t1d\d)", path).group(1), 1000
    return fam, 0


def collect() -> list[dict]:
    from evaluation.text_metrics import corpus_bleu
    rows = []
    for mf in sorted(ROOT.rglob("realsi_*/metrics.json")):
        run = mf.parent
        d = run.name.split("_", 1)[1]
        try:
            m = json.load(open(mf))
            recs = [json.loads(l) for l in open(run / "results.jsonl", encoding="utf-8")]
        except Exception:
            continue
        x = json.load(open(run / "metrics_extra.json")) if (run / "metrics_extra.json").exists() else {}
        first = recs[0]
        policy = first.get("text_policy", "clause")
        parent = str(run.parent.relative_to(ROOT))
        if policy == "file":   # system[step]_<commit rule>_m<window>: hold, la2, vote8g0.8[_snap_sp], head...
            m_kind = re.match(r"system\d*_(.+)_m[\d.]+[a-z]?$", run.parent.name)
            kind = m_kind.group(1) if m_kind else ("la2" if "la2" in run.parent.name else "hold")
        else:
            kind = f"talker-{policy}"
        fam, step = checkpoint(parent + "/")
        lang = first["tgt_lang"]
        tb = corpus_bleu([r["translated_text"] for r in recs], [r["translation_ref"] for r in recs], language=lang)["score"]
        rows.append({"dir": d, "kind": kind, "window": float(first.get("window_s", 0)), "family": fam, "step": step,
                     "path": parent, "n": m["sentences"], "asr_bleu": m["asr_bleu"], "text_bleu": round(tb, 2),
                     "laal": x.get("laal_s"), "start": x.get("start_offset_aligned_s_p50", m.get("start_offset_s_p50")),
                     "end": m.get("end_offset_s_p50"), "silence": m.get("silence_ratio_mean"), "apcp": x.get("apcp"),
                     "sim": x.get("sim_src"), "stretched": m.get("stretched_blocks"),
                     "speed_policy": first.get("speed_policy", "fixed")})
    return rows


def fmt(v, unit="", nd=2):
    return "—" if v is None else (f"{v:.{nd}f}{unit}" if isinstance(v, float) else f"{v}{unit}")


def tier(w: float) -> str:
    return f"m{w:g}"


def pool_seeds(rows: list[dict]) -> list[dict]:
    """One row per setting: runs that differ only in the talker's seed (-seedN) become their mean."""
    groups = {}
    for r in rows:
        base = re.sub(r"-seed\d+$", "", r["kind"])
        groups.setdefault((base, r["family"], r["step"], r["window"], r["dir"]), []).append(r)
    out = []
    for (base, *_), v in groups.items():
        m = dict(v[0], kind=base, seeds=len(v))
        for k in ("asr_bleu", "laal", "start", "end", "silence", "apcp", "sim", "stretched"):
            xs = [x[k] for x in v if isinstance(x.get(k), (int, float)) and x[k] == x[k]]
            m[k] = (round(sum(xs) / len(xs), 2 if k in ("asr_bleu", "laal", "start", "end") else 3) if xs else None)
        out.append(m)
    return out


def seeds_note(r: dict) -> str:
    return f"（{r['seeds']} 个种子均值）" if r.get("seeds", 1) > 1 else ""


def report(rows: list[dict]) -> str:
    sys_rows = [r for r in pool_seeds(rows) if not r["kind"].startswith("talker") and r["n"] >= 300]
    out = ["# RealSI 同传结果与已有工作对比（自动生成）", "",
           "我们的系统：冻结的 Qwen3-Omni 思考模型（流式重新翻译，hold-n 或 LocalAgreement-2 定稿）+ 流式 UniSS 说话模型（T1 系列）。"
           "RealSI 句子级全量（英→中 346 句，中→英 431 句），ASR 与 SimulS2ST-Omni 相同（Whisper-large-v3 / Paraformer-zh）。"
           "方向按输出语言对齐；我们的 LAAL 为输出语音强制对齐后的词结束时刻，不含计算时间。",
           "定稿规则：hold = Hold-n；la2 = LocalAgreement-2；voteNgG = 共识投票（每次读入采 N 个续写，至少 G 比例一致的前缀才定稿，RALCP）；"
           "_snap = 读入截在停顿处；_sp = 流式提示；headXtT = 单次前向的定稿头（阈值 T）。", ""]
    latest = {}
    for r in sys_rows:
        key = (r["kind"], r["dir"], r["window"])
        rank = lambda x: ((x["family"], x["step"]) == MAIN_TALKER, x["family"], x["step"])
        if key not in latest or rank(r) > rank(latest[key]):
            latest[key] = r
    findings = []
    for d in ("en2zh", "zh2en"):
        out += [f"## {NAME[d]}", "", "### 1. 相同读窗档位 vs SimulS2ST-Omni（说话模型优先 T1c@1500，其余取最新检查点）", "",
                "| 定稿 | 档位 | 检查点 | 我们 ASR-BLEU | SimulS2ST-Omni | 差 | 思考模型文字 BLEU | 首次发音 | LAAL | 结尾落后 | 静音比 | A.PCP | 源音色 | 拖长段 |",
                "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        wins = losses = 0
        for kind in sorted({k for k, dd, _ in latest if dd == d}, key=lambda k: (k not in ("hold", "la2"), k)):
            for (k, dd, w), r in sorted(latest.items(), key=lambda t: t[0][2]):
                if k != kind or dd != d:
                    continue
                ref = OMNI[d].get(int(w)) if float(w).is_integer() else None
                diff = None if ref is None else round(r["asr_bleu"] - ref, 2)
                if diff is not None:
                    wins += diff > 0
                    losses += diff <= 0
                out.append(f"| {kind}{seeds_note(r)} | {tier(w)} | {r['family']}@{r['step']} | **{r['asr_bleu']:.2f}** | {fmt(ref)} | "
                           f"{'—' if diff is None else f'{diff:+.2f}'} | {r['text_bleu']:.2f} | {fmt(r['start'], 's')} | "
                           f"{fmt(r['laal'], 's')} | {fmt(r['end'], 's')} | {fmt(r['silence'], '', 3)} | {fmt(r['apcp'], '', 3)} | "
                           f"{fmt(r['sim'], '', 3)} | {fmt(r['stretched'], '', 3)} |")
        bleu_live, laal_live = LIVE[d]
        within = [r for r in latest.values() if r["dir"] == d and r["laal"] is not None and r["laal"] <= laal_live + 0.05]
        best = max(within, key=lambda r: r["asr_bleu"]) if within else None
        out += ["", f"### 2. 同等延迟（LAAL ≤ {laal_live:.2f} s，LiveInterpret 2.0 的报告延迟；SimulS2ST-Omni m5 与之对齐）", "",
                "| 系统 | ASR-BLEU | LAAL |", "|---|---|---|",
                f"| LiveInterpret 2.0 | {bleu_live:.2f} | {laal_live:.2f} s |",
                f"| SimulS2ST-Omni m5 | {OMNI[d][5]:.2f} | ≈{laal_live:.2f} s（按其论文与 LiveInterpret 对齐） |"]
        if best:
            out.append(f"| **我们**（{best['kind']} {tier(best['window'])}，{best['family']}@{best['step']}{seeds_note(best)}） | **{best['asr_bleu']:.2f}** | {best['laal']:.2f} s |")
        out += ["", "### 3. 我们的质量–延迟点（按 LAAL 排序，最新检查点）", "", "| LAAL | ASR-BLEU | 定稿 | 档位 | 首次发音 |", "|---|---|---|---|---|"]
        for r in sorted((r for r in latest.values() if r["dir"] == d and r["laal"] is not None), key=lambda r: r["laal"]):
            out.append(f"| {r['laal']:.2f} s | {r['asr_bleu']:.2f} | {r['kind']} | {tier(r['window'])} | {fmt(r['start'], 's')} |")
        out.append("")
        # findings
        if wins + losses:
            findings.append(f"{NAME[d]}：相同档位下 {wins}/{wins + losses} 个点高于 SimulS2ST-Omni。")
        if best:
            g1, g2 = best["asr_bleu"] - bleu_live, best["asr_bleu"] - OMNI[d][5]
            findings.append(f"{NAME[d]}：LAAL ≤ {laal_live:.2f} s 时我们最好 {best['asr_bleu']:.2f}（{best['kind']} {tier(best['window'])}{seeds_note(best)}），"
                            f"比 LiveInterpret 2.0 {g1:+.2f}，比 SimulS2ST-Omni m5 {g2:+.2f}"
                            + ("——已超过两者。" if g1 > 0 and g2 > 0 else "——尚未全部超过。"))
        gaps = [r["text_bleu"] - r["asr_bleu"] for r in latest.values() if r["dir"] == d]
        if gaps:
            findings.append(f"{NAME[d]}：思考模型文字 BLEU 与语音 ASR-BLEU 平均相差 {sum(gaps) / len(gaps):+.2f}，"
                            + ("差距主要在思考模型（翻译）一侧。" if sum(gaps) / len(gaps) < 2 else "说话模型仍有可提升空间。"))
    talk = [r for r in rows if r["kind"].startswith("talker") and r["n"] >= 300]
    out += ["## 4. 说话模型单测（给定参考译文，衡量说完整度，非翻译质量）", "",
            "| 检查点 | 交文字方式 | 读窗 | 中→英 ASR-BLEU | 英→中 ASR-BLEU | 拖长段（中→英/英→中） |", "|---|---|---|---|---|---|"]
    by = {}
    for r in talk:
        by.setdefault((r["family"], r["step"], r["kind"], r["window"]), {})[r["dir"]] = r
    for (fam, step, kind, w), v in sorted(by.items()):
        z, e = v.get("zh2en"), v.get("en2zh")
        out.append(f"| {fam}@{step} | {kind.split('-')[1]} | {w:g}s | {fmt(z and z['asr_bleu'])} | {fmt(e and e['asr_bleu'])} | "
                   f"{fmt(z and z['stretched'], '', 3)} / {fmt(e and e['stretched'], '', 3)} |")
    # the talker samples (temperature 0.7 unless the run says -T...): the same blocks spoken with other seeds
    seeds = {}
    for r in rows:
        if r["kind"].startswith("talker") or r["n"] < 300 or r["laal"] is None:
            continue
        base = re.sub(r"-seed\d+$", "", r["kind"])
        seeds.setdefault((base, r["family"], r["step"], r["window"], r["dir"]), []).append(r)
    multi = {k: v for k, v in seeds.items() if len(v) > 1}
    if multi:
        out += ["", "## 6. 说话模型采样的随机性（同一份思考模型文字，换说话模型的随机种子重说）", "",
                "| 定稿 | 档位 | 检查点 | 方向 | 种子数 | ASR-BLEU 均值 ± 标准差 | 各次 | LAAL 均值 |", "|---|---|---|---|---|---|---|---|"]
        for (base, fam, step, w, d), v in sorted(multi.items()):
            b = [x["asr_bleu"] for x in v]
            sd = (sum((x - sum(b) / len(b)) ** 2 for x in b) / (len(b) - 1)) ** 0.5
            out.append(f"| {base} | {tier(w)} | {fam}@{step} | {d} | {len(v)} | {sum(b) / len(b):.2f} ± {sd:.2f} | "
                       f"{' / '.join(f'{x:.2f}' for x in b)} | {sum(x['laal'] for x in v) / len(v):.2f} s |")
    out += ["", "## 5. 自动分析要点", ""] + [f"- {f}" for f in findings] + [""]
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(ROOT / "si_report"))
    args = ap.parse_args()
    rows = collect()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "rows.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    md = report(rows)
    (out / "report.md").write_text(md, encoding="utf-8")
    (out / "index.html").write_text(page(md), encoding="utf-8")
    print(f"{len(rows)} runs -> {out}/report.md, index.html", flush=True)


def page(md: str) -> str:
    """The report as a page: the tables wide-scrolling, both themes."""
    import datetime
    import markdown

    body = markdown.markdown(md, extensions=["tables"])
    body = re.sub(r"<table>", '<div class="tw"><table>', body).replace("</table>", "</table></div>")
    stamp = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=2))).strftime("%Y-%m-%d %H:%M")
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>RealSI 同传对比</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@400;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root{{--bg:#f4f5f2;--card:#fff;--ink:#1c2320;--muted:#5c6862;--line:#dce2de;--acc:#1f7a5c;--acc-soft:#d9efe5;--bad:#b3392f}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{--bg:#111614;--card:#192019;--ink:#e5ebe8;--muted:#98a6a0;--line:#2a3430;--acc:#4fbf95;--acc-soft:#1c3a2f;--bad:#f08a7e}}}}
:root[data-theme="dark"]{{--bg:#111614;--card:#192019;--ink:#e5ebe8;--muted:#98a6a0;--line:#2a3430;--acc:#4fbf95;--acc-soft:#1c3a2f;--bad:#f08a7e}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.65 "Noto Sans SC",system-ui,sans-serif}}
main{{max-width:1180px;margin:0 auto;padding:28px 16px 64px}}
h1{{font-size:26px;margin:0 0 6px;text-wrap:balance}}h2{{font-size:20px;margin:34px 0 8px;padding-top:14px;border-top:1px solid var(--line)}}
h3{{font-size:15.5px;margin:22px 0 8px;color:var(--acc)}}p,li{{max-width:80ch;color:var(--muted)}}li{{margin:4px 0;color:var(--ink)}}
.stamp{{font-family:"IBM Plex Mono",monospace;font-size:12.5px;color:var(--muted)}}
.tw{{overflow-x:auto;background:var(--card);border:1px solid var(--line);border-radius:10px;margin:8px 0}}
table{{border-collapse:collapse;width:100%;font-size:13.5px;font-variant-numeric:tabular-nums}}
th,td{{padding:6px 10px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}}th{{color:var(--muted);font-weight:500;font-size:12.5px}}
td strong{{color:var(--acc);font-weight:600}}tr:last-child td{{border-bottom:none}}
</style></head><body><main><div class="stamp">更新于 {stamp}（布拉格时间）</div>{body}</main></body></html>"""


if __name__ == "__main__":
    main()
