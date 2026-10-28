"""Listening page for every handbook scheme: the fixed-rule test sample of make_listen_page.py rendered by the
references, the best system, the commitment policies, the acoustic replacements and the S1 planner; one table with
the test-500 metrics recomputed from each system's own output (commits for durations/lookahead, audio_metrics.json).

  python make_listen_page_all.py --out runs/all_schemes/listen --html index_v3.html
Wavs are copied once into <out>/audio/<system>/; the html file is never overwritten (pass a new name).
"""
import argparse
import html
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "vits"))
sys.path.insert(0, str(ROOT / "scripts"))
from eval_metrics import load_runs  # noqa: E402
from make_listen_page import pick  # noqa: E402
from make_report_all import Ctx  # noqa: E402

F, A = "runs/final", "runs/all_schemes"
# group, key, label, verdict, dir (None: the recording), streaming
SYSTEMS = [
    ("参照", "gt", "真实录音", "LJSpeech 原始录音", None, False),
    ("参照", "e0", "E0 离线 VITS", "整句可见，上界参照", f"{F}/eval_render_test/e0_offline", False),
    ("参照", "e1", "E1 · k=1", "主 baseline", f"{F}/eval_render_test/e1_fixed1", True),
    ("最佳组合", "best", "S6-future + T5 + S5 解码器", "所有指标不输 E1：等待少 43%，词与停顿误差显著更低，WER、UTMOS 持平",
     f"{A}/render_best/t5_0.05_s5dec", True),
    ("提交策略（声学用 E0 解码器）", "s6f", "S6-future · k=1", "时长与停顿显著优于 E1，音频持平", f"{F}/eval_render_test/s6_future_fixed1", True),
    ("提交策略（声学用 E0 解码器）", "t3", "+ T3 ε=1.0", "等待少约 1/3，但 WER 显著高 0.45 个百分点", f"{F}/eval_render_test/t3_s6_future_1.0", True),
    ("提交策略（声学用 E0 解码器）", "t5", "+ T5 λ=0.05", "策略最好：比 E1 词与停顿显著更好、等待少 43%、音频持平；比 T3 等得更少", f"{F}/eval_render_pol_test/t5_s6_future_0.05", True),
    ("提交策略（声学用 E0 解码器）", "t1", "+ T1 τ=0.05", "与 T3 大致持平，中延迟段较差", f"{F}/eval_render_pol_test/t1_s6_future_0.05", True),
    ("提交策略（声学用 E0 解码器）", "t2", "+ T2 τ=0.7", "比 T3 和熵门控都差", f"{F}/eval_render_pol_test/t2_s6_future_0.7", True),
    ("声学替换（同一组 S6-future k=1 时长）", "s5", "S5 WavLM 判别器", "声学最好：UTMOS +0.017（边缘），F0 误差显著更低", f"{A}/render_test/s5", True),
    ("声学替换（同一组 S6-future k=1 时长）", "s4", "S4 Qwen3-TTS token", "UTMOS −0.03，WER +0.6 个百分点（均显著）", f"{A}/render_test/s4", True),
    ("声学替换（同一组 S6-future k=1 时长）", "s2", "S2 CFM（DiT）", "UTMOS −0.54", f"{A}/render_test/s2", True),
    ("声学替换（同一组 S6-future k=1 时长）", "t4", "T4 因果 flow + 解码器", "UTMOS −1.0，WER 7.6%：无右上下文的解码器音质下降", f"{A}/render_test/t4", True),
    ("声学替换（同一组 S6-future k=1 时长）", "s3", "S3 VoxCPM2 AudioVAE", "失败：编解码器重建 UTMOS 4.31，但文本条件先验合成只有 1.5", f"{A}/render_test/s3", True),
    ("时长规划", "s1", "S1 扩散先验 · k=1", "词与停顿误差比 S6-future 显著更差，语速偏快", f"{A}/eval_render_test/s1_fixed1", True),
]


def fmt(x, nd=4, pct=False):
    if x is None:
        return "—"
    return f"{100 * x:.1f}%" if pct else f"{x:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--html", required=True)
    ap.add_argument("--n", type=int, default=6)
    args = ap.parse_args()
    out = Path(args.out)
    if (out / args.html).exists():
        sys.exit(f"refusing to overwrite {out / args.html}")
    ctx = Ctx(ROOT / F)
    utts = pick(ctx.rows, args.n)
    rows_html, waits = [], {}
    group = None
    for g, key, label, verdict, d, stream in SYSTEMS:
        m = ctx.m(ROOT / d) if d else {}
        am = json.load(open(ROOT / d / "audio_metrics.json")) if d else json.load(open(ROOT / "runs/gt_audio_test/audio_metrics.json"))
        if stream:
            runs = load_runs(ROOT / d)
            waits[key] = {u["utt_id"]: [x["j"] - x["i"] for x in runs[u["utt_id"]]["units"]] for u in utts}
        for u in utts:
            src = Path(u["audio_path"]) if d is None else ROOT / d / "wavs" / f"{u['utt_id']}.wav"
            dst = out / "audio" / key / f"{u['utt_id']}.wav"
            if not dst.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(src, dst)
        if g != group:
            rows_html.append(f'<tr class="grp"><th colspan="7" scope="colgroup">{html.escape(g)}</th></tr>')
            group = g
        look = "整句" if key == "e0" else "—" if key == "gt" else fmt(m.get("look_mean"), 2)
        rows_html.append(
            f'<tr class="{"hl" if key == "best" else ""}"><th scope="row"><span class="sys">{html.escape(label)}</span>'
            f'<span class="note">{html.escape(verdict)}</span></th><td>{look}</td><td>{fmt(m.get("mae_word")) if m else "—"}</td>'
            f'<td>{fmt(m.get("mae_pause")) if m else "—"}</td><td>{fmt(am.get("wer"), pct=True)}</td>'
            f'<td>{fmt(am.get("utmos"), 2)}</td><td>{fmt(am.get("f0_logmae"), 3)}</td></tr>')
    cards = []
    for u in utts:
        words = [u["clean"][a:b].strip() for a, b in u["units"]]
        players, group = [], None
        for g, key, label, verdict, d, stream in SYSTEMS:
            if g != group:
                players.append(f'<div class="ghead">{html.escape(g)}</div>')
                group = g
            strip = ""
            if stream:
                cells = "".join(f'<i class="w{min(x, 3)}" title="{html.escape(words[n])}：等待 {x} 个词"></i>'
                                for n, x in enumerate(waits[key][u["utt_id"]]))
                strip = f'<div class="strip" aria-label="每个词提交前等待的词数">{cells}</div>'
            players.append(f'<div class="row r-{key}"><div class="lab"><span class="sys">{html.escape(label)}</span>{strip}</div>'
                           f'<audio controls preload="none" src="audio/{key}/{u["utt_id"]}.wav"></audio></div>')
        cards.append(f'<section class="card"><header><span class="uid">{u["utt_id"]}</span><span class="dur">{u["duration_s"]:.1f} s · '
                     f'{len(words)} 词</span></header><p class="text">{html.escape(u["text"])}</p>{"".join(players)}</section>')
    page = open(Path(__file__).with_name("listen_page_all_template.html")).read()
    page = page.replace("%ROWS%", "".join(rows_html)).replace("%CARDS%", "".join(cards))
    (out / args.html).write_text(page)
    files = sorted(str(p.relative_to(out)) for p in (out / "audio").rglob("*.wav"))
    (out / f"{args.html}.files.json").write_text(json.dumps(files))
    print(f"{len(utts)} utterances, {len(SYSTEMS)} systems, {len(files)} wavs -> {out / args.html}")


if __name__ == "__main__":
    main()
