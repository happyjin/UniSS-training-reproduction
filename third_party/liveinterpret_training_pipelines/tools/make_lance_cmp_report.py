#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成 mentor 汇报用单文件 HTML：Lance 数据通路优化的环节与占比（3 页密排版）。

这是整份汇报里「数据通路优化」这一节：结论占比、改了哪些环节、全量数据与正确性。

所有数字的唯一真源是 docs/LANCE_SEVEN_WAY_200STEP_CMP.md；本脚本只做换算
（占比 = (base − lance) / base），不引入任何未实测的数字。

产物是自包含 HTML（内联 CSS / SVG / JS，无外部依赖），内网可直接打开：
  · ← → / 空格 翻页，F 全屏，P 切换连续滚动
  · 浏览器「打印 → 存为 PDF」即可得到横版讲稿

用法：
    python tools/make_lance_cmp_report.py [-o 输出路径.html]
"""

from __future__ import annotations

import argparse
import html
import re
from pathlib import Path

# ---------------------------------------------------------------- 实测数据
# 口径见 docs/LANCE_SEVEN_WAY_200STEP_CMP.md「性能对照（稳态）」
COLUMNS = [
    # key, 拓扑, steady_base, steady_lance, wall_base, wall_lance, rem_base, rem_lance
    ("B", "单机 DeepSpeed DP=8", 3.7, 2.89, "15m57s", "10m46s",
     "4d 3h 20m 32s", "2d 19h 4m 29s"),
    ("C", "单机 Megatron DP=8", 5.2, 4.52, "18m6s", "15m26s",
     "4d 16h 38m 47s", "4d 0h 7m 7s"),
    ("D", "Ray+DS 双节点 DP=16", 1.8, 1.54, "9m11s", "6m4s",
     "2d 9h 12m 32s", "1d 13h 43m 30s"),
    ("E", "Megatron 双节点 DP=16", 3.0, 2.66, "11m21s", "9m2s",
     "2d 22h 37m 1s", "2d 8h 14m 21s"),
    ("F", "DS 直连双节点 DP=16", 2.69, 2.06, "8m59s", "7m49s",
     "2d 7h 53m 17s", "2d 0h 40m 28s"),
]

E_BASE_NOTE = "E base 稳态原文为区间 2.9–3.1，换算取中值 3.0"

DLOSS = {"B": ("6.3625", "6.3696", "+0.007"),
         "C": ("6.3719", "6.3742", "+0.002"),
         "D": ("6.3605", "6.3619", "+0.001"),
         "E": ("6.3765", "6.3704", "−0.006"),
         "F": ("6.3605", "6.3645", "+0.004")}

LOSS_STEPS = [1, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
LOSS_B_BASE = [16.9301, 17.0098, 16.2513, 13.9580, 11.0488, 9.2522, 8.4513,
               7.4423, 6.8584, 6.5506, 6.3625]
LOSS_B_LANCE = [17.0005, 16.9984, 16.2111, 13.9397, 11.0142, 9.2422, 8.4511,
                7.4517, 6.8501, 6.5436, 6.3696]


def parse_dur(text: str) -> int:
    total, unit = 0, {"d": 86400, "h": 3600, "m": 60, "s": 1}
    for value, suffix in re.findall(r"(\d+)\s*([dhms])", text):
        total += int(value) * unit[suffix]
    return total


def fmt_hm(seconds: int) -> str:
    d, rem = divmod(seconds, 86400)
    h, rem = divmod(rem, 3600)
    m = rem // 60
    parts = []
    if d:
        parts.append(f"{d}d")
    if h:
        parts.append(f"{h}h")
    if m:
        parts.append(f"{m}m")
    return " ".join(parts) or "0m"


class Col:
    def __init__(self, key, topo, sb, sl, wb, wl, rb, rl):
        self.key, self.topo = key, topo
        self.steady_base, self.steady_lance = sb, sl
        self.wall_base, self.wall_lance = wb, wl
        self.wall_base_s, self.wall_lance_s = parse_dur(wb), parse_dur(wl)
        self.rem_base, self.rem_lance = rb, rl
        self.rem_base_s, self.rem_lance_s = parse_dur(rb), parse_dur(rl)

    steady_saved = property(lambda s: round(s.steady_base - s.steady_lance, 3))
    steady_share = property(lambda s: s.steady_saved / s.steady_base * 100)
    wall_share = property(
        lambda s: (s.wall_base_s - s.wall_lance_s) / s.wall_base_s * 100)
    rem_share = property(
        lambda s: (s.rem_base_s - s.rem_lance_s) / s.rem_base_s * 100)
    rem_saved_s = property(lambda s: s.rem_base_s - s.rem_lance_s)


COLS = [Col(*c) for c in COLUMNS]
ST = [c.steady_share for c in COLS]
WA = [c.wall_share for c in COLS]
RM = [c.rem_share for c in COLS]
MEAN_ST = sum(ST) / len(ST)
MEAN_FULL = (sum(WA) + sum(RM)) / (len(WA) + len(RM))
BEST_HOURS = max(c.rem_saved_s for c in COLS) / 3600


# ---------------------------------------------------------------- SVG 图表

def _txt(x, y, s, *, size=12, fill="#12203a", anchor="middle", weight=400):
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" '
            f'text-anchor="{anchor}" font-weight="{weight}">{html.escape(str(s))}</text>')


def svg_stacked(W=560, H=286) -> str:
    """稳态：堆叠柱（Lance 后 + 节省），总高 = base 每步耗时。"""
    pad_l, pad_r, pad_t, pad_b = 38, 8, 26, 40
    pw, ph = W - pad_l - pad_r, H - pad_t - pad_b
    ymax, n = 5.5, len(COLS)
    slot, bw = pw / n, pw / n * 0.46

    out = [f'<svg viewBox="0 0 {W} {H}" class="chart">']
    for i in range(6):
        v = i * 1.0
        y = pad_t + ph - v / ymax * ph
        out.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{W-pad_r}" y2="{y:.1f}" '
                   f'stroke="#e9eef5"/>')
        out.append(_txt(pad_l - 7, y + 4, f"{v:g}", size=10, fill="#8595ab", anchor="end"))
    out.append(_txt(pad_l - 7, pad_t - 12, "s/it", size=10, fill="#8595ab", anchor="end"))

    for i, c in enumerate(COLS):
        cx = pad_l + slot * (i + 0.5)
        x = cx - bw / 2
        hl, hs = c.steady_lance / ymax * ph, c.steady_saved / ymax * ph
        yl = pad_t + ph - hl
        ys = yl - hs
        out.append(f'<rect x="{x:.1f}" y="{yl:.1f}" width="{bw:.1f}" height="{hl:.1f}" '
                   f'fill="#c8d4e8" rx="2"/>')
        out.append(f'<rect x="{x:.1f}" y="{ys:.1f}" width="{bw:.1f}" height="{hs:.1f}" '
                   f'fill="#2563eb" rx="2"/>')
        out.append(_txt(cx, yl + hl / 2 + 4, f"{c.steady_lance:g}", size=11.5, weight=600))
        out.append(_txt(cx, ys + hs / 2 + 4, f"+{c.steady_saved:g}", size=10.5,
                        fill="#fff", weight=600))
        out.append(_txt(cx, ys - 7, f"{c.steady_share:.1f}%", size=12,
                        fill="#2563eb", weight=700))
        out.append(_txt(cx, H - pad_b + 17, c.key, size=12, weight=700))
        out.append(_txt(cx, H - pad_b + 31, f"base {c.steady_base:g}", size=9.5,
                        fill="#8595ab"))
    out.append(f'<line x1="{pad_l}" y1="{pad_t+ph}" x2="{W-pad_r}" y2="{pad_t+ph}" '
               f'stroke="#c3ccd9" stroke-width="1.5"/></svg>')

    legend = ('<span class="lg"><i style="background:#2563eb"></i>数据通路暴露部分 = 节省</span>'
              '<span class="lg"><i style="background:#c8d4e8"></i>Lance 后仍需的每步耗时</span>')
    return f'<div class="legend">{legend}</div>' + "".join(out)


def svg_grouped(W=560, H=286) -> str:
    """三口径分组柱（百分比）。"""
    pad_l, pad_r, pad_t, pad_b = 38, 8, 26, 40
    pw, ph = W - pad_l - pad_r, H - pad_t - pad_b
    ymax = 40.0
    series = [("稳态 s/it", ST, "#c8d4e8"),
              ("200-step 墙钟", WA, "#2563eb"),
              ("remaining_time", RM, "#0e9f6e")]
    n = len(COLS)
    slot, bw = pw / n, pw / n * 0.21

    out = [f'<svg viewBox="0 0 {W} {H}" class="chart">']
    for i in range(5):
        v = i * 10.0
        y = pad_t + ph - v / ymax * ph
        out.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{W-pad_r}" y2="{y:.1f}" '
                   f'stroke="#e9eef5"/>')
        out.append(_txt(pad_l - 7, y + 4, f"{v:g}%", size=10, fill="#8595ab", anchor="end"))

    for i, c in enumerate(COLS):
        cx = pad_l + slot * (i + 0.5)
        for j, (_, vals, color) in enumerate(series):
            v = vals[i]
            h = v / ymax * ph
            x = cx + (j - 1) * (bw + 3) - bw / 2
            y = pad_t + ph - h
            out.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" '
                       f'height="{h:.1f}" fill="{color}" rx="2"/>')
            out.append(_txt(x + bw / 2, y - 5, f"{v:.1f}", size=9.5, weight=600))
        out.append(_txt(cx, H - pad_b + 17, c.key, size=12, weight=700))
    out.append(f'<line x1="{pad_l}" y1="{pad_t+ph}" x2="{W-pad_r}" y2="{pad_t+ph}" '
               f'stroke="#c3ccd9" stroke-width="1.5"/></svg>')

    legend = "".join(f'<span class="lg"><i style="background:{c}"></i>{html.escape(n_)}</span>'
                     for n_, _, c in series)
    return f'<div class="legend">{legend}</div>' + "".join(out)


def svg_loss(W=560, H=270) -> str:
    """B 列 loss 逐 10 步双线：base 画粗浅灰在下，Lance 画细蓝虚线在上，重合可见。"""
    pad_l, pad_r, pad_t, pad_b = 38, 10, 20, 36
    pw, ph = W - pad_l - pad_r, H - pad_t - pad_b
    ymin, ymax = 6.0, 17.5
    n = len(LOSS_STEPS)

    def X(i):
        return pad_l + pw * i / (n - 1)

    def Y(v):
        return pad_t + ph - (v - ymin) / (ymax - ymin) * ph

    out = [f'<svg viewBox="0 0 {W} {H}" class="chart">']
    for v in [6, 8, 10, 12, 14, 16]:
        y = Y(v)
        out.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{W-pad_r}" y2="{y:.1f}" '
                   f'stroke="#e9eef5"/>')
        out.append(_txt(pad_l - 7, y + 4, f"{v:g}", size=10, fill="#8595ab", anchor="end"))
    out.append(_txt(pad_l - 7, pad_t - 6, "loss", size=10, fill="#8595ab", anchor="end"))

    pts_b = " ".join(f"{X(i):.1f},{Y(v):.1f}" for i, v in enumerate(LOSS_B_BASE))
    pts_l = " ".join(f"{X(i):.1f},{Y(v):.1f}" for i, v in enumerate(LOSS_B_LANCE))
    out.append(f'<polyline points="{pts_b}" fill="none" stroke="#9fb3cd" '
               f'stroke-width="7" stroke-linejoin="round" stroke-linecap="round"/>')
    out.append(f'<polyline points="{pts_l}" fill="none" stroke="#2563eb" '
               f'stroke-width="2.2" stroke-dasharray="6 5" stroke-linejoin="round"/>')
    for i, v in enumerate(LOSS_B_LANCE):
        out.append(f'<circle cx="{X(i):.1f}" cy="{Y(v):.1f}" r="2.8" fill="#fff" '
                   f'stroke="#2563eb" stroke-width="1.8"/>')
    for i, s in enumerate(LOSS_STEPS):
        out.append(_txt(X(i), H - pad_b + 16, s, size=9.5, fill="#8595ab"))
    out.append(_txt(pad_l + pw / 2, H - pad_b + 30, "step", size=9.5, fill="#8595ab"))
    out.append(f'<line x1="{pad_l}" y1="{pad_t+ph}" x2="{W-pad_r}" y2="{pad_t+ph}" '
               f'stroke="#c3ccd9" stroke-width="1.5"/></svg>')

    legend = ('<span class="lg"><i style="background:#9fb3cd"></i>B base（WDS 在线）</span>'
              '<span class="lg"><i class="dash"></i>B Lance（离线 mel+音色）</span>'
              '<span class="lg" style="color:#0e9f6e;font-weight:700">两条线全程重合</span>')
    return f'<div class="legend">{legend}</div>' + "".join(out)


# ---------------------------------------------------------------- 样式 / 脚本

CSS = """
*{box-sizing:border-box;margin:0;padding:0}
:root{--ink:#12203a;--accent:#2563eb;--good:#0e9f6e;--warn:#b45409;
  --muted:#64748b;--light:#f2f5f9;--line:#dde3ed}
html,body{height:100%}
body{background:#0d1526;color:var(--ink);
  font-family:-apple-system,BlinkMacSystemFont,"PingFang SC","Hiragino Sans GB",
    "Microsoft YaHei",system-ui,sans-serif;-webkit-font-smoothing:antialiased;
  overflow:hidden}
#stage{position:fixed;inset:0;display:grid;place-items:center}
.slide{position:absolute;width:1280px;height:720px;background:#fff;
  padding:34px 42px 26px;display:none;flex-direction:column;border-radius:6px;
  box-shadow:0 24px 70px rgba(0,0,0,.45);transform-origin:center center}
.slide.on{display:flex}

.hd{display:flex;align-items:baseline;gap:14px;margin-bottom:4px}
.kicker{font-size:11.5px;font-weight:700;color:var(--accent);letter-spacing:.09em;
  text-transform:uppercase;background:#eaf1ff;padding:3px 9px;border-radius:5px}
h2{font-size:26px;font-weight:700;letter-spacing:-.01em}
.rule{width:88px;height:3px;background:var(--accent);margin:9px 0 12px}
.lead{font-size:13px;line-height:1.6;margin-bottom:12px;max-width:1180px}
.foot{margin-top:auto;font-size:10.5px;color:var(--muted);line-height:1.55;
  border-top:1px solid var(--line);padding-top:8px}
b{font-weight:700}

.cards{display:grid;gap:12px}
.c2{grid-template-columns:repeat(2,1fr)}
.c3{grid-template-columns:repeat(3,1fr)}
.c4{grid-template-columns:repeat(4,1fr)}
.c5{grid-template-columns:repeat(5,1fr)}
.card{background:#fff;border:1px solid var(--line);border-radius:10px;padding:14px 16px}
.card.soft{background:var(--light);border-color:transparent}
.card.blue{background:#eff4ff;border-color:#d3e0fb}
.card.green{background:#ecfaf3;border-color:#b8e6d2}
.card.amber{background:#fff7e8;border-color:#f2ddb4}
.card.red{background:#feefef;border-color:#f0c0c0}
.card h4{font-size:14px;font-weight:700;margin-bottom:6px;line-height:1.4}
.card p{font-size:12px;line-height:1.6}
.card p+p{margin-top:6px}
.tag{font-size:10.5px;font-weight:700;letter-spacing:.05em;margin-bottom:5px}
.t-accent{color:var(--accent)}.t-good{color:var(--good)}
.t-warn{color:var(--warn)}.t-muted{color:var(--muted)}

.stat{background:var(--light);border-radius:10px;padding:12px 15px}
.stat b{font-size:10.5px;color:var(--muted);font-weight:700;display:block;
  letter-spacing:.04em;margin-bottom:5px}
.stat .v{font-size:25px;font-weight:700;letter-spacing:-.02em;line-height:1.1}
.stat .n{font-size:10.5px;color:var(--ink);margin-top:5px;line-height:1.5}

table{width:100%;border-collapse:collapse;font-size:11.5px}
thead th{background:var(--ink);color:#fff;font-weight:700;padding:6px 8px;
  text-align:left;font-size:11px;white-space:nowrap}
thead th:not(:first-child){text-align:center}
tbody td{padding:5.5px 8px;border-bottom:1px solid #e8edf4;text-align:center;
  line-height:1.45}
tbody td:first-child{text-align:left}
tbody tr:nth-child(even){background:var(--light)}
table.tight{font-size:11px}
table.tight thead th{padding:5px 6px;font-size:10.5px}
table.tight tbody td{padding:4.5px 6px}
td.nw{white-space:nowrap}
td.a{color:var(--accent);font-weight:700}
td.g{color:var(--good);font-weight:700}
td.w{color:var(--warn);font-weight:700}
td.m{color:var(--muted)}
td.n{font-weight:700}
.tblcap{font-size:11px;font-weight:700;color:var(--muted);margin-bottom:6px;
  letter-spacing:.03em}

.chart{width:100%;height:auto;display:block}
.legend{display:flex;flex-wrap:wrap;gap:14px;margin-bottom:2px;font-size:10.5px;
  color:var(--muted)}
.legend .lg{display:flex;align-items:center;gap:6px}
.legend i{width:10px;height:10px;border-radius:3px;display:inline-block}
.legend i.dash{height:0;border-top:2.5px dashed #2563eb;width:14px;border-radius:0}

.flow{display:grid;grid-template-columns:repeat(5,1fr);gap:10px}
.step{background:#fff;border:1px solid var(--line);border-radius:10px;
  padding:0 12px 11px;overflow:hidden}
.step .bar{height:4px;margin:0 -12px 10px}
.step .no{font-size:9.5px;font-weight:700;letter-spacing:.08em;margin-bottom:4px}
.step h5{font-size:13.5px;font-weight:700;margin-bottom:5px}
.step p{font-size:10.8px;line-height:1.55;color:var(--ink)}
.step .where{font-size:10px;font-weight:700;color:var(--muted);margin-top:7px}

.kv{display:flex;flex-direction:column;gap:6px}
.kv div{font-size:12px;line-height:1.55}
.banner{background:var(--ink);color:#fff;border-radius:10px;padding:13px 18px;
  font-size:13.5px;font-weight:700;line-height:1.55}
.banner.warn{background:#fff7e8;color:var(--warn);border:1px solid #f2ddb4}
.banner.warn span{color:var(--ink);font-weight:400}
.split{display:grid;gap:14px}
.grow{flex:1;min-height:0}
.row{display:grid;gap:14px}

/* HUD / 打印 */
#hud{position:fixed;right:16px;bottom:12px;color:#8ea3c4;font-size:11.5px;
  font-variant-numeric:tabular-nums;z-index:9;user-select:none}
#hud kbd{background:rgba(255,255,255,.12);border-radius:4px;padding:1px 5px;
  font-family:inherit;font-size:10.5px}
#bar{position:fixed;left:0;top:0;height:3px;background:var(--accent);z-index:9;
  transition:width .18s ease}
body.flowmode{overflow:auto}
body.flowmode #stage{position:static;display:block;padding:24px 0}
body.flowmode .slide{position:relative;display:flex;margin:0 auto 24px;
  transform:none!important}
@media print{
  @page{size:1280px 720px;margin:0}
  body{background:#fff;overflow:visible}
  #hud,#bar{display:none}
  #stage{position:static;display:block}
  .slide{position:relative!important;display:flex!important;transform:none!important;
    box-shadow:none;border-radius:0;margin:0;page-break-after:always;break-after:page}
}
"""

JS = """
const slides=[...document.querySelectorAll('.slide')];let i=0;
function fit(){
  if(document.body.classList.contains('flowmode')){
    const s=Math.min(1,(window.innerWidth-40)/1280);
    slides.forEach(el=>{el.style.transform='scale('+s+')';
      el.style.marginBottom=(24-720*(1-s))+'px';});
    return;
  }
  const s=Math.min(window.innerWidth/1280,window.innerHeight/720)*0.955;
  slides.forEach(el=>{el.style.transform='scale('+s+')';el.style.marginBottom='';});
}
function show(n){
  i=Math.max(0,Math.min(slides.length-1,n));
  slides.forEach((el,k)=>el.classList.toggle('on',k===i));
  document.getElementById('num').textContent=(i+1)+' / '+slides.length;
  document.getElementById('bar').style.width=((i+1)/slides.length*100)+'%';
}
addEventListener('keydown',e=>{const k=e.key;
  if(k==='ArrowRight'||k==='ArrowDown'||k===' '||k==='PageDown'){e.preventDefault();show(i+1);}
  else if(k==='ArrowLeft'||k==='ArrowUp'||k==='PageUp'){e.preventDefault();show(i-1);}
  else if(k==='Home'){show(0);}else if(k==='End'){show(slides.length-1);}
  else if(k==='f'||k==='F'){document.fullscreenElement?document.exitFullscreen()
    :document.documentElement.requestFullscreen();}
  else if(k==='p'||k==='P'){document.body.classList.toggle('flowmode');
    if(document.body.classList.contains('flowmode')){slides.forEach(el=>el.classList.add('on'));}
    else{show(i);} fit();}
});
addEventListener('resize',fit);
document.getElementById('stage').addEventListener('click',e=>{
  if(document.body.classList.contains('flowmode'))return;
  show(e.clientX<window.innerWidth*0.28?i-1:i+1);
});
fit();show(0);
"""


# ---------------------------------------------------------------- 构件

def esc(s):
    return html.escape(str(s))


def slide(body):
    return f'<section class="slide">{body}</section>'


def header(title, kicker, lead=None):
    out = (f'<div class="hd"><span class="kicker">{esc(kicker)}</span>'
           f"<h2>{esc(title)}</h2></div><div class='rule'></div>")
    if lead:
        out += f'<p class="lead">{lead}</p>'
    return out


def foot(text):
    return f'<div class="foot">{text}</div>'


def table_html(header_row, rows, *, classes=None, caption=None, widths=None,
               tight=False):
    """widths: 各列百分比，用 colgroup 固定，避免中文列被挤到换行。"""
    th = "".join(f"<th>{esc(h)}</th>" for h in header_row)
    body = []
    for r, row in enumerate(rows):
        tds = "".join(f'<td class="{(classes or {}).get((r, c), "")}">{v}</td>'
                      for c, v in enumerate(row))
        body.append(f"<tr>{tds}</tr>")
    cap = f'<div class="tblcap">{esc(caption)}</div>' if caption else ""
    cg = ""
    if widths:
        total = sum(widths)
        cg = "<colgroup>" + "".join(
            f'<col style="width:{w/total*100:.2f}%">' for w in widths) + "</colgroup>"
    cls = ' class="tight"' if tight else ""
    return (cap + f"<table{cls}>{cg}<thead><tr>{th}</tr></thead>"
            f"<tbody>{''.join(body)}</tbody></table>")


def stat(label, value, note, color):
    return (f'<div class="stat"><b>{esc(label)}</b>'
            f'<div class="v" style="color:{color}">{esc(value)}</div>'
            f'<div class="n">{note}</div></div>')


# ---------------------------------------------------------------- 页 1

def p1_result():
    stats = "".join([
        stat("数据通路占原步时（稳态）", f"{min(ST):.1f}–{max(ST):.1f}%",
             f"190→200 步 s/it，五列均值 <b>{MEAN_ST:.1f}%</b>", "var(--accent)"),
        stat("数据通路占全程", f"{min(WA):.1f}–{max(RM):.1f}%",
             f"墙钟 / remaining_time，均值 <b>{MEAN_FULL:.1f}%</b>", "var(--good)"),
        stat("换算成训练时长", f"最多少 {BEST_HOURS:.0f} 小时",
             "B 列全程 4d3h → 2d19h", "var(--good)"),
        stat("数学等价性", "Δloss ≤ 0.007", "五列 loss@200 与各自 base 同族", "var(--ink)"),
    ])
    concl = [
        ("var(--accent)", "可观，但不是大头",
         f"稳态每步省 <b>{MEAN_ST:.1f}%</b>（均值），全程口径 <b>{MEAN_FULL:.1f}%</b>。"
         "不同拓扑差异大：GPU 越快（D 列 1.8 s/it），数据越容易成为瓶颈。"),
        ("var(--good)", "只有三个环节真被优化",
         "去掉 tar 字节读取、mel 与音色改列读。"
         "<b>codec 两边本来都已离线</b>，不能算收益。"),
        ("var(--warn)", "环节级比例还没实测",
         f"现有实验是端到端 A/B，{MEAN_ST:.1f}% 拆不开到每一环；需 3 次消融补测（约 40 min 卡时）。"),
    ]
    cc = "".join(
        f'<div class="card soft"><div class="tag" style="color:{c}">{esc(t)}</div>'
        f"<p>{b}</p></div>" for c, t, b in concl)
    return slide(
        header("数据通路优化：省了多少", "结论先行",
               "唯一变量 = 数据通路。批次 / SCENE / warmup / 前反向全部未改，"
               "所以两边的差值可以直接归给数据这一块。")
        + f'<div class="cards c4">{stats}</div>'
        + '<div class="cards c2" style="margin-top:13px">'
          f'<div class="card"><h4>稳态口径：柱子总高 = 原 base 每步耗时</h4>{svg_stacked()}</div>'
          f'<div class="card"><h4>三口径对照：全程比稳态更高，差的是冷启动</h4>{svg_grouped()}</div>'
          "</div>"
        + f'<div class="cards c3" style="margin-top:13px">{cc}</div>'
        + foot("数字唯一真源：<b>docs/LANCE_SEVEN_WAY_200STEP_CMP.md</b>（trainer 实跑日志）。"
               "占比 = (base − Lance) / base。" + esc(E_BASE_NOTE)
               + "；F base 稳态与累计均记作 2.69，故 F 稳态占比高于其墙钟占比。"))


# ---------------------------------------------------------------- 页 2

def p2_what():
    steps = [
        ("STEP 1", "读清单", "FSX manifest-train-*.jsonl<br>≈18.99M 行",
         "CPU · 建表期", "var(--muted)"),
        ("STEP 2", "取音频字节", "wds:// → os.pread<br>本地 NVMe tar byte-range",
         "CPU · worker", "var(--warn)"),
        ("STEP 3", "在线 log-mel", "Whisper feature_extractor<br>torch.stft / fbank",
         "CPU · worker", "var(--warn)"),
        ("STEP 4", "取 target codes", "manifest 内联 [T,16]<br>已离线，不重算",
         "CPU · worker", "var(--muted)"),
        ("STEP 5", "在线音色向量", "ECAPA x-vector<br>从源 wav 现算",
         "GPU · forward", "var(--warn)"),
    ]
    flow = "".join(
        f'<div class="step"><div class="bar" style="background:{c}"></div>'
        f'<div class="no" style="color:{c}">{esc(no)}</div><h5>{esc(t)}</h5>'
        f'<p>{d}</p><div class="where">{esc(w)}</div></div>'
        for no, t, d, w, c in steps)

    rows = [
        ["数据落地", "WDS tar shards + FSX manifest",
         "processed.lance v506（6.2 TiB）", "替换"],
        ["清单来源", "manifest-train-*.jsonl（18.99M）",
         "Lance filter → 指针 JSONL（19.18M）", "替换"],
        ["源音频字节", "wds:// → os.pread byte-range", "不再读（audios=[]）", "去掉"],
        ["源 log-mel", "在线 Whisper，CPU worker 现算", "读 source_input_features 列", "替换"],
        ["音色向量", "在线 ECAPA，GPU forward 现算", "读 source_speaker_embedding 列", "替换"],
        ["target codec", "manifest 内联 codes（已离线）", "读 codes 列（仍是离线）", "仅换 IO"],
        ["DataLoader prefetch", "num_workers≈8，prefetch_factor=8", "完全相同", "未改"],
        ["批次 / SCENE / 前反向", "GBS 512，thinker 冻结", "完全相同", "未改"],
    ]
    cmap = {"去掉": "g", "替换": "a", "仅换 IO": "w", "未改": "m"}
    classes = {(r, 3): cmap[row[3]] + " nw" for r, row in enumerate(rows)}

    pf = [
        ("（a）S3 → 节点 NVMe 批量预取", "var(--accent)", "提交阶段，训练前",
         "6.2 TiB 整表同步到每个白名单节点的 /nvme，训练期数据面完全不碰 S3。"
         "这是最大的一块，也是 G 列卡住的原因。"),
        ("（b）Lance 列批量点查", "var(--good)", "DataLoader 取 batch 时",
         "__getitems__ 里一次 take_by_stable_rowid 同时取回 mel + speaker + codes，"
         "避免每样本多次点查。"),
        ("（c）PyTorch DataLoader prefetch", "var(--muted)", "稳态训练中",
         "num_workers≈8，prefetch_factor=8。两条通路完全一样，"
         "<b>这次没有动它</b> —— 属于公平性保证。"),
    ]
    pfc = "".join(
        f'<div class="card soft"><div class="tag" style="color:{c}">{esc(t)}</div>'
        f'<p style="color:{c};font-weight:700">{esc(w)}</p><p>{b}</p></div>'
        for t, c, w, b in pf)

    return slide(
        header("到底改了哪些环节", "现状 vs Lance",
               "原通路里 manifest 只是清单，音频字节在 tokenize 阶段才按引用现读（lazy）。"
               "热路径上真正花时间的是 <b>STEP 2 / 3 / 5</b>（下方橙色），"
               "STEP 1 是建表期一次性开销，STEP 4 两边都已离线 —— 所以 Lance 能动的只有三步。")
        + f'<div class="flow">{flow}</div>'
        + '<div class="row" style="grid-template-columns:1.45fr 1fr;margin-top:12px">'
        + "<div>"
        + table_html(["环节", "原通路", "Lance 通路", "状态"], rows,
                     classes=classes, caption="环节对照（末两行 = 公平性保证）",
                     widths=[17, 30, 34, 9], tight=True)
        + '<div class="card red" style="margin-top:11px">'
          '<div class="tag t-warn">口径澄清 · 别说「Lance 省掉了 GPU 在线编 codec 的开销」</div>'
          "<p>cmp200 这套 unist 数据，原 manifest 本来就内联了 <b>target_codec_codes</b>，"
          "两条通路训练期都不跑 codec encode —— 这么讲会被一句话问穿。<br>"
          "正确说法：收益来自<b>源侧 mel + 音色 + IO 布局</b>三件事。</p></div>"
        + "</div><div>"
        + '<div class="tblcap">「预读取」其实是三件不同的事，别混着讲</div>'
        + f'<div class="cards" style="gap:9px">{pfc}</div>'
        + "</div></div>"
        + foot("代码位置：<b>custom/wds_audio.py</b>（pread）· "
               "<b>custom/template.py</b>（_encode / mel / _build_precomputed_codec_fields）· "
               "<b>custom/speaker_encoder.py</b>（ECAPA）· "
               "<b>custom/lance_lazy_batch_patch.py</b>（批量点查）。"))


# ---------------------------------------------------------------- 页 3

def p3_data():
    rows, classes = [], {}
    for r, c in enumerate(COLS):
        rows.append([f"{c.key} {c.topo}",
                     f"{c.steady_base:g} → {c.steady_lance:g}",
                     f"{c.steady_share:.1f}%",
                     f"{c.wall_base} → {c.wall_lance}",
                     f"{c.wall_share:.1f}%",
                     f"{fmt_hm(c.rem_base_s)} → {fmt_hm(c.rem_lance_s)}",
                     f"{c.rem_share:.1f}%",
                     f"−{fmt_hm(c.rem_saved_s)}",
                     DLOSS[c.key][2]])
        classes[(r, 0)] = "n"
        classes[(r, 2)] = "a"
        classes[(r, 4)] = "g"
        classes[(r, 6)] = "g"
        classes[(r, 7)] = "n"
        classes[(r, 8)] = "m"

    lrows = [[k, b, l, d] for k, (b, l, d) in DLOSS.items()]
    lclasses = {(r, 3): "g" for r in range(len(lrows))}

    return slide(
        header("五列全量数据与正确性守门", "全量数据",
               "三个口径同时给：<b>稳态 s/it</b> 看关键路径上的每步开销，"
               "<b>200-step 墙钟</b>含冷启动，<b>remaining_time</b> 是 trainer 按累计均值速率"
               "外推的全程（max_steps≈74910–74912）。三者互相印证。")
        + table_html(["列 / 拓扑", "稳态 s/it", "稳态省", "200-step 墙钟", "墙钟省",
                      "remaining@200", "全程省", "少花", "loss@200 Δ"],
                     rows, classes=classes)
        + '<div class="row" style="grid-template-columns:1.15fr 1fr;margin-top:13px">'
        + f'<div class="card"><h4>B 列 loss 逐 10 步：换通路没有改变数学轨迹</h4>{svg_loss()}</div>'
        + "<div>"
        + table_html(["列", "loss@200 base", "Lance", "Δ"], lrows,
                     classes=lclasses, caption="五列 loss@200 全部同族")
        + '<div class="cards c2" style="margin-top:11px">'
          '<div class="card green"><div class="tag t-good">正确性</div>'
          '<h4 style="font-size:20px;color:var(--good)">|Δ| ≤ 0.007</h4>'
          "<p>远小于原文七列两两差 ≤0.032 的容差。step 1 的 Δ 偏大（B +0.070）"
          "是首步随机性，第 10 步即收敛到 −0.007。</p></div>"
          '<div class="card soft"><div class="tag t-accent">怎么读这三个口径</div>'
          "<p>全程口径普遍高于稳态，差出来的是<b>冷启动</b>（建表、首批预热）。"
          "F 列反过来（稳态 23.4% &gt; 墙钟 13.0%），因为其 base 稳态与累计均值都记作 2.69。</p>"
          "</div></div>"
          "</div></div>"
        + foot("loss@200 Δ = Lance − base。全程省与墙钟省高度一致（差 ≤0.2 个百分点），"
               "说明 remaining_time 外推是可信的。"
               "B Lance 的 remaining@200 为回推值：ktrain 停任务后持久化日志被清，"
               "用同步的 train_speed=3.232 复原。"))


# ---------------------------------------------------------------- 入口

SLIDES = [p1_result, p2_what, p3_data]


def build(path: Path) -> Path:
    body = "".join(fn() for fn in SLIDES)
    doc = ("<!DOCTYPE html>\n<html lang='zh-CN'><head><meta charset='utf-8'>"
           "<meta name='viewport' content='width=device-width,initial-scale=1'>"
           "<title>Lance 数据通路优化 · 环节与占比</title>"
           f"<style>{CSS}</style></head><body>"
           f"<div id='bar'></div><div id='stage'>{body}</div>"
           "<div id='hud'><span id='num'></span>　"
           "<kbd>←</kbd><kbd>→</kbd> 翻页　<kbd>F</kbd> 全屏　<kbd>P</kbd> 连续滚动</div>"
           f"<script>{JS}</script></body></html>\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(doc, encoding="utf-8")
    return path


def main():
    default = (Path(__file__).resolve().parents[1] / "docs" /
               "Lance数据通路优化_环节与占比.html")
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--output", type=Path, default=default)
    args = ap.parse_args()

    out = build(args.output)
    print(f"wrote {out}  ({out.stat().st_size/1024:.0f} KiB, {len(SLIDES)} slides)")

    print("\n占比自检（占比 = (base − Lance) / base）")
    print(f"{'col':4}{'steady':>10}{'wall':>10}{'remaining':>11}{'saved':>12}")
    for c in COLS:
        print(f"{c.key:4}{c.steady_share:9.1f}%{c.wall_share:9.1f}%"
              f"{c.rem_share:10.1f}%{fmt_hm(c.rem_saved_s):>12}")
    print(f"\nsteady {min(ST):.1f}–{max(ST):.1f}%  mean {MEAN_ST:.1f}%")
    print(f"wall   {min(WA):.1f}–{max(WA):.1f}%  mean {sum(WA)/len(WA):.1f}%")
    print(f"remain {min(RM):.1f}–{max(RM):.1f}%  mean {sum(RM)/len(RM):.1f}%")
    print(f"full-run mean (wall+remain) {MEAN_FULL:.1f}%  best saved {BEST_HOURS:.1f}h")


if __name__ == "__main__":
    main()
