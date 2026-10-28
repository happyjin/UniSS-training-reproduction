"""Listening page: a fixed-rule sample of test utterances (4-9 s with a comma, ordered by crc32 of the id, first N)
rendered by every key system, the real recording, the per-word wait pattern of each streaming system, and the
test-500 summary table (results_v2.json + audio_metrics.json where present).

  python make_listen_page.py --exp runs/final --out runs/final/listen --html index_v1.html [--n 6]
Wavs are copied once into <out>/audio/<system>/; the html file is never overwritten (pass a new name).
"""
import argparse
import html
import json
import shutil
import sys
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "vits"))
sys.path.insert(0, str(ROOT / "scripts"))
from eval_metrics import load_runs  # noqa: E402

# key, label, note, render dir (None = real recording), results key, stream?
SYSTEMS = [
    ("gt", "真实录音", "LJSpeech 原始录音", None, None, False),
    ("e0", "E0 离线 VITS", "整句可见，本仓库复现的官方 VITS", "eval_render_test/e0_offline", "fixed:test/e0_offline", False),
    ("e1_k0", "E1 · k=0", "主 baseline，零等待", "eval_render_test/e1_fixed0", "fixed:test/e1/k0", True),
    ("e1_k1", "E1 · k=1", "主 baseline，等 1 个词", "eval_render_test/e1_fixed1", "fixed:test/e1/k1", True),
    ("e2_k1", "E2 · k=1", "+ Qwen3 前缀隐状态", "eval_render_test/e2_fixed1", "fixed:test/e2/k1", True),
    ("s6f_k0", "S6-future · k=0", "+ LLM 想象未来 + r，零等待", "eval_render_test/s6_future_fixed0", "fixed:test/s6_future/k0", True),
    ("s6f_k1", "S6-future · k=1", "+ LLM 想象未来 + r，等 1 个词", "eval_render_test/s6_future_fixed1", "fixed:test/s6_future/k1", True),
    ("ent", "S6-future + 熵门控 τ=4.25", "策略 baseline（E5）", "eval_render_test/s6_future_entropy_4.25", "policy:entropy@0.5", True),
    ("t3", "S6-future + T3 ε=1.0", "风险头 + 轨迹级 conformal 证书", "eval_render_test/t3_s6_future_1.0", "policy:t3@0.5", True),
    ("t3lite", "S6-future + T3-lite ε=1.5", "不用声学探针的轻量 T3", "eval_render_lite_test/t3_s6_future_1.5", "policy:t3_lite@0.25", True),
]


def pick(rows, n):
  cand = [r for r in rows.values() if r["split"] == "test" and 4.0 <= r["duration_s"] <= 9.0 and "," in r["text"]]
  return sorted(cand, key=lambda r: zlib.crc32(r["utt_id"].encode()))[:n]


def fmt(x, nd=4, pct=False):
  if x is None:
    return '<span class="na">评测中</span>'
  return f"{100 * x:.1f}%" if pct else f"{x:.{nd}f}"


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--exp", required=True)
  ap.add_argument("--out", required=True)
  ap.add_argument("--html", required=True)
  ap.add_argument("--n", type=int, default=6)
  args = ap.parse_args()
  exp, out = Path(args.exp).resolve(), Path(args.out)
  if (out / args.html).exists():
    sys.exit(f"refusing to overwrite {out / args.html}")
  rows = {r["utt_id"]: r for r in map(json.loads, open(ROOT / "data/manifest_v2.jsonl"))}
  res = json.load(open(exp / "results_v2.json"))
  utts = pick(rows, args.n)

  table, waits = [], {}
  for key, label, note, d, rkey, stream in SYSTEMS:
    m = {}
    if rkey:
      kind, k = rkey.split(":")
      m = res["fixed"][k] if kind == "fixed" else res["policies"][k]["test"]
    am = {}
    if d and (exp / d / "audio_metrics.json").exists():
      am = json.load(open(exp / d / "audio_metrics.json"))
    elif d is None and (ROOT / "runs/gt_audio_test/audio_metrics.json").exists():
      am = json.load(open(ROOT / "runs/gt_audio_test/audio_metrics.json"))
    table.append((key, label, note, m, am, stream))
    if stream:
      runs = load_runs(exp / d)
      waits[key] = {u["utt_id"]: [x["j"] - x["i"] for x in runs[u["utt_id"]]["units"]] for u in utts}
    for u in utts:
      src = Path(u["audio_path"]) if d is None else exp / d / "wavs" / f"{u['utt_id']}.wav"
      dst = out / "audio" / key / f"{u['utt_id']}.wav"
      if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, dst)

  trs = []
  for key, label, note, m, am, stream in table:
    look = "整句" if key == "e0" else ("—" if key == "gt" else fmt(m.get("look_mean"), 2))
    trs.append(
        f'<tr class="{"hl" if key in ("t3", "s6f_k1") else ""}"><th scope="row"><span class="sys">{html.escape(label)}</span>'
        f'<span class="note">{html.escape(note)}</span></th><td>{look}</td>'
        f'<td>{fmt(m.get("mae_word")) if m else "—"}</td><td>{fmt(m.get("mae_pause")) if m else "—"}</td>'
        f'<td>{fmt(m.get("pause_f1"), 3) if m else "—"}</td><td>{fmt(am.get("wer"), pct=True)}</td>'
        f'<td>{fmt(am.get("utmos"), 2)}</td></tr>')

  cards = []
  for u in utts:
    words = [u["clean"][a:b].strip() for a, b in u["units"]]
    players = []
    for key, label, note, m, am, stream in table:
      strip = ""
      if stream:
        w = waits[key][u["utt_id"]]
        cells = "".join(f'<i class="w{min(x, 3)}" title="{html.escape(words[n])}：等待 {x} 个词"></i>' for n, x in enumerate(w))
        strip = f'<div class="strip" aria-label="每个词提交前等待的词数">{cells}</div>'
      players.append(
          f'<div class="row r-{key}"><div class="lab"><span class="sys">{html.escape(label)}</span>{strip}</div>'
          f'<audio controls preload="none" src="audio/{key}/{u["utt_id"]}.wav"></audio></div>')
    cards.append(
        f'<section class="card"><header><span class="uid">{u["utt_id"]}</span><span class="dur">{u["duration_s"]:.1f} s · '
        f'{len(words)} 词</span></header><p class="text">{html.escape(u["text"])}</p>{"".join(players)}</section>')

  page = TEMPLATE.replace("%ROWS%", "".join(trs)).replace("%CARDS%", "".join(cards))
  (out / args.html).write_text(page)
  print(f"{len(utts)} utterances, {len(SYSTEMS)} systems -> {out / args.html}")


TEMPLATE = open(Path(__file__).with_name("listen_page_template.html")).read()

if __name__ == "__main__":
  main()
