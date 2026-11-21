"""Add one decoded version (arm) to an existing E10 esports listening page (build_demo_page.py output): same page,
player and audio format; the new arm's audio is built exactly as build_demo_page.py builds E10's (per-speaker
queue on the source clock, 16 kHz stereo mp3, L = source, R = translation, -26.5 dBFS active RMS). Pieces without
any letter / digit / CJK character (a lone "," from a split at a speaker tag) are not spoken.

    $PYE e10_multispk/add_demo_arm.py --page-dir e10_multispk/output/demo_game_v1/page \
        --raw e10_multispk/output/demo_game_v3/raw_zh2en,e10_multispk/output/demo_game_v3/raw_en2zh \
        --key V3 --label "LiveInterpret v3（按 B 训练）" --run e10v3_joint/checkpoint-6000 \
        --out e10_multispk/output/demo_game_v3/page
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

import build_demo_page as BD

WORD = re.compile(r"[0-9A-Za-z一-鿿]")


def build_arm(sid, d, ws, R, out_audio, key):
    sub, name = BD.SESS[d]
    src = BD.load16(BD.UNISS / "data/game_esports_si_v1" / d / "session_audio" / sub / name.format(m=sid))
    right = np.zeros(len(src) + 30 * BD.SR, dtype=np.float32)
    tgt, rep = [], defaultdict(int)
    for w in ws:
        r = R[w["id"]]
        rep["errors"] += bool(r.get("error"))
        rep["guard_cuts"] += int(r.get("n_guard") or 0)
        end = defaultdict(int)
        for g in r.get("segs", []):
            if not g.get("wav") or not WORD.search(g["text"] or ""):
                rep["skipped_pieces"] += bool(g.get("wav"))
                continue
            x = sf.read(g["wav"], dtype="float32")[0]
            k = g["spk"] if isinstance(g["spk"], int) else -1
            st = max(int(g["write_ms"]), end[k])
            end[k] = st + int(round(len(x) / 16))
            t = w["start_s"] + st / 1000
            a0 = int(t * BD.SR)
            if a0 + len(x) > len(right):
                right = np.concatenate([right, np.zeros(a0 + len(x) - len(right), dtype=np.float32)])
            right[a0: a0 + len(x)] += x
            tgt.append({"t": round(t, 2), "e": round(t + len(x) / BD.SR, 2), "spk": k, "text": g["text"].strip()})
            rep["pieces_spoken"] += 1
    tgt.sort(key=lambda x: x["t"])
    last = max([len(src)] + [int(x["e"] * BD.SR) for x in tgt])
    BD.mp3(BD.normalize(src), BD.normalize(right[:last]), out_audio / f"{sid}__{key}.mp3")
    rep["speakers"] = sorted({x["spk"] for x in tgt})
    return tgt, dict(rep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--page-dir", required=True)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--key", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--relabel", default="E10=LiveInterpret E10（上一版）")
    ap.add_argument("--out", required=True)
    ap.add_argument("--extra-audio", default=str(BD.UNISS / "demo_packages/game_multispk_e10_demo_offline_v1/audio"),
                    help="comma-separated dirs holding referenced mp3s missing from --page-dir (B's, published by copy)")
    a = ap.parse_args()
    src_dir, out = Path(a.page_dir), Path(a.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    html = (src_dir / "index.html").read_text(encoding="utf-8")
    i = html.index("const DATA = ") + len("const DATA = ")
    j = html.index("\n", i)
    D = json.loads(html[i:j].rstrip().rstrip(";"))
    R = {}
    for d in a.raw.split(","):
        for f in sorted(Path(d).glob("part_*.jsonl")):
            for line in open(f, encoding="utf-8"):
                r = json.loads(line)
                R[r["id"]] = r
    W = defaultdict(list)
    for p in BD.WIN.values():
        for line in open(BD.UNISS / p, encoding="utf-8"):
            w = json.loads(line)
            W[w["meeting"]].append(w)
    (out / "audio").mkdir(parents=True)
    for f in sorted((src_dir / "audio").glob("*.mp3")):
        shutil.copy2(f, out / "audio" / f.name)
    report = {}
    for s in D["samples"]:
        sid = s["id"]
        ws = sorted(W[sid], key=lambda w: w["start_s"])
        miss = [w["id"] for w in ws if w["id"] not in R]
        if miss:
            raise SystemExit(f"{sid}: {len(miss)} windows not decoded, e.g. {miss[:3]}")
        tgt, rep = build_arm(sid, s["dir"], ws, R, out / "audio", a.key)
        s["arms"] = {a.key: {"label": a.label, "run": a.run, "audio": f"audio/{sid}__{a.key}.mp3", "tgt": tgt},
                     **s["arms"]}
        report[sid] = rep
        print(sid, rep, flush=True)
    D["arm_order"] = [a.key] + [k for k in D["arm_order"] if k != a.key]
    D["arm_labels"] = {a.key: a.label, **D["arm_labels"]}
    for kv in filter(None, a.relabel.split(";")):
        k, v = kv.split("=", 1)
        D["arm_labels"][k] = v
        for s in D["samples"]:
            if k in s["arms"]:
                s["arms"][k]["label"] = v
    D["foot"] = (f"{a.label}：同一个 LiveInterpret S5 模型（Voxtral 编码器 + Qwen3.5-4B Thinker + 0.6B Talker，CosyVoice3 "
                 "声码器），按 baseline B 的数据、两阶段（O2 离线 2000 步 + 联合 6000 步）、学习率 1e-4→1e-5、"
                 "LoRA r32（只加在注意力层）和说话人流（每 5 帧一个说话人 token）重新训练 Thinker，Talker 冻结；"
                 "解码加了 B 的循环保护。LiveInterpret E10（上一版）是此前的版本。" + D.get("foot", ""))
    html = html[: html.index("const DATA = ")] + "const DATA = " + json.dumps(D, ensure_ascii=False) + ";" + html[j:]
    k0 = html.index('<p class="lede">')
    k1 = html.index("</p>", k0)
    lede = html[k0:k1].replace('<b>每段都可以在"版本"里切换 LiveInterpret（E10）和现有 baseline B</b>',
                               f'<b>每段都可以在"版本"里切换 {a.label}、上一版 E10 和现有 baseline B</b>')
    html = html[:k0] + lede + html[k1:]
    for s in D["samples"]:            # every referenced mp3 must be in the new page
        for arm in s["arms"].values():
            f = out / arm["audio"]
            if not f.exists():
                cand = [Path(d) / f.name for d in a.extra_audio.split(",") if (Path(d) / f.name).exists()]
                if not cand:
                    raise SystemExit(f"missing audio {arm['audio']}")
                shutil.copy2(cand[0], f)
    (out / "index.html").write_text(html, encoding="utf-8")
    (out / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(json.dumps({"page": str(out / "index.html"), "arms": D["arm_order"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
