"""E10 esports listening demo: the same page, player and audio format as the B esports page
(artifact 王者荣耀解说同传试听): 16 kHz stereo mp3, left = the 5-min source session, right = the translation laid on the
source clock, both channels at the B page's level (active RMS about -26.5 dBFS); a version switch E10 / B.

E10 timeline (build_e10.py / E8 rule): inside a window every speaker's pieces are queued, a piece starts at
max(its write time, end of that speaker's previous piece); absolute time = window start + that.

    $PYE e10_multispk/build_demo_page.py --page-data <B page DATA json> --page-html <B page html> \
        --raw e10_multispk/output/demo_game_v1/raw_zh2en,e10_multispk/output/demo_game_v1/raw_en2zh \
        --ids gz01,gz02,gz03,ge01,ge02,ge04 --out e10_multispk/output/demo_game_v1/page
"""
from __future__ import annotations

import argparse
import json
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

UNISS = Path("/opt/dlami/nvme/jasonleeeli/projects/UniSS")
FFMPEG = "/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval/lib/python3.12/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2"
SR = 16000
TARGET_DB = -26.5
WIN = {"zh2en": "eval_outputs/mist_w3po_v1/data/windows_game_zh2en_c2g06_v1/windows_test_online_zh2en.jsonl",
       "en2zh": "eval_outputs/mist_w3po_v1/data/windows_game_en2zh_c2g06_v1/windows_test_online_en2zh.jsonl"}
SESS = {"zh2en": ("mix", "{m}.wav"), "en2zh": ("ihm_mix", "{m}.Mix-Headset.wav")}


def load16(path):
    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    x = x.mean(1)
    if sr != SR:
        import torch
        import torchaudio
        x = torchaudio.functional.resample(torch.from_numpy(x), sr, SR).numpy()
    return x


def normalize(x):
    """Scale so the active 100 ms frames sit at TARGET_DB RMS (active = above 10 % of the loud frames' RMS)."""
    n = len(x) // 1600 * 1600
    if n == 0:
        return x
    r = np.sqrt((x[:n].reshape(-1, 1600) ** 2).mean(1))
    if r.max() <= 1e-6:
        return x
    act = r[r > max(1e-4, 0.1 * np.percentile(r, 95))]
    g = 10 ** (TARGET_DB / 20) / np.sqrt((act ** 2).mean())
    return np.clip(x * g, -0.98, 0.98)


def mp3(left, right, out):
    n = max(len(left), len(right))
    st = np.zeros((n, 2), dtype=np.float32)
    st[: len(left), 0], st[: len(right), 1] = left, right
    subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-f", "f32le", "-ar", str(SR), "-ac", "2",
                    "-i", "-", "-b:a", "64k", str(out)], input=st.tobytes(), check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--page-data", required=True)
    ap.add_argument("--page-html", required=True)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--ids", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    (out / "audio").mkdir(parents=True)
    D = json.load(open(a.page_data, encoding="utf-8"))
    B = {s["id"]: s for s in D["samples"]}
    R = {}
    for d in a.raw.split(","):
        for f in sorted(Path(d).glob("part_*.jsonl")):
            for l in open(f, encoding="utf-8"):
                r = json.loads(l)
                R[r["id"]] = r
    W = defaultdict(list)
    for d, p in WIN.items():
        for l in open(UNISS / p, encoding="utf-8"):
            w = json.loads(l)
            W[w["meeting"]].append(w)
    samples, report = [], {}
    for sid in a.ids.split(","):
        s, d = B[sid], B[sid]["dir"]
        ws = sorted(W[sid], key=lambda w: w["start_s"])
        miss = [w["id"] for w in ws if w["id"] not in R]
        if miss:
            raise SystemExit(f"{sid}: {len(miss)} windows not decoded, e.g. {miss[:3]}")
        sub, name = SESS[d]
        src = load16(UNISS / "data/game_esports_si_v1" / d / "session_audio" / sub / name.format(m=sid))
        right = np.zeros(len(src) + 30 * SR, dtype=np.float32)
        tgt, n_spoken, n_err, n_coll, spk_seen = [], 0, 0, 0, set()
        for w in ws:
            r = R[w["id"]]
            n_err += bool(r.get("error"))
            n_coll += bool(r.get("collapse"))
            end = defaultdict(int)
            for g in r.get("segs", []):
                if not g.get("wav") or not g["text"].strip():
                    continue
                x = sf.read(g["wav"], dtype="float32")[0]
                k = g["spk"] if isinstance(g["spk"], int) else -1
                st = max(int(g["write_ms"]), end[k])
                end[k] = st + int(round(len(x) / 16))
                t = w["start_s"] + st / 1000
                a0 = int(t * SR)
                if a0 + len(x) > len(right):
                    right = np.concatenate([right, np.zeros(a0 + len(x) - len(right), dtype=np.float32)])
                right[a0: a0 + len(x)] += x
                tgt.append({"t": round(t, 2), "e": round(t + len(x) / SR, 2), "spk": k, "text": g["text"].strip()})
                n_spoken += 1
                spk_seen.add(k)
        tgt.sort(key=lambda x: x["t"])
        last = max([len(src)] + [int(x["e"] * SR) for x in tgt])
        mp3(normalize(src), normalize(right[:last]), out / "audio" / f"{sid}__E10.mp3")
        e10 = {"label": "LiveInterpret（E10）", "run": "e10_thinker_v2/checkpoint-1670",
               "audio": f"audio/{sid}__E10.mp3", "tgt": tgt}
        samples.append({**{k: v for k, v in s.items() if k != "arms"}, "arms": {"E10": e10, "B": s["arms"]["B"]}})
        report[sid] = {"windows": len(ws), "pieces_spoken": n_spoken, "errors": n_err, "collapsed_windows": n_coll,
                       "speakers": sorted(spk_seen), "speech_s": round(float((np.abs(right) > 1e-4).sum()) / SR, 1)}
        print(sid, report[sid], flush=True)
    data = {"samples": samples, "arm_order": ["E10", "B"],
            "arm_labels": {"E10": "LiveInterpret（E10）", "B": "现有（baseline B）"}, "stats": {},
            "foot": ("E10：LiveInterpret S5（Voxtral 编码器 + Qwen3.5-4B Thinker + 0.6B Talker，CosyVoice3 声码器），"
                     "Thinker 在 AMI 开源会议数据上按 baseline B 的方式微调（checkpoint-1670），Talker 保持冻结；"
                     "每位解说的音色按 B 的因果规则取（只用该说话人此前单独说话的片段）。B 是现有系统（联合双向 Thinker + T1c Talker），"
                     "两个版本的翻译文本不同。原声来自 YouTube 公开电竞解说，只用于试听，不参与任何训练。")}
    html = open(a.page_html, encoding="utf-8").read()
    i = html.index("const DATA = ")
    j = html.index("\n", i)
    html = html[:i] + "const DATA = " + json.dumps(data, ensure_ascii=False) + ";" + html[j:]
    rep = [("<title>王者荣耀解说同传试听</title>", "<title>LiveInterpret 电竞同传试听</title>"),
           ('<div class="label">多说话人同传 · baseline B · 电竞解说原声</div>',
            '<div class="label">多说话人同传 · LiveInterpret（E10） · 电竞解说原声</div>'),
           ("<h1>王者荣耀解说同传试听</h1>", "<h1>LiveInterpret 电竞同传试听</h1>"),
           ('<div class="label" id="h-tgt">同传译文（baseline B 实时输出）</div>',
            '<div class="label" id="h-tgt">同传译文（当前版本）</div>')]
    for old, new in rep:
        if html.count(old) != 1:
            raise SystemExit(f"template changed: {old!r}")
        html = html.replace(old, new)
    k0 = html.index('<p class="lede">')
    k1 = html.index("</p>", k0)
    n_zh = sum(s["dir"] == "zh2en" for s in samples)
    lede = (f'<p class="lede">{len(samples)} 段真实赛事解说，每段 5 分钟连续原声：{n_zh} 段中文解说译成英文，'
            f'{len(samples) - n_zh} 段英文解说译成中文，与 baseline B 试听页是同一批音频。同传系统：低延迟说话人分离，'
            '接 LiveInterpret 的 Thinker–Talker（E10：Thinker 按多说话人方式微调，Talker 冻结）。每位解说各有一路译音，'
            '按原声中的时间位置播放。<b>每段都可以在"版本"里切换 LiveInterpret（E10）和现有 baseline B</b>，'
            '两者的翻译文本不同；切换时保持当前播放位置。默认左声道放原声、右声道放同传译音，建议戴耳机听。')
    html = html[:k0] + lede + html[k1:]
    (out / "index.html").write_text(html, encoding="utf-8")
    (out / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(json.dumps({"page": str(out / "index.html"), "samples": len(samples)}))


if __name__ == "__main__":
    main()
