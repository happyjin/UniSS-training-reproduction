"""Tables of the FINAL TEST (reports/mist_w3po_v1/final_test_prereg_v1.zh-CN.md §3-§4): reads the outputs the chain /
post-hoc scripts wrote (whatever exists; missing pieces print as "-") and prints markdown; --json <new file> also writes
every number. CPU, read-only.

    python -m experiments.mist_w3po_v1.evaluation.final_test_table --set ami|ali [--json <new file>]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

DEC = Path("eval_outputs/mist_w3po_v1/decode")
SPD = Path("eval_outputs/mist_w3po_v1/speech")
TE = Path("eval_outputs/mist_w3po_v1/crf/t_eval")
Q = Path("eval_outputs/mist_w3po_v1/talker_crf/quality")
WIN = {"ami": "eval_outputs/mist_w3po_v1/data/windows_test_c2g06_v1/windows_test_online_zh2en.jsonl",
       "ali": "eval_outputs/mist_w3po_v1/data/windows_test_c2g06_v1/windows_test_online_ali.jsonl"}
REGS = ("all", "ov", "1spk")


def jload(p: Path):
    try:
        return json.load(open(p))
    except (OSError, ValueError):
        return None


def mean(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 4) if xs else None


def f(x, d=2):
    return "-" if x is None else f"{x:.{d}f}"


def unit(p: Path, key: str) -> dict:
    j = jload(p)
    if not j:
        return {}
    u = j.get("unit", {}).get(key) or {}
    out = {r: (u[r]["mean"] if u.get(r) else None) for r in REGS}
    out["n"] = u["all"]["n"] if u.get("all") else None
    return out


def final_share(name: str, win: str) -> float | None:
    hp = DEC / name / "hyp.jsonl"
    if not hp.exists():
        return None
    W = {json.loads(l)["id"]: json.loads(l) for l in open(win)}
    tot = fin = 0
    for l in open(hp):
        h = json.loads(l)
        w = W[h["id"]]
        for s in h["segments"]:
            n = len(s["text"].split())
            tot += n
            fin += n * (s["write_ms"] >= w["dur_ms"])
    return round(fin / max(tot, 1), 4)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", required=True, choices=["ami", "ali"])
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    S = a.set
    O = TE / "final_test_v1" / S
    FA, FV, FD = (f"ft_{S}_{t}_s3000" for t in ("aprime", "ve1", "d1v"))
    systems = [("(a) A′（c1+b1+s2）", f"{FA}__c1b1s2", FA), ("(b) A′ + PL-EM", f"{FA}__plem", FA),
               ("(c) A′rl + PL-EM", f"{FA}__rlplem", FA), ("(d) VE1rl + PL-EM", f"{FV}__rlplem", FV),
               ("(e) D1vrl + PL-EM", f"{FD}__rlplem", FD)]
    R = {"set": S, "text": {}, "speech": {}, "merge": {}, "gates": {}, "utmos_paired": {}}
    L = [f"## {S}", "", "### 文本（每个 Thinker 一份解码）", "",
         "| Thinker | tcpBLEU all / ov / 1spk | 文本 LAAL 现行 all / ov / 1spk | chunk all / ov / 1spk | 最终读词占比 | 读级块合并 |",
         "|---|---|---|---|---|---|"]
    for lab, F in (("A′", FA), ("VE1", FV), ("D1v", FD)):
        st = jload(DEC / F / "score_text.json")
        t = {r: (st or {}).get(f"tcpBLEU_{r}") for r in REGS}
        lo = unit(TE / F / f"textlat_{F}.json", "text_laal_online")
        lc = unit(O / f"textlat_chunk_{F}.json", "text_laal_online")
        mj = jload(DEC / f"{F}_rl" / "merge_read_segments.json") or {}
        fs = final_share(F, WIN[S])
        mcell = (f"pair_lost {mj.get('pair_lost')}，配对 {mj.get('pair_ok')} / {mj.get('windows')}（源 {mj.get('src_pair_ok')}）"
                 if mj else "-")
        R["text"][lab] = {"bleu": t, "laal_old": lo, "laal_chunk": lc, "final_read_share": fs}
        R["merge"][lab] = {k: mj.get(k) for k in ("windows", "pair_ok", "src_pair_ok", "pair_lost", "seg_in", "seg_out")}
        L.append(f"| {lab} | {f(t['all'])} / {f(t['ov'])} / {f(t['1spk'])} | "
                 f"{f(lo.get('all'), 3)} / {f(lo.get('ov'), 3)} / {f(lo.get('1spk'), 3)} | "
                 f"{f(lc.get('all'), 3)} / {f(lc.get('ov'), 3)} / {f(lc.get('1spk'), 3)} | "
                 f"{'-' if fs is None else f'{100 * fs:.1f}%'} | "
                 f"{mcell} |")
    L += ["", "### 语音质量与延迟（两种子；延迟为两种子平均）", "",
          "| 系统 | ASR-BLEU all（种子 1 / 2，均值） | ov / 1spk（均值） | 每条流语音 LAAL 现行 all / ov / 1spk | chunk all / ov / 1spk | "
          "EndOffset 现行 / chunk | 单通道 LAAL 现行 / chunk | 单元数 |",
          "|---|---|---|---|---|---|---|---|"]
    for lab, N, F in systems:
        sc = [jload(DEC / n / "score_asr.json") for n in (N, N + "_tts2")]
        b = {r: [(s or {}).get(f"tcpBLEU_{r}") for s in sc] for r in REGS}
        lat = {}
        for proto, paths in (("old", [TE / N / f"lat_{n}.json" for n in (N, N + "_tts2")]),
                             ("chunk", [O / f"latchunk_{n}.json" for n in (N, N + "_tts2")])):
            lat[proto] = {}
            for k in ("speech_laal_online", "speech_end_online", "speech_laal_online_1ch", "text_laal_online"):
                us = [unit(p, k) for p in paths]
                lat[proto][k] = {r: mean([u.get(r) for u in us]) for r in REGS}
                lat[proto][k]["seeds"] = [u.get("all") for u in us]
                lat[proto][k]["n"] = us[0].get("n") if us else None
        R["speech"][lab] = {"name": N, "bleu": b, "lat": lat}
        so, sc_ = lat["old"]["speech_laal_online"], lat["chunk"]["speech_laal_online"]
        L.append(f"| {lab} | {f(b['all'][0])} / {f(b['all'][1])}，{f(mean(b['all']))} | {f(mean(b['ov']))} / {f(mean(b['1spk']))} | "
                 f"{f(so['all'], 3)} / {f(so['ov'], 3)} / {f(so['1spk'], 3)} | {f(sc_['all'], 3)} / {f(sc_['ov'], 3)} / "
                 f"{f(sc_['1spk'], 3)} | {f(lat['old']['speech_end_online']['all'], 3)} / "
                 f"{f(lat['chunk']['speech_end_online']['all'], 3)} | {f(lat['old']['speech_laal_online_1ch']['all'], 3)} / "
                 f"{f(lat['chunk']['speech_laal_online_1ch']['all'], 3)} | {so.get('n') or '-'} |")
    L += ["", "### 听感（种子 1 / 2）", "",
          "| 系统 | UTMOS | SIM | talker WER | 强制截断块 | 块数 | 每块词数 | 单词块占比 | 平均块时长（p50 / p90） | "
          "句中停顿 ≥ 0.5 s：次 / 分钟，秒 / 分钟 | 全部停顿 次 / 分钟 |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for lab, N, F in systems:
        qs = [jload(Q / n / "quality.json") or {} for n in (N, N + "_tts2")]
        gs = [((jload(O / f"gap_{n}.json") or {}).get("fidelity") or {}).get("wer") for n in (N, N + "_tts2")]
        bp = {}
        p = O / f"block_pauses_{N}.jsonl"
        if p.exists():
            for l in open(p):
                r = json.loads(l)
                bp[r["talker"].rstrip("/").split("/")[-2]] = r
        bs = [bp.get(n, {}) for n in (N, N + "_tts2")]
        R["speech"][lab]["listen"] = {"utmos": [q.get("utmos_mean") for q in qs], "sim": [q.get("sim_mean") for q in qs],
                                      "talker_wer": gs, "blocks": bs}
        b0 = bs[0]
        s1w = "-" if b0.get("share_1w") is None else f"{100 * b0['share_1w']:.1f}%"

        def g(k, sub=None, d=3):
            v = [(x.get(k) or {}).get(sub) if sub else x.get(k) for x in bs]
            return " / ".join(f(y, d) for y in v)
        L.append(f"| {lab} | {' / '.join(f(q.get('utmos_mean'), 3) for q in qs)} | {' / '.join(f(q.get('sim_mean'), 3) for q in qs)} | "
                 f"{' / '.join('-' if w is None else f'{100 * w:.2f}%' for w in gs)} | "
                 f"{' / '.join(str(x.get('forced_blocks', '-')) for x in bs)} | {b0.get('blocks', '-')} | "
                 f"{f(b0.get('words_per_block'))} | {s1w} | "
                 f"{f((b0.get('block_dur_s') or {}).get('mean'))} s（{f((b0.get('block_dur_s') or {}).get('p50'))} / "
                 f"{f((b0.get('block_dur_s') or {}).get('p90'))}） | {g('pauses_mid_sent', 'per_min', 2)}，"
                 f"{g('pauses_mid_sent', 'sec_per_min', 2)} | {g('pauses_all', 'per_min', 2)} |")
    L += ["", "### 门槛（gate_2seed，描述性确认）", "",
          "| 比较 | 文本 all：差，CI，P | 文本 ov | 语音 all（两种子合并） | 三项全过 |", "|---|---|---|---|---|"]
    cmp_ = [("(c) vs (a)", f"{FA}__rlplem", f"{FA}__c1b1s2"), ("(c) vs (b)", f"{FA}__rlplem", f"{FA}__plem"),
            ("(d) vs (c)", f"{FV}__rlplem", f"{FA}__rlplem"), ("(e) vs (c)", f"{FD}__rlplem", f"{FA}__rlplem")]
    for lab, c, r in cmp_:
        gj = jload(O / f"gate_{c}_vs_{r}.json")
        R["gates"][lab] = gj and gj.get("items")
        if not gj:
            L.append(f"| {lab} | - | - | - | - |")
            continue
        it = gj["items"]

        def cell(k):
            x = it[k]
            return f"{x['B']} vs {x['A']}，**{x['diff']:+.2f}**，[{x['ci95'][0]:.2f}, {x['ci95'][1]:.2f}]，P {x['p_le0']:.3f}{'' if x['pass'] else '（不过）'}"
        L.append(f"| {lab} | {cell('text_all')} | {cell('text_ov')} | {cell('speech_all')} | "
                 f"{'是' if all(it[k]['pass'] for k in it) else '否'} |")
    L += ["", "### UTMOS 配对差（同流配对，两种子合并，按会议 bootstrap）", "", "| 比较 | 差 | 95% CI | 对数 | 会议 |", "|---|---|---|---|---|"]
    for tag, labs in (("c_vs_ab", [("(c) vs (a)", "a_c1b1s2"), ("(c) vs (b)", "b_plem")]),
                      ("d_vs_c", [("(d) vs (c)", "c_aprl_plem")]), ("e_vs_c", [("(e) vs (c)", "c_aprl_plem")])):
        pj = jload(O / f"utmos_paired_{tag}.json")
        for lab, k in labs:
            u = pj and pj["vs"][k]["utmos_paired"]
            ci = f"[{u['ci95'][0]:.4f}, {u['ci95'][1]:.4f}]" if u else "-"
            R["utmos_paired"][lab] = u
            L.append(f"| {lab} | {f(u['diff'], 4) if u else '-'} | {ci} | "
                     f"{u['pairs'] if u else '-'} | {u['meetings'] if u else '-'} |")
    print("\n".join(L))
    if a.json:
        with open(a.json, "x", encoding="utf-8") as fh:
            json.dump(R, fh, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
