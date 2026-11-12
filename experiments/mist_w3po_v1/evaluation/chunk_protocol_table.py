"""Chunk protocol latency table (Task C, user decision 2026-11-12 ~14:05 Prague; reports/mist_w3po_v1/
latency_chunk_protocol_table_v1.zh-CN.md). CPU only, dev only, read-only on every existing output.

The chunk protocol (reports/mist_w3po_v1/thinker_detector_specclose_prereg_v1.zh-CN.md 0.1) is an accounting
correction: the lowlat diarizer emits its posteriors in 9-frame chunks, frame f is known at
T(f) = (9 floor(f/9) + 13) * 80 ms <= f * 80 ms + 1.04 s, so a window opens at T(st) and a gap / cap close is known at
T(j - 1). Windows, posteriors, decodes and talker blocks are unchanged; only the detection stamps change.

Procedure = exactly the one that produced eval_outputs/mist_w3po_v1/detector_v1/ref_chunk/ (its jsons are reproduced
byte for byte): speech_latency_lit on the existing decode + talker outputs with
  old    --win eval_outputs/mist_w3po_v1/data/windows_zh2en_c2g06_v1/windows_dev_online.jsonl   (C2 windows)
  chunk  --win eval_outputs/mist_w3po_v1/detector_v1/windows_chunk_v1/windows_dev_online.jsonl (the same windows,
         chunk stamps; specclose build --judge off)
and speech_latency_lit --text-only on each text decode (text LAAL per text system). Every system below was decoded on
the C2 windows (3618 ids; T1 / T2 read every 0.5 s, "w05", on the same windows).
Decomposition (descriptive): open_only = the C2 windows with only the chunk OPEN stamp (old close), written to
<out>/windows_open_only.jsonl; open part = open_only - old, close part = chunk - open_only. Also the final-read word
share of each text decode (words of segments with write_ms >= the window duration, C2 windows).

Checks: (1) the chunk window file equals the C2 file row by row except the two stamps, never later; (2) the chunk
recomputation of the ref_chunk systems is byte-identical to ref_chunk; (3) the old recomputation equals every stored
old-protocol latency json on every field the stored json has (stored jsons from older tool versions lack
speech_laal_online_1ch). Writes <out>/{old,chunk,open_only}/lat_<name>.json, textlat_<text>.json (+ .log),
windows_open_only.jsonl, table.json, table.md. Refuses an existing <out>.

    python -m experiments.mist_w3po_v1.evaluation.chunk_protocol_table --out <new dir> [--jobs 8]
"""
from __future__ import annotations

import argparse
import filecmp
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

E = Path("eval_outputs/mist_w3po_v1")
WIN = {"old": E / "data/windows_zh2en_c2g06_v1/windows_dev_online.jsonl",
       "chunk": E / "detector_v1/windows_chunk_v1/windows_dev_online.jsonl"}
REF_CHUNK = E / "detector_v1/ref_chunk"
T = E / "crf/t_eval"
AP = "dev_crf_v2p_s3000_finalpost_s8_c2g06"

# label, text system, stored old-protocol latency json per TTS seed (its run / talker fields name the inputs)
SYSTEMS = [
    ("A′（c1+b1+s2）", "A′", [E / "speech_laal_v1/Ap.json", E / "speech_laal_v1/Ap_tts2.json"]),
    ("A′ + PL-EM", "A′", [E / f"talker_crf/eval/plem/lat_{AP}__plem_c1b1s2{s}.json" for s in ("", "_tts2")]),
    ("A′rl（c1+b1+s2）", "A′", [T / f"{AP}__rl/lat_{AP}__rl{s}.json" for s in ("", "_tts2")]),
    ("A′rl + PL-EM", "A′", [T / f"{AP}__rlplem/lat_{AP}__rlplem{s}.json" for s in ("", "_tts2")]),
    ("VE1rl + PL-EM", "VE1", [T / f"dev_ve1_s3000__rlplem/lat_dev_ve1_s3000__rlplem{s}.json" for s in ("", "_tts2")]),
    ("VE1（不合块）+ PL-EM", "VE1", [T / f"dev_ve1_s3000__segplem/lat_dev_ve1_s3000__segplem{s}.json" for s in ("", "_tts2")]),
    ("VE1-s2rl + PL-EM", "VE1-s2", [T / f"dev_ve1s2_s3000__rlplem/lat_dev_ve1s2_s3000__rlplem{s}.json" for s in ("", "_tts2")]),
    ("PMC rl + PL-EM", "PMC", [T / f"dev_pmc_s3000__rlplem/lat_dev_pmc_s3000__rlplem{s}.json" for s in ("", "_tts2")]),
    ("T1（c1+b1+s2）", "T1", [T / f"dev_t1_piece_t0_s3000_w05/lat_dev_t1_piece_t0_s3000_w05{s}.json" for s in ("", "_tts2")]),
    ("T2（c1+b1+s2）", "T2", [T / f"dev_t2_latent_t0_s3000_w05/lat_dev_t2_latent_t0_s3000_w05{s}.json" for s in ("", "_tts2")]),
    ("T2 + PL-EM", "T2", [T / f"dev_t2_latent_t0_s3000_w05__plem/lat_dev_t2_latent_t0_s3000_w05__plem{s}.json" for s in ("", "_tts2")]),
    ("T2rl（c1+b1+s2）", "T2", [T / f"dev_t2_latent_t0_s3000_w05__rl/lat_dev_t2_latent_t0_s3000_w05__rl{s}.json" for s in ("", "_tts2")]),
]
# text system -> (decode dir, stored old-protocol text-only json or None)
TEXT = {
    "A′": (E / f"decode/{AP}", None),
    "VE1": (E / "decode/dev_ve1_s3000", T / "dev_ve1_s3000/textlat_dev_ve1_s3000.json"),
    "VE1-s2": (E / "decode/dev_ve1s2_s3000", T / "dev_ve1s2_s3000/textlat_dev_ve1s2_s3000.json"),
    "PMC": (E / "decode/dev_pmc_s3000", T / "dev_pmc_s3000/textlat_dev_pmc_s3000.json"),
    "T1": (E / "decode/dev_t1_piece_t0_s3000_w05", None),
    "T2": (E / "decode/dev_t2_latent_t0_s3000_w05", None),
}
TEXT_SLUG = {"A′": "aprime", "VE1": "ve1", "VE1-s2": "ve1s2", "PMC": "pmc", "T1": "t1", "T2": "t2"}
# chunk recomputations that must equal ref_chunk byte for byte
REF_FILES = {f"lat_{AP}__rlplem.json", f"lat_{AP}__rlplem_tts2.json", "lat_dev_ve1_s3000__rlplem.json",
             "lat_dev_ve1_s3000__rlplem_tts2.json"}
SPEECH_KEYS = ("speech_laal_online", "speech_end_online", "speech_laal_online_1ch", "text_laal_online")
REGS = ("all", "ov", "1spk")
REF_SYS = "A′rl + PL-EM"
PROTOS = ("old", "chunk", "open_only")


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def check_windows() -> dict:
    a, c = jl(WIN["old"]), jl(WIN["chunk"])
    same = len(a) == len(c)
    later = 0
    for x, y in zip(a, c):
        same &= all(x[k] == y[k] for k in ("id", "meeting", "start_s", "dur_ms", "audio"))
        later += int(y["open_detect_ms"] > x["open_detect_ms"] or y["close_detect_ms"] > x["close_detect_ms"])
    d_open = [x["open_detect_ms"] - y["open_detect_ms"] for x, y in zip(a, c)]
    d_close = [x["close_detect_ms"] - y["close_detect_ms"] for x, y in zip(a, c)]
    return {"windows": len(a), "rows_equal_except_stamps": bool(same), "chunk_later_than_old": later,
            "open_earlier_ms_mean": round(sum(d_open) / len(d_open), 1), "open_earlier_ms_max": max(d_open),
            "close_earlier_ms_mean": round(sum(d_close) / len(d_close), 1), "close_earlier_ms_max": max(d_close),
            "pass": bool(same and later == 0)}


def final_share(dec: Path) -> dict:
    W = {w["id"]: w for w in jl(WIN["old"])}
    tot = fin = 0
    for h in jl(dec / "hyp.jsonl"):
        w = W[h["id"]]
        for s in h["segments"]:
            n = len(s["text"].split())
            tot += n
            fin += n * (s["write_ms"] >= w["dur_ms"])
    return {"words": tot, "final_read_words": fin, "share": round(fin / max(tot, 1), 4)}


def run_job(cmd: list[str], log: Path) -> int:
    env = dict(os.environ)
    env.update({"CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": ".:third_party/Megatron-LM", "OMP_NUM_THREADS": "4"})
    with open(log, "x", encoding="utf-8") as fh:
        return subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env).returncode


def metrics(p: Path, keys) -> dict:
    u = json.load(open(p))["unit"]
    out = {}
    for k in keys:
        if k in u:
            out[k] = {r: (u[k][r]["mean"] if u[k].get(r) else None) for r in REGS}
            out[k]["n"] = u[k]["all"]["n"] if u[k].get("all") else 0
    return out


def mean2(a, b):
    if a is None or b is None:
        return None
    return round((a + b) / 2, 4)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"拒绝覆盖已有输出: {out}")
    wchk = check_windows()
    if not wchk["pass"]:
        raise SystemExit(f"chunk 窗口与 C2 窗口不一致: {wchk}")
    PY = sys.executable
    out.mkdir(parents=True)
    WIN["open_only"] = out / "windows_open_only.jsonl"
    with open(WIN["open_only"], "x", encoding="utf-8") as fh:
        for x, y in zip(jl(WIN["old"]), jl(WIN["chunk"])):
            z = dict(x)
            z["open_detect_ms"] = y["open_detect_ms"]
            fh.write(json.dumps(z) + "\n")
    jobs = []   # (proto, kind, key, out json, cmd)
    speech = []
    for label, tsys, stored in SYSTEMS:
        seeds = []
        for sp in stored:
            st = json.load(open(sp))
            run, talker = Path(st["run"]), Path(st["talker"])
            assert (run / "hyp.jsonl").exists() and (talker / "results.jsonl").exists(), (run, talker)
            name = talker.parent.name
            seeds.append({"name": name, "run": str(run), "talker": str(talker), "stored": str(sp)})
            for proto in PROTOS:
                o = out / proto / f"lat_{name}.json"
                jobs.append((o, [PY, "-m", "experiments.mist_w3po_v1.evaluation.speech_latency_lit", "--run", str(run),
                                 "--talker", str(talker), "--win", str(WIN[proto]), "--out", str(o)]))
        speech.append({"label": label, "text": tsys, "seeds": seeds})
    for tsys, (dec, _) in TEXT.items():
        for proto in PROTOS:
            o = out / proto / f"textlat_{TEXT_SLUG[tsys]}.json"
            jobs.append((o, [PY, "-m", "experiments.mist_w3po_v1.evaluation.speech_latency_lit", "--run", str(dec),
                             "--text-only", "--win", str(WIN[proto]), "--out", str(o)]))
    for proto in PROTOS:
        (out / proto).mkdir()
    with ThreadPoolExecutor(a.jobs) as ex:
        rcs = list(ex.map(lambda j: run_job(j[1], j[0].with_suffix(".log")), jobs))
    bad = [str(j[0]) for j, rc in zip(jobs, rcs) if rc != 0]
    if bad:
        raise SystemExit(f"失败的作业: {bad}")

    # ---- checks
    ref = {}
    for f in sorted(REF_FILES):
        ref[f] = filecmp.cmp(out / "chunk" / f, REF_CHUNK / f, shallow=False)
    ref["textlat_aprime.json"] = filecmp.cmp(out / "chunk/textlat_aprime.json", REF_CHUNK / "textlat_aprime.json",
                                             shallow=False)
    stored_chk = []
    for s in speech:
        for sd in s["seeds"]:
            u_st = json.load(open(sd["stored"]))["unit"]
            u_new = json.load(open(out / "old" / f"lat_{sd['name']}.json"))["unit"]
            keys = [k for k in u_st]
            diff = [k for k in keys if u_st[k] != u_new.get(k)]
            stored_chk.append({"name": sd["name"], "stored": sd["stored"], "fields": len(keys), "differ": diff,
                               "missing_in_stored": [k for k in SPEECH_KEYS if k not in u_st]})
    for tsys, (dec, st) in TEXT.items():
        if st is None:
            continue
        u_st = json.load(open(st))["unit"]
        u_new = json.load(open(out / "old" / f"textlat_{TEXT_SLUG[tsys]}.json"))["unit"]
        stored_chk.append({"name": f"text {tsys}", "stored": str(st), "fields": len(u_st),
                           "differ": [k for k in u_st if u_st[k] != u_new.get(k)], "missing_in_stored": []})

    # ---- table
    rows = []
    for s in speech:
        r = {"label": s["label"], "text_system": s["text"], "names": [sd["name"] for sd in s["seeds"]],
             "runs": [sd["run"] for sd in s["seeds"]], "win": {p: str(WIN[p]) for p in WIN}}
        for proto in PROTOS:
            m = [metrics(out / proto / f"lat_{sd['name']}.json", SPEECH_KEYS) for sd in s["seeds"]]
            r[proto] = {"per_seed": m,
                        "mean": {k: {g: mean2(m[0][k][g], m[1][k][g]) for g in REGS} for k in SPEECH_KEYS},
                        "units": [x["speech_laal_online"]["n"] for x in m]}
        r["delta"] = {k: {g: (round(r["chunk"]["mean"][k][g] - r["old"]["mean"][k][g], 4)
                              if r["old"]["mean"][k][g] is not None else None) for g in REGS} for k in SPEECH_KEYS}
        r["delta_open_part"] = {k: round(r["open_only"]["mean"][k]["all"] - r["old"]["mean"][k]["all"], 4)
                                for k in SPEECH_KEYS}
        r["delta_close_part"] = {k: round(r["chunk"]["mean"][k]["all"] - r["open_only"]["mean"][k]["all"], 4)
                                 for k in SPEECH_KEYS}
        rows.append(r)
    trows = []
    for tsys, (dec, _) in TEXT.items():
        r = {"text_system": tsys, "decode": str(dec), "final_read": final_share(dec)}
        for proto in PROTOS:
            r[proto] = metrics(out / proto / f"textlat_{TEXT_SLUG[tsys]}.json", ("text_laal_online",))["text_laal_online"]
        r["delta"] = {g: round(r["chunk"][g] - r["old"][g], 4) for g in REGS}
        r["delta_open_part"] = round(r["open_only"]["all"] - r["old"]["all"], 4)
        r["delta_close_part"] = round(r["chunk"]["all"] - r["open_only"]["all"], 4)
        trows.append(r)
    # differences to the reference system and rankings under both protocols
    refr = next(r for r in rows if r["label"] == REF_SYS)
    vs = {}
    for r in rows:
        vs[r["label"]] = {}
        for k in ("speech_laal_online", "speech_end_online", "speech_laal_online_1ch"):
            o = r["old"]["mean"][k]["all"] - refr["old"]["mean"][k]["all"]
            c = r["chunk"]["mean"][k]["all"] - refr["chunk"]["mean"][k]["all"]
            vs[r["label"]][k] = {"old": round(o, 4), "chunk": round(c, 4), "change": round(c - o, 4)}
    rank = {p: [r["label"] for r in sorted(rows, key=lambda x: x[p]["mean"]["speech_laal_online"]["all"])]
            for p in ("old", "chunk")}
    trank = {p: [r["text_system"] for r in sorted(trows, key=lambda x: x[p]["all"])] for p in ("old", "chunk")}
    res = {"windows_check": wchk, "ref_chunk_identical": ref, "stored_old_check": stored_chk,
           "speech": rows, "text": trows, "vs_ref": {"ref": REF_SYS, "by_system": vs},
           "rank_speech_laal": rank, "rank_text_laal": trank,
           "same_rank_speech": rank["old"] == rank["chunk"], "same_rank_text": trank["old"] == trank["chunk"]}
    with open(out / "table.json", "x", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, ensure_ascii=False)

    # ---- markdown
    f3 = lambda x: "—" if x is None else f"{x:.3f}"
    sg = lambda x: "—" if x is None else f"{x:+.3f}"
    L = ["| 系统 | 语音 LAAL 现行 | chunk | 差 | ov 现行 / chunk | 1spk 现行 / chunk | EndOffset 现行 / chunk（差） | "
         "单通道 LAAL 现行 / chunk（差） | 单元数 |", "|" + "---|" * 9]
    for r in rows:
        o, c, d = r["old"]["mean"], r["chunk"]["mean"], r["delta"]
        L.append(f"| {r['label']} | {f3(o['speech_laal_online']['all'])} | {f3(c['speech_laal_online']['all'])} | "
                 f"{sg(d['speech_laal_online']['all'])} | {f3(o['speech_laal_online']['ov'])} / "
                 f"{f3(c['speech_laal_online']['ov'])} | {f3(o['speech_laal_online']['1spk'])} / "
                 f"{f3(c['speech_laal_online']['1spk'])} | {f3(o['speech_end_online']['all'])} / "
                 f"{f3(c['speech_end_online']['all'])}（{sg(d['speech_end_online']['all'])}） | "
                 f"{f3(o['speech_laal_online_1ch']['all'])} / {f3(c['speech_laal_online_1ch']['all'])}"
                 f"（{sg(d['speech_laal_online_1ch']['all'])}） | {r['old']['units'][0]} |")
    L += ["", "| 文本系统 | 文本 LAAL 现行 | chunk | 差 | ov 现行 / chunk | 1spk 现行 / chunk | 单元数 |", "|" + "---|" * 7]
    for r in trows:
        L.append(f"| {r['text_system']} | {f3(r['old']['all'])} | {f3(r['chunk']['all'])} | {sg(r['delta']['all'])} | "
                 f"{f3(r['old']['ov'])} / {f3(r['chunk']['ov'])} | {f3(r['old']['1spk'])} / {f3(r['chunk']['1spk'])} | "
                 f"{r['old']['n']} |")
    fs = {r["text_system"]: r["final_read"]["share"] for r in trows}
    L += ["", "| 系统 | 语音 LAAL 差 | 其中开窗部分 | 其中关窗部分 | 最终读词占比（文本系统） | 对 A′rl + PL-EM 现行 / chunk（变化） |",
          "|" + "---|" * 6]
    for r in rows:
        v = vs[r["label"]]["speech_laal_online"]
        L.append(f"| {r['label']} | {sg(r['delta']['speech_laal_online']['all'])} | "
                 f"{sg(r['delta_open_part']['speech_laal_online'])} | {sg(r['delta_close_part']['speech_laal_online'])} | "
                 f"{fs[r['text_system']]:.1%} | {sg(v['old'])} / {sg(v['chunk'])}（{sg(v['change'])}） |")
    (out / "table.md").open("x", encoding="utf-8").write("\n".join(L) + "\n")
    print(json.dumps({"windows_check": wchk, "ref_chunk_identical": ref,
                      "stored_old_differ": [x for x in stored_chk if x["differ"]],
                      "same_rank_speech": res["same_rank_speech"], "same_rank_text": res["same_rank_text"]},
                     ensure_ascii=False, indent=1))
    print("\n".join(L))


if __name__ == "__main__":
    main()
