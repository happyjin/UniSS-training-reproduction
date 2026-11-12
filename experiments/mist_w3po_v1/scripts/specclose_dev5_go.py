"""SpecClose dev5 GO lines (reports/mist_w3po_v1/thinker_detector_specclose_prereg_v1.zh-CN.md §3), CPU: the full dev
decode cut to the 5 dev5 meetings (the meetings of windows_zh2en_c2g06_v1_dev5; SpecClose windows have their own ids,
so the cut is by meeting) -> decode/dev5cut_<name>; score_run (tcpBLEU) and speech_latency_lit --text-only on the
SpecClose windows of those meetings. GO = text all >= 28.16 and ov >= 23.92 (A' dev5 - 0.2) and text LAAL <= 3.335 s
(A' dev5 under the chunk protocol 3.435 - 0.10). Writes crf/t_eval/dev5cut_<name>/go_specclose.json (noclobber);
exit 0 always (GO is the "GO" field).
    python specclose_dev5_go.py <full decode name> <SpecClose windows jsonl>"""
import json
import os
import subprocess
import sys
from pathlib import Path

PY = "/opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python"
W5_C2 = "eval_outputs/mist_w3po_v1/data/windows_zh2en_c2g06_v1_dev5/windows_dev_online.jsonl"
REF = "data/ami_multispk_si_v1/e2e_gstar_zh2en_v1/windows_validation_ref.jsonl"
LINES = {"text_all_min": 28.16, "text_ov_min": 23.92, "text_laal_max": 3.335,
         "aprime_dev5": {"text_all": 28.36, "text_ov": 24.12, "text_laal_old": 3.635, "text_laal_chunk": 3.435}}


def main():
    name, win = sys.argv[1], sys.argv[2]
    D = Path(os.environ.get("SPECCLOSE_DEC_ROOT", "eval_outputs/mist_w3po_v1/decode"))   # env: smoke tests only
    c = D / f"dev5cut_{name}"
    o = Path(os.environ.get("SPECCLOSE_T_ROOT", "eval_outputs/mist_w3po_v1/crf/t_eval")) / f"dev5cut_{name}"
    for p in (c, o):
        if p.exists():
            raise SystemExit(f"拒绝覆盖已有输出: {p}")
    ms = {json.loads(l)["meeting"] for l in open(W5_C2)}
    c.mkdir(parents=True)
    o.mkdir(parents=True)
    w5 = o / "windows_dev5.jsonl"
    ids = set()
    with open(w5, "x") as fh:
        for l in open(win):
            w = json.loads(l)
            if w["meeting"] in ms:
                fh.write(l)
                ids.add(w["id"])
    n = 0
    with open(c / "hyp.jsonl", "x") as fh:
        for l in open(D / name / "hyp.jsonl"):
            if json.loads(l)["id"] in ids:
                fh.write(l)
                n += 1
    r1 = subprocess.run([PY, "-m", "experiments.mist_w3po_v1.evaluation.score_run", "--run", str(c), "--windows", str(w5),
                         "--ref-windows", REF, "--set", "zh2en"], capture_output=True, text=True)
    r2 = subprocess.run([PY, "-m", "experiments.mist_w3po_v1.evaluation.speech_latency_lit", "--run", str(c),
                         "--text-only", "--win", str(w5), "--out", str(o / "textlat.json")], capture_output=True, text=True)
    res = {"name": name, "win": win, "lines": LINES, "windows": len(ids), "hyp_rows": n,
           "rc": [r1.returncode, r2.returncode]}
    if r1.returncode == 0 and r2.returncode == 0:
        sc = json.loads((c / "score_text.json").read_text())
        tl = json.loads((o / "textlat.json").read_text())["unit"]["text_laal_online"]
        res.update({"text_all": sc["tcpBLEU_all"], "text_ov": sc["tcpBLEU_ov"], "text_1spk": sc["tcpBLEU_1spk"],
                    "text_laal": tl["all"]["mean"], "text_laal_ov": tl["ov"]["mean"], "text_laal_1spk": tl["1spk"]["mean"]})
        res["line1_text"] = bool(res["text_all"] >= LINES["text_all_min"] and res["text_ov"] >= LINES["text_ov_min"])
        res["line2_text_laal"] = bool(res["text_laal"] <= LINES["text_laal_max"])
        res["GO"] = res["line1_text"] and res["line2_text_laal"]
    else:
        res["GO"] = False
        res["error"] = (r1.stderr + r2.stderr)[-1500:]
    with open(o / "go_specclose.json", "x", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
