"""PMC dev5 quick-check GO lines (reports/mist_w3po_v1/thinker_pmc_prereg_v1.zh-CN.md §3.2) from the dev5 cut of the full dev
decode (t_dev5_cut.sh): text all >= 28.16 and ov >= 23.92 (A' - 0.2); text LAAL (unit text_laal_online all) < 3.26 s
(VE1 dev5); final-read word share <= 49.7 %. Descriptive: the dev5 CPU speech proxy (joint_speech_proxy, read-level
blocks) vs A''s. Writes crf/t_eval/dev5cut_<name>/go_pmc.json (noclobber); exit 0 always (GO is the "GO" field).
    python pmc_dev5_go.py <full decode name>"""
import json
import subprocess
import sys
from pathlib import Path

PY = "/opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python"
W5 = "eval_outputs/mist_w3po_v1/data/windows_zh2en_c2g06_v1_dev5/windows_dev_online.jsonl"
AP5 = "logs/mist_w3po_v1/j1/proxy/proxy_dev5_aprime.json"
LINES = {"text_all_min": 28.16, "text_ov_min": 23.92, "text_laal_max_excl": 3.26, "final_share_max": 0.497}


def main():
    name = sys.argv[1]
    c = Path("eval_outputs/mist_w3po_v1/decode") / f"dev5cut_{name}"
    o = Path("eval_outputs/mist_w3po_v1/crf/t_eval") / f"dev5cut_{name}"
    sc = json.loads((c / "score_text.json").read_text())
    tl = json.loads((o / "textlat.json").read_text())["unit"]["text_laal_online"]
    fs = json.loads((o / "final_share_dev5.json").read_text())
    res = {"name": name, "lines": LINES, "text_all": sc["tcpBLEU_all"], "text_ov": sc["tcpBLEU_ov"],
           "text_1spk": sc["tcpBLEU_1spk"], "text_laal": tl["all"]["mean"], "text_laal_ov": tl["ov"]["mean"],
           "text_laal_1spk": tl["1spk"]["mean"], "final_share": fs["final_share"], "windows": fs["windows"]}
    res["line1_text"] = bool(res["text_all"] >= LINES["text_all_min"] and res["text_ov"] >= LINES["text_ov_min"])
    res["line2_text_laal"] = bool(res["text_laal"] < LINES["text_laal_max_excl"])
    res["line3_final_share"] = bool(res["final_share"] <= LINES["final_share_max"])
    res["GO"] = res["line1_text"] and res["line2_text_laal"] and res["line3_final_share"]
    p = subprocess.run([PY, "-m", "experiments.mist_w3po_v1.crf.joint_speech_proxy", "--run", str(c), "--win", W5,
                        "--merge", "1", "--out", str(o / "proxy_rl.json")], capture_output=True, text=True)
    if p.returncode == 0 and Path(AP5).exists():
        res["proxy_rl_descriptive"] = json.loads((o / "proxy_rl.json").read_text())["speech_laal"]
        res["aprime_proxy"] = json.loads(Path(AP5).read_text())["speech_laal"]
    else:
        res["proxy_error"] = (p.stderr or "")[-500:]
    with open(o / "go_pmc.json", "x", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
