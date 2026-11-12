"""dev5 quick-check GO lines of D1v and J1-v2 (reports/mist_w3po_v1/thinker_dagger_d1v_prereg_v1.zh-CN.md §3.2,
thinker_joint_crf_v2_prereg_v1.zh-CN.md §4.2), from the dev5 cut of the full dev decode (t_dev5_cut.sh):
  1. text all >= 28.16 and ov >= 23.92 (A' - 0.2)
  2. text LAAL (unit text_laal_online all) < 3.26 s (VE1's dev5)
Descriptive (no line): final-read word share; the stall classes of t_dev5_cut's stall_diag.txt (A / B / C, A share
among the >= 6 s windows, B mean silent reads) next to VE1's dev5 (12.9 %, 2.15); talker blocks (segment / read_spk,
talker_block_words.py); the dev5 CPU speech proxy (joint_speech_proxy, read-level blocks) vs A''s.
Writes crf/t_eval/dev5cut_<name>/go_<tag>.json (noclobber); exit 0 always (GO is the "GO" field).
    python v4a_dev5_go.py <full decode name> <tag: d1v | j1v2>"""
import json
import subprocess
import sys
from pathlib import Path

PY = "/opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python"
W5 = "eval_outputs/mist_w3po_v1/data/windows_zh2en_c2g06_v1_dev5/windows_dev_online.jsonl"
AP5 = "logs/mist_w3po_v1/j1/proxy/proxy_dev5_aprime.json"
LINES = {"text_all_min": 28.16, "text_ov_min": 23.92, "text_laal_max_excl": 3.26}
VE1_DEV5 = {"ge6s_A_share": 0.129, "B_mean_silent_reads": 2.15, "final_share": 0.413, "text_laal": 3.26}


def stall(path: Path) -> dict:
    if not path.exists():
        return {"error": "no stall_diag.txt"}
    line = path.read_text().splitlines()[0]
    d = json.loads(line[line.index("{"):])
    ge6 = d.get("by_dur", {}).get(">=6s", {})
    n6 = sum(ge6.values())
    return {"A": d["A"], "B": d["B"], "C": d["C"], "ge6s_A_share": round(ge6.get("A", 0) / n6, 4) if n6 else None,
            "ge6s_windows": n6, "B_mean_silent_reads": d["B_mean_silent_reads"],
            "note": "first line of stall_diag.txt (its label is hard-coded 'T1 (0.5 s reads)'; it is this model)"}


def main():
    name, tag = sys.argv[1], sys.argv[2]
    c = Path("eval_outputs/mist_w3po_v1/decode") / f"dev5cut_{name}"
    o = Path("eval_outputs/mist_w3po_v1/crf/t_eval") / f"dev5cut_{name}"
    sc = json.loads((c / "score_text.json").read_text())
    tl = json.loads((o / "textlat.json").read_text())["unit"]["text_laal_online"]
    fs = json.loads((o / "final_share_dev5.json").read_text())
    res = {"name": name, "tag": tag, "lines": LINES, "text_all": sc["tcpBLEU_all"], "text_ov": sc["tcpBLEU_ov"],
           "text_1spk": sc["tcpBLEU_1spk"], "text_laal": tl["all"]["mean"], "text_laal_ov": tl["ov"]["mean"],
           "text_laal_1spk": tl["1spk"]["mean"]}
    res["line1_text"] = bool(res["text_all"] >= LINES["text_all_min"] and res["text_ov"] >= LINES["text_ov_min"])
    res["line2_text_laal"] = bool(res["text_laal"] < LINES["text_laal_max_excl"])
    res["GO"] = res["line1_text"] and res["line2_text_laal"]
    res["descriptive"] = {"final_share": fs["final_share"], "stall": stall(o / "stall_diag.txt"), "ve1_dev5": VE1_DEV5}
    st = res["descriptive"]["stall"]
    if st.get("ge6s_A_share") is not None:
        res["descriptive"]["mechanism_below_ve1"] = {
            "ge6s_A": bool(st["ge6s_A_share"] < VE1_DEV5["ge6s_A_share"]),
            "B_silent": bool(st["B_mean_silent_reads"] < VE1_DEV5["B_mean_silent_reads"])}
    p = subprocess.run([PY, "experiments/mist_w3po_v1/scripts/talker_block_words.py", str(c / "hyp.jsonl"), W5],
                       capture_output=True, text=True)
    res["descriptive"]["talker_blocks"] = json.loads(p.stdout) if p.returncode == 0 and p.stdout.strip().startswith(
        "{") else (p.stdout + p.stderr)[-500:]
    p = subprocess.run([PY, "-m", "experiments.mist_w3po_v1.crf.joint_speech_proxy", "--run", str(c), "--win", W5,
                        "--merge", "1", "--out", str(o / "proxy_rl.json")], capture_output=True, text=True)
    if p.returncode == 0 and Path(AP5).exists():
        res["descriptive"]["proxy_rl"] = json.loads((o / "proxy_rl.json").read_text())["speech_laal"]
        res["descriptive"]["aprime_proxy"] = json.loads(Path(AP5).read_text())["speech_laal"]
    else:
        res["descriptive"]["proxy_error"] = (p.stderr or "")[-500:]
    with open(o / f"go_{tag}.json", "x", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
