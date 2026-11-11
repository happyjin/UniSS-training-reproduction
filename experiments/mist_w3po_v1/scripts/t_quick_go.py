"""T4 / T5 registered dev5 quick-check GO lines (reports/mist_w3po_v1/thinker_crf_t_prereg_v1.zh-CN.md §T4, §T5):
final-read word share <= 49.7 % (A' dev5), text LAAL (unit text_laal_online, all) < 3.64 s, text all >= 28.16 and
ov >= 23.92. Writes <out>/go.json (noclobber) and prints it; exit 0 always (GO is the "GO" field).
    python t_quick_go.py <decode name> <t_eval out dir> <dev5 windows>"""
import json
import subprocess
import sys
from pathlib import Path

PY = "/opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python"


def main():
    name, out, win = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
    d = Path("eval_outputs/mist_w3po_v1/decode") / name
    sc = json.loads((d / "score_text.json").read_text())
    tl = json.loads((out / f"textlat_{name}.json").read_text())["unit"]["text_laal_online"]
    fs = json.loads(subprocess.run([PY, "experiments/mist_w3po_v1/scripts/final_read_share.py", str(d / "hyp.jsonl"), win],
                                   capture_output=True, text=True, check=True).stdout)
    tb = json.loads(subprocess.run([PY, "experiments/mist_w3po_v1/scripts/talker_block_words.py", str(d / "hyp.jsonl"), win],
                                   capture_output=True, text=True, check=True).stdout)
    res = {"name": name, "text_all": sc["tcpBLEU_all"], "text_ov": sc["tcpBLEU_ov"], "text_1spk": sc["tcpBLEU_1spk"],
           "text_laal": tl["all"]["mean"], "text_laal_ov": tl["ov"]["mean"], "text_laal_1spk": tl["1spk"]["mean"],
           "final_share": fs["final_share"], "ge4s_only_final": fs["ge4s_only_final"], "ge4s_windows": fs["ge4s_windows"],
           "talker_blocks": tb,
           "go_final": fs["final_share"] <= 0.497, "go_laal": tl["all"]["mean"] < 3.64,
           "go_text": sc["tcpBLEU_all"] >= 28.16 and sc["tcpBLEU_ov"] >= 23.92}
    res["GO"] = bool(res["go_final"] and res["go_laal"] and res["go_text"])
    with open(out / "go.json", "x") as fh:
        fh.write(json.dumps(res, indent=1))
    print(json.dumps(res))


if __name__ == "__main__":
    main()
