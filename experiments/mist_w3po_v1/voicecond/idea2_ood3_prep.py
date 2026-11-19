"""Idea 2 prereg v3 §C: OOD-3 subset inputs (AMI test EN2002d + IS1009c in both directions, AliMeeting test R8009_M8021 +
R8009_M8024; the sha1 rule of the prereg). Writes, never overwriting:
  data/open_voicecond_v1/idea2/ood3/<set>_{win,chunkwin,ref}.jsonl   windows / chunk-protocol windows / reference windows
  eval_outputs/mist_w3po_v1/decode/idea2_ood3_<set>_txt/{hyp.jsonl, score_text.json}   baseline B's text, filtered
<set> = ami (zh2en, Chinese-dubbed AMI), en (en2zh), ali (zh2en, AliMeeting).
    python3 experiments/mist_w3po_v1/voicecond/idea2_ood3_prep.py
"""
import json, shutil, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[3]
DD = "eval_outputs/mist_w3po_v1/data"
B = "eval_outputs/mist_w3po_v1/decode/jtest_b{s}_joint_v8h6k_20261113T073658Z_s06000"
SETS = {
    "ami": {"win": f"{DD}/windows_test_c2g06_v1/windows_test_online_zh2en.jsonl",
            "chunk": f"{DD}/windows_test_c2g06_chunk_v1/zh2en/windows_dev_online.jsonl",
            "ref": "data/ami_multispk_si_v1/e2e_gstar_zh2en_v1/windows_test_ref.jsonl", "M": ["EN2002d", "IS1009c"]},
    "en": {"win": f"{DD}/windows_test_en2zh_c2g06_v1/windows_test_online_en2zh.jsonl",
           "chunk": f"{DD}/windows_test_en2zh_c2g06_chunk_v1/windows_dev_online.jsonl",
           "ref": "data/ami_multispk_si_v1/e2e_gstar_v1/windows_test_ref.jsonl", "M": ["EN2002d", "IS1009c"]},
    "ali": {"win": f"{DD}/windows_test_c2g06_v1/windows_test_online_ali.jsonl",
            "chunk": f"{DD}/windows_test_c2g06_chunk_v1/ali/windows_dev_online.jsonl",
            "ref": "data/diarist_alimeeting_refs_v1/windows_test_ref_human.jsonl", "M": ["R8009_M8021", "R8009_M8024"]},
}
OD = REPO / "data/open_voicecond_v1/idea2/ood3"


def meeting_of(r):
    if "meeting" in r:
        return r["meeting"]
    return r["id"].rsplit("_on_", 1)[0] if "_on_" in r["id"] else r["id"].rsplit("_", 1)[0]


def filt(src, dst, M):
    if dst.exists():
        print("exists", dst); return
    rows = [l for l in open(REPO / src, encoding="utf-8") if meeting_of(json.loads(l)) in M]
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "x", encoding="utf-8") as f:
        f.writelines(rows)
    print(dst.name, len(rows))


for s, S in SETS.items():
    filt(S["win"], OD / f"{s}_win.jsonl", S["M"])
    filt(S["chunk"], OD / f"{s}_chunkwin.jsonl", S["M"])
    filt(S["ref"], OD / f"{s}_ref.jsonl", S["M"])
    td = REPO / f"eval_outputs/mist_w3po_v1/decode/idea2_ood3_{s}_txt"
    if not (td / "hyp.jsonl").exists():
        td.mkdir(parents=True, exist_ok=True)
        src = REPO / B.format(s=s)
        rows = [l for l in open(src / "hyp.jsonl", encoding="utf-8") if json.loads(l)["id"].rsplit("_on_", 1)[0] in S["M"]]
        with open(td / "hyp.jsonl", "x", encoding="utf-8") as f:
            f.writelines(rows)
        shutil.copyfile(src / "score_text.json", td / "score_text.json")
        print(td.name, "hyp rows", len(rows))
