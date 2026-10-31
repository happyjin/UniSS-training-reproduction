"""gpuq7 jobs for paper v6 decodes (docs/mist_acl_review_v5_plan.zh-CN.md, paper/gstar_si_v6/docs/prereg_v6.zh-CN.md).

One job = one card = one gstar_multi_v6 process (one Thinker copy, ~72 GB) with W decode workers (~7 GB each at
batch 8); declared memory 72,000 + 7,000 W MiB. Every output goes to eval_outputs/.../gstar_v6_20261031/<name>/
(hyp_s<k>.jsonl, infer_s<k>.log); run files to reports/gstar_si_v6/jobs/<phase>/ (mode "x": nothing is overwritten).

  dec1   whole-meeting online runs (M01), LA2 / fixed-lag on the prefix-trained models (M04 1-2), streaming, oracle-turn
         and context cascades (M04 4, M05 2-3), single-stream read timing (M03)
  dec2:<what>  decodes of the v6-trained models (each phase once its model is merged): zhchar (dev gate + test, prereg
         §4), seeds (E-seed), dlabc (target control), cturn (turn-trained cascade), o2seeds, t13 (batch invariance)
  dec1c  dec1b without LocalAgreement on MIST-Simul-M (compute; prereg_v6 amendment 1); dec1b never started a job
  dec1b  the dec1 jobs that had not started when the dec1 queue was stopped (its first six jobs ran with 4 workers per
         Thinker and kept running), re-made with 6 workers per Thinker (a worker takes ~5 GB at batch 8)

    python3 reports/gstar_si_v6/make_v6_jobs.py dec1 > reports/gstar_si_v6/jobs/dec1.json
"""
import json
import sys
from pathlib import Path

EV = "eval_outputs/uniss_qwen3_omni_v1"
OUT = f"{EV}/gstar_v6_20261031"
D = "data/ami_multispk_si_v1"
ON = "data/online_v6_20261031"
T = {"S2DL": f"{EV}/thinker_gstar_s2dl_20261029b/thinker_merged",
     "S2DLZ": f"{EV}/thinker_gstar_s2dl_zh2en_3k_20261030/thinker_merged",
     "S2DLM": f"{EV}/thinker_gstar_s2dlm_3k_20261030/thinker_merged",
     "O2": f"{EV}/thinker_gstar_o2_20261029b/thinker_merged",
     "O2N": f"{EV}/thinker_gstar_o2nosd_20261029b/thinker_merged"}
WIN = {"en2zh": f"{D}/e2e_gstar_stream_v1/windows_test_ref.jsonl", "zh2en": f"{D}/e2e_gstar_zh2en_v1/windows_test_ref.jsonl",
       "on_en2zh": f"{ON}/windows_test_online_en2zh.jsonl", "on_zh2en": f"{ON}/windows_test_online_zh2en.jsonl",
       "on_ali": f"{ON}/windows_test_online_ali.jsonl"}
J = Path("reports/gstar_si_v6/jobs")
PYM = "/opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python"
THINKER_MIB, WORKER_MIB, BATCH = 72000, 7000, 8


def online(name, thinker, win, d, policy, n, extra=(), batch=BATCH):
    return [{"module": "gstar_si_stream_online_v6",
             "argv": ["--thinker", T[thinker], "--windows", win, "--out", f"{OUT}/{name}/hyp_s{k}.jsonl", "--shard", str(k),
                      "--num-shards", str(n), "--policy", policy, "--window", "1", "--post-key", "lowlat", "--dir", d,
                      "--batch", str(batch)] + list(extra),
             "log": f"{OUT}/{name}/infer_s{k}.log"} for k in range(n)]


def offline(name, thinker, win, d, n, extra=()):
    return [{"module": "gstar_si_stream",
             "argv": ["--thinker", T[thinker], "--windows", win, "--out", f"{OUT}/{name}/hyp_s{k}.jsonl", "--shard", str(k),
                      "--num-shards", str(n), "--inst", "gstar", "--post-key", "offline", "--dir", d] + list(extra),
             "log": f"{OUT}/{name}/infer_s{k}.log"} for k in range(n)]


def jobs_of(phase, thinker, runs, per_job, worker_mib=WORKER_MIB, thinker_mib=THINKER_MIB):
    out = []
    for i in range(0, len(runs), per_job):
        rs = runs[i: i + per_job]
        f = J / phase / f"{thinker}_{i // per_job}.json"
        f.parent.mkdir(parents=True, exist_ok=True)
        for r in rs:
            Path(r["log"]).parent.mkdir(parents=True, exist_ok=True)
        with open(f, "x") as fh:
            json.dump(rs, fh, indent=1)
        jid = f"{phase}_{thinker}_{i // per_job}"
        out.append({"id": jid, "gpus": 1, "vllm": False, "deps": [], "mem": thinker_mib + worker_mib * len(rs),
                    "cmd": f"CUDA_VISIBLE_DEVICES=$GPU0 PYTHONPATH=.:third_party/Megatron-LM {PYM} -m "
                           f"experiments.uniss_qwen3_omni_v1.evaluation.gstar_multi_v6 --thinker {T[thinker]} {f}"})
    return out


def dec1():
    jobs = []
    jobs += jobs_of("dec1", "O2", online("on_en2zh_O2_LA2", "O2", WIN["on_en2zh"], "en2zh", "la2", 4)
                    + online("on_zh2en_O2_LA2", "O2", WIN["on_zh2en"], "zh2en", "la2", 4)
                    + online("on_ali_O2_LA2", "O2", WIN["on_ali"], "zh2en", "la2", 4), 4)
    jobs += jobs_of("dec1", "S2DL", online("on_en2zh_S2dl", "S2DL", WIN["on_en2zh"], "en2zh", "wait", 2)
                    + online("en2zh_S2dl_LA2_d1", "S2DL", WIN["en2zh"], "en2zh", "la2", 2)
                    + [r for L in (1, 2, 3) for r in online(f"en2zh_S2dl_lag{L}", "S2DL", WIN["en2zh"], "en2zh", "lag", 2,
                                                            ["--lag-ms", str(1000 * L)])], 4)
    jobs += jobs_of("dec1", "S2DLZ", online("on_zh2en_S2dl", "S2DLZ", WIN["on_zh2en"], "zh2en", "wait", 2)
                    + online("on_ali_S2dl", "S2DLZ", WIN["on_ali"], "zh2en", "wait", 2)
                    + online("zh2en_S2dl_LA2_d1", "S2DLZ", WIN["zh2en"], "zh2en", "la2", 2)
                    + online("zh2en_S2dl_lag2", "S2DLZ", WIN["zh2en"], "zh2en", "lag", 2, ["--lag-ms", "2000"]), 4)
    jobs += jobs_of("dec1", "S2DLM", online("on_en2zh_S2dlm", "S2DLM", WIN["on_en2zh"], "en2zh", "wait", 2)
                    + online("en2zh_S2dlm_LA2_d1", "S2DLM", WIN["en2zh"], "en2zh", "la2", 2), 4)
    jobs += jobs_of("dec1", "O2N", online("en2zh_SC_seg", "O2N", f"{OUT}/en2zh_SC-segments/segments.jsonl", "en2zh", "la2", 2,
                                          ["--zero-sd"])
                    + online("zh2en_SC_seg", "O2N", f"{OUT}/zh2en_SC-segments/segments.jsonl", "zh2en", "la2", 2, ["--zero-sd"])
                    + offline("en2zh_OC_seg", "O2N", f"{OUT}/en2zh_OC-segments/segments.jsonl", "en2zh", 1, ["--zero-sd"])
                    + offline("zh2en_OC_seg", "O2N", f"{OUT}/zh2en_OC-segments/segments.jsonl", "zh2en", 1, ["--zero-sd"])
                    + offline("en2zh_CC_seg", "O2N", f"{OUT}/en2zh_CC-segments/segments.jsonl", "en2zh", 1, ["--zero-sd"])
                    + offline("zh2en_CC_seg", "O2N", f"{OUT}/zh2en_CC-segments/segments.jsonl", "zh2en", 1, ["--zero-sd"]), 4)
    # single-stream read timing (batch 1, one worker on the card): 40 windows each
    for th, name, win, d, pol in (("S2DL", "time_en2zh_S2dl", WIN["en2zh"], "en2zh", "wait"),
                                  ("O2", "time_en2zh_O2_LA2", WIN["en2zh"], "en2zh", "la2"),
                                  ("S2DLZ", "time_zh2en_S2dl", WIN["zh2en"], "zh2en", "wait")):
        jobs += jobs_of("dec1t", th, online(name, th, win, d, pol, 1, ["--time-reads", "--limit", "40"], batch=1), 1)
    return jobs


# measured with 4 workers: Thinker 59 GiB + ~5 GB per worker at batch 8
M6 = (5500, 66000)


def dec1b():
    jobs = []
    jobs += jobs_of("dec1b", "S2DLZ", online("on_zh2en_S2dl", "S2DLZ", WIN["on_zh2en"], "zh2en", "wait", 3)
                    + online("on_ali_S2dl", "S2DLZ", WIN["on_ali"], "zh2en", "wait", 3)
                    + online("zh2en_S2dl_LA2_d1", "S2DLZ", WIN["zh2en"], "zh2en", "la2", 3)
                    + online("zh2en_S2dl_lag2", "S2DLZ", WIN["zh2en"], "zh2en", "lag", 3, ["--lag-ms", "2000"]), 6, *M6)
    jobs += jobs_of("dec1b", "S2DLM", online("on_en2zh_S2dlm", "S2DLM", WIN["on_en2zh"], "en2zh", "wait", 3)
                    + online("en2zh_S2dlm_LA2_d1", "S2DLM", WIN["en2zh"], "en2zh", "la2", 3), 6, *M6)
    jobs += jobs_of("dec1b", "O2N", online("en2zh_SC_seg", "O2N", f"{OUT}/en2zh_SC-segments/segments.jsonl", "en2zh", "la2", 2,
                                           ["--zero-sd"])
                    + online("zh2en_SC_seg", "O2N", f"{OUT}/zh2en_SC-segments/segments.jsonl", "zh2en", "la2", 2, ["--zero-sd"])
                    + offline("en2zh_OC_seg", "O2N", f"{OUT}/en2zh_OC-segments/segments.jsonl", "en2zh", 2, ["--zero-sd"])
                    + offline("zh2en_OC_seg", "O2N", f"{OUT}/zh2en_OC-segments/segments.jsonl", "zh2en", 2, ["--zero-sd"])
                    + offline("en2zh_CC_seg", "O2N", f"{OUT}/en2zh_CC-segments/segments.jsonl", "en2zh", 2, ["--zero-sd"])
                    + offline("zh2en_CC_seg", "O2N", f"{OUT}/zh2en_CC-segments/segments.jsonl", "zh2en", 2, ["--zero-sd"]), 6, *M6)
    for th, name, win, d, pol in (("S2DL", "time_en2zh_S2dl", WIN["en2zh"], "en2zh", "wait"),
                                  ("O2", "time_en2zh_O2_LA2", WIN["en2zh"], "en2zh", "la2"),
                                  ("S2DLZ", "time_zh2en_S2dl", WIN["zh2en"], "zh2en", "wait")):
        jobs += jobs_of("dec1bt", th, online(name, th, win, d, pol, 1, ["--time-reads", "--limit", "40"], batch=1), 1)
    return jobs


def dec1c():
    jobs = []
    jobs += jobs_of("dec1c", "S2DLZ", online("on_zh2en_S2dl", "S2DLZ", WIN["on_zh2en"], "zh2en", "wait", 3)
                    + online("on_ali_S2dl", "S2DLZ", WIN["on_ali"], "zh2en", "wait", 3)
                    + online("zh2en_S2dl_LA2_d1", "S2DLZ", WIN["zh2en"], "zh2en", "la2", 3)
                    + online("zh2en_S2dl_lag2", "S2DLZ", WIN["zh2en"], "zh2en", "lag", 3, ["--lag-ms", "2000"]), 6, *M6)
    jobs += jobs_of("dec1c", "S2DLM", online("on_en2zh_S2dlm", "S2DLM", WIN["on_en2zh"], "en2zh", "wait", 5), 6, *M6)
    jobs += jobs_of("dec1c", "O2N", online("en2zh_SC_seg", "O2N", f"{OUT}/en2zh_SC-segments/segments.jsonl", "en2zh", "la2", 2,
                                           ["--zero-sd"])
                    + online("zh2en_SC_seg", "O2N", f"{OUT}/zh2en_SC-segments/segments.jsonl", "zh2en", "la2", 2, ["--zero-sd"])
                    + offline("en2zh_OC_seg", "O2N", f"{OUT}/en2zh_OC-segments/segments.jsonl", "en2zh", 2, ["--zero-sd"])
                    + offline("zh2en_OC_seg", "O2N", f"{OUT}/zh2en_OC-segments/segments.jsonl", "zh2en", 2, ["--zero-sd"])
                    + offline("en2zh_CC_seg", "O2N", f"{OUT}/en2zh_CC-segments/segments.jsonl", "en2zh", 2, ["--zero-sd"])
                    + offline("zh2en_CC_seg", "O2N", f"{OUT}/zh2en_CC-segments/segments.jsonl", "zh2en", 2, ["--zero-sd"]), 6, *M6)
    for th, name, win, d, pol in (("S2DL", "time_en2zh_S2dl", WIN["en2zh"], "en2zh", "wait"),
                                  ("O2", "time_en2zh_O2_LA2", WIN["en2zh"], "en2zh", "la2"),
                                  ("S2DLZ", "time_zh2en_S2dl", WIN["zh2en"], "zh2en", "wait")):
        jobs += jobs_of("dec1ct", th, online(name, th, win, d, pol, 1, ["--time-reads", "--limit", "40"], batch=1), 1)
    return jobs


T.update({"ZCHAR": f"{EV}/thinker_gstar_s2dlchar_zh2en_3k_20261031/thinker_merged",
          "SEED1": f"{EV}/thinker_gstar_s2dl_seed20261101/thinker_merged",
          "SEED2": f"{EV}/thinker_gstar_s2dl_seed20261102/thinker_merged",
          "DLABC": f"{EV}/thinker_gstar_s2dlabc_20261031/thinker_merged",
          "CTURN": f"{EV}/thinker_gstar_cturn_20261031/thinker_merged",
          "O2S1": f"{EV}/thinker_gstar_o2_seed20261101/thinker_merged",
          "O2S2": f"{EV}/thinker_gstar_o2_seed20261102/thinker_merged"})
DEV800 = "data/v6_sets/zh2en_dev800.jsonl"
V3SEG = f"{EV}/gstar_v3_20261029"
ALI = "data/alimeeting_si_v1/windows_test_ref.jsonl"


def dec2(what):
    j = []
    if what == "zhchar":
        j += jobs_of(f"dec2_{what}", "S2DLZ", online("dev_zh2en_S2dl3k", "S2DLZ", DEV800, "zh2en", "wait", 4), 6, *M6)
        j += jobs_of(f"dec2_{what}", "ZCHAR", online("dev_zh2en_S2dlchar", "ZCHAR", DEV800, "zh2en", "wait", 2)
                     + online("zh2en_S2dlchar_d1", "ZCHAR", WIN["zh2en"], "zh2en", "wait", 4)
                     + online("ali_S2dlchar_d1", "ZCHAR", ALI, "zh2en", "wait", 6), 6, *M6)
    elif what == "seeds":
        for k in ("SEED1", "SEED2"):
            j += jobs_of(f"dec2_{what}", k, online(f"en2zh_S2dl_{k.lower()}_d1", k, WIN["en2zh"], "en2zh", "wait", 6), 6, *M6)
    elif what == "dlabc":
        j += jobs_of(f"dec2_{what}", "DLABC", online("en2zh_S2dlabc_d1", "DLABC", WIN["en2zh"], "en2zh", "wait", 5)
                     + online("zh2en_S2dlabc_d1", "DLABC", WIN["zh2en"], "zh2en", "wait", 7), 6, *M6)
    elif what == "cturn":
        j += jobs_of(f"dec2_{what}", "CTURN", offline("en2zh_C-turn_seg", "CTURN", f"{V3SEG}/segments_en2zh/segments.jsonl", "en2zh", 3,
                                              ["--zero-sd"])
                     + offline("zh2en_C-turn_seg", "CTURN", f"{V3SEG}/segments_zh2en/segments.jsonl", "zh2en", 3, ["--zero-sd"])
                     + offline("ali_C-turn_seg", "CTURN", f"{V3SEG}/segments_ali/segments.jsonl", "zh2en", 6, ["--zero-sd"]), 6, *M6)
    elif what == "o2seeds":
        for k in ("O2S1", "O2S2"):
            j += jobs_of(f"dec2_{what}", k, offline(f"en2zh_O2_{k.lower()}", k, f"{D}/e2e_gstar_v1/windows_test_ref.jsonl", "en2zh", 3)
                         + offline(f"zh2en_O2_{k.lower()}", k, WIN["zh2en"], "zh2en", 3), 6, *M6)
    elif what == "t13":   # the same 96 Zh->En windows at batch 16 and at batch 4 (one worker each)
        j += jobs_of(f"dec2_{what}", "S2DLZ", online("t13_zh2en_b16", "S2DLZ", WIN["zh2en"], "zh2en", "wait", 1, ["--limit", "96"], batch=16)
                     + online("t13_zh2en_b4", "S2DLZ", WIN["zh2en"], "zh2en", "wait", 1, ["--limit", "96"], batch=4), 2)
    for x in j:
        x["id"] = f"{x['id']}_{what}"
    return j


if __name__ == "__main__":
    if sys.argv[1].startswith("dec2:"):
        print(json.dumps(dec2(sys.argv[1].split(":", 1)[1]), indent=1))
    else:
        print(json.dumps({"dec1": dec1, "dec1b": dec1b, "dec1c": dec1c}[sys.argv[1]](), indent=1))
