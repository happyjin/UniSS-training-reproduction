"""Paper v3 evaluation (plan v9 §6): both directions, on the 2,375 en->zh and 3,317 zh->en AMI test windows.

  offline   prompted Thinker, cascades (original Thinker; O2 without speaker stream), O2 without speaker stream, O2,
            O2 with the oracle speaker stream
  simul     prompted Thinker + LA2 (zh->en: si_thinker_stream, as en->zh's existing B-LA2), O2 + LA2 (1, 2 s),
            S2 (ABC) with bias b0 in {-1, 0, 1} at 1 s, S2 at 0.5 / 2 s (b0 0), S2 + SDR + FR (b0 -1, 0, 1), S2 + FR (b0 0)
  ablation  en->zh: O2 continued on v2's deadline labels (S2-dl), the same base as S2
  real      AliMeeting test (20 real Mandarin meetings, headsets summed; build_alimeeting_test.py), Zh->En: Nemotron
            posteriors (both presets), Hy-MT2 references, audit, windows, then prompted, same-data cascade, O2 without
            speaker stream, O2, O2 + LA2, S2 and S2 + SDR + FR. Queued first, so a failure shows early
Each system: 8 decode shards, then hyp.jsonl and score.json (+ analysis.json). en->zh systems that exist already
(prompted, LA2 on the prompted Thinker, original-Thinker cascade) are not rerun.

    python reports/gstar_si_paper_v1/make_v3_jobs.py > reports/gstar_si_paper_v1/jobs/v3_jobs.json
"""
import json

EV = "eval_outputs/uniss_qwen3_omni_v1"
OUT = f"{EV}/gstar_v3_20261029"
D = "data/ami_multispk_si_v1"
PYE = "/opt/dlami/nvme/jasonleeeli/conda_envs/uniss-eval/bin/python"
PYM = "/opt/dlami/nvme/jasonleeeli/conda_envs/qwen3omni/bin/python"
PYV = "/opt/dlami/nvme/jasonleeeli/venvs/vllm_omni_030/bin/python"
TS = "20261029b"
O2 = f"{EV}/thinker_gstar_o2_{TS}/thinker_merged"
S2 = f"{EV}/thinker_gstar_s2abc_{TS}/thinker_merged"
O2N = f"{EV}/thinker_gstar_o2nosd_{TS}/thinker_merged"
S2DL = f"{EV}/thinker_gstar_s2dl_{TS}/thinker_merged"   # label ablation: deadline labels on the same O2 base (en->zh)
REL = "pretrained_models/Qwen3-Omni-30B-A3B-Instruct"
WIN = {"en2zh": {"off": f"{D}/e2e_gstar_v1/windows_test_ref.jsonl", "str": f"{D}/e2e_gstar_stream_v1/windows_test_ref.jsonl"},
       "zh2en": {"off": f"{D}/e2e_gstar_zh2en_v1/windows_test_ref.jsonl",
                 "str": f"{D}/e2e_gstar_zh2en_v1/windows_test_ref.jsonl"}}
M = "experiments.uniss_qwen3_omni_v1.evaluation"
DC = "experiments.uniss_qwen3_omni_v1.data_clean"
PYN = "/opt/dlami/nvme/jasonleeeli/venvs/nemo_diar/bin/python"
ALI = "data/alimeeting_si_v1"
WIN["ali"] = {"off": f"{ALI}/windows_test_ref.jsonl", "str": f"{ALI}/windows_test_ref.jsonl"}
N = 8
jobs = []


def decode(name, mod, thinker, win, extra, post_key, deps=(), analyze=True):
    d = f"{OUT}/{name}"
    for k in range(N):
        jobs.append({"id": f"{name}_dec{k}", "gpus": 1, "vllm": False, "deps": list(deps),
                     "cmd": f"mkdir -p {d} && CUDA_VISIBLE_DEVICES=$GPU0 PYTHONPATH=.:third_party/Megatron-LM {PYM} -m "
                            f"{M}.{mod} --thinker {thinker} --windows {win} --out {d}/hyp_s{k}.jsonl --shard {k} "
                            f"--num-shards {N} {extra} > {d}/infer_s{k}.log 2>&1"})
    cmd = (f"set -e; cat {d}/hyp_s*.jsonl > {d}/hyp.jsonl; PYTHONPATH=. {PYE} -m {M}.gstar_score --windows {win} "
           f"--hyp {d}/hyp.jsonl --post-key {post_key} --out {d}/score.json > {d}/score.log 2>&1")
    if analyze:
        cmd += (f"; PYTHONPATH=. {PYE} -m {M}.gstar_analyze --windows {win} --hyp {d}/hyp.jsonl "
                f"--out {d}/analysis.json > /dev/null 2>&1 || true")
    jobs.append({"id": f"{name}_score", "gpus": 1, "vllm": False, "share": True, "need_mib": 0,
                 "deps": [f"{name}_dec{k}" for k in range(N)], "cmd": cmd})


def cascade(dr, name, thinker, extra, deps=()):
    win = WIN[dr]["off"]
    seg = f"{OUT}/segments_{dr}"
    if not any(j["id"] == f"seg_{dr}" for j in jobs):
        jobs.append({"id": f"seg_{dr}", "gpus": 1, "vllm": False, "share": True, "need_mib": 0, "deps": list(deps),
                     "cmd": f"PYTHONPATH=. {PYE} -m {M}.gstar_cascade segment --windows {win} --out {seg}"})
    sd = "zh2en" if dr == "ali" else dr
    decode(f"{name}_seg", "gstar_si_stream", thinker, f"{seg}/segments.jsonl", f"{extra} --dir {sd}", "offline",
           deps=[f"seg_{dr}"], analyze=False)
    d = f"{OUT}/{name}"
    jobs[-1]["cmd"] = jobs[-1]["cmd"].split("; PYTHONPATH")[0]   # the segment run only concatenates
    jobs.append({"id": f"{name}_score", "gpus": 1, "vllm": False, "share": True, "need_mib": 0,
                 "deps": [f"{name}_seg_score"],
                 "cmd": f"set -e; mkdir -p {d}; PYTHONPATH=. {PYE} -m {M}.gstar_cascade assemble --seg {seg}/segments.jsonl "
                        f"--hyp {OUT}/{name}_seg/hyp.jsonl --windows {win} --out {d}/hyp.jsonl; PYTHONPATH=. {PYE} -m "
                        f"{M}.gstar_score --windows {win} --hyp {d}/hyp.jsonl --post-key offline --out {d}/score.json "
                        f"> {d}/score.log 2>&1"})


# real Mandarin meetings (AliMeeting test), Zh->En
for p in ("offline", "lowlat"):
    jobs.append({"id": f"ali_nemo_{p}", "gpus": 1, "vllm": False, "deps": [],
                 "cmd": f"CUDA_VISIBLE_DEVICES=$GPU0 {PYN} experiments/uniss_qwen3_omni_v1/data_clean/dump_nemotron_posteriors.py "
                        f"--preset {p} --data {ALI} --audio near_mix --out-name nemotron_{p} > {ALI}/nemo_{p}.log 2>&1"})
jobs.append({"id": "ali_mt", "gpus": 1, "vllm": True, "deps": [],
             "cmd": f"CUDA_VISIBLE_DEVICES=$GPU0 VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_CACHE_ROOT=/opt/dlami/nvme/jasonleeeli/cache/vllm_omni_030 "
                    f"PYTHONPATH=. {PYV} -m {DC}.translate_zh2en_refs --items {ALI}/mt_items.jsonl --out {ALI}/refs.jsonl "
                    f"> {ALI}/mt.log 2>&1"})
jobs.append({"id": "ali_windows", "gpus": 1, "vllm": False, "share": True, "need_mib": 0,
             "deps": ["ali_nemo_offline", "ali_nemo_lowlat", "ali_mt"],
             "cmd": f"set -e; PYTHONPATH=. {PYE} -m {DC}.build_alimeeting_test audit --out {ALI} > {ALI}/audit.log 2>&1; "
                    f"PYTHONPATH=. {PYE} -m {DC}.build_alimeeting_test windows --out {ALI} > {ALI}/windows.log 2>&1"})
AW = ["ali_windows"]
decode("ali_B-off", "gstar_si_stream", REL, WIN["ali"]["off"], "--inst sd --dir zh2en", "offline", deps=AW, analyze=False)
cascade("ali", "ali_C-ft2", O2N, "--inst gstar --post-key offline --zero-sd", deps=AW)
decode("ali_O2-nosd", "gstar_si_stream", O2N, WIN["ali"]["off"], "--inst gstar --post-key offline --zero-sd --dir zh2en", "offline", deps=AW)
decode("ali_O2", "gstar_si_stream", O2, WIN["ali"]["off"], "--inst gstar --post-key offline --dir zh2en", "offline", deps=AW)
decode("ali_O2_LA2_d1", "gstar_si_stream_online", O2, WIN["ali"]["str"], "--policy la2 --window 1 --post-key lowlat --dir zh2en", "lowlat", deps=AW)
decode("ali_S2_b0_d1", "gstar_si_stream_online", S2, WIN["ali"]["str"], "--policy abc --bias 0 --window 1 --post-key lowlat --dir zh2en", "lowlat", deps=AW)
decode("ali_S2_sdrfr_b0_d1", "gstar_si_stream_online", S2, WIN["ali"]["str"],
       "--policy abc --bias 0 --sdr --fr --window 1 --post-key lowlat --dir zh2en", "lowlat", deps=AW)
for dr in ("en2zh", "zh2en"):
    off, st = WIN[dr]["off"], WIN[dr]["str"]
    decode(f"{dr}_O2", "gstar_si_stream", O2, off, f"--inst gstar --post-key offline --dir {dr}", "offline")
for dr in ("en2zh", "zh2en"):
    st = WIN[dr]["str"]
    decode(f"{dr}_S2_b0_d1", "gstar_si_stream_online", S2, st, f"--policy abc --bias 0 --window 1 --post-key lowlat --dir {dr}", "lowlat")
    decode(f"{dr}_S2_sdrfr_b0_d1", "gstar_si_stream_online", S2, st,
           f"--policy abc --bias 0 --sdr --fr --window 1 --post-key lowlat --dir {dr}", "lowlat")
    decode(f"{dr}_O2_LA2_d1", "gstar_si_stream_online", O2, st, f"--policy la2 --window 1 --post-key lowlat --dir {dr}", "lowlat")
for dr in ("en2zh", "zh2en"):
    off = WIN[dr]["off"]
    cascade(dr, f"{dr}_C-ft2", O2N, "--inst gstar --post-key offline --zero-sd")
    decode(f"{dr}_O2-nosd", "gstar_si_stream", O2N, off, f"--inst gstar --post-key offline --zero-sd --dir {dr}", "offline")
for dr in ("en2zh", "zh2en"):
    st, off = WIN[dr]["str"], WIN[dr]["off"]
    for b in ("-1", "1"):
        decode(f"{dr}_S2_b{b}_d1", "gstar_si_stream_online", S2, st,
               f"--policy abc --bias {b} --window 1 --post-key lowlat --dir {dr}", "lowlat")
        decode(f"{dr}_S2_sdrfr_b{b}_d1", "gstar_si_stream_online", S2, st,
               f"--policy abc --bias {b} --sdr --fr --window 1 --post-key lowlat --dir {dr}", "lowlat")
    decode(f"{dr}_S2_fr_b0_d1", "gstar_si_stream_online", S2, st, f"--policy abc --bias 0 --fr --window 1 --post-key lowlat --dir {dr}", "lowlat")
    for w in ("0.5", "2"):
        decode(f"{dr}_S2_b0_d{w}", "gstar_si_stream_online", S2, st, f"--policy abc --bias 0 --window {w} --post-key lowlat --dir {dr}", "lowlat")
    decode(f"{dr}_O2_LA2_d2", "gstar_si_stream_online", O2, st, f"--policy la2 --window 2 --post-key lowlat --dir {dr}", "lowlat")
    decode(f"{dr}_O2-oracle", "gstar_si_stream", O2, off, f"--inst gstar --post-key oracle_offline --dir {dr}", "offline")
# label ablation on the same base: deadline labels (v2's MIST-Simul recipe) vs stable-prefix labels, en->zh
decode("en2zh_S2dl_d1", "gstar_si_stream_online", S2DL, WIN["en2zh"]["str"], "--policy wait --window 1 --post-key lowlat --dir en2zh", "lowlat")
# zh->en baselines that en->zh already has
decode("zh2en_B-off", "gstar_si_stream", REL, WIN["zh2en"]["off"], "--inst sd --dir zh2en", "offline", analyze=False)
cascade("zh2en", "zh2en_C-van", REL, "--inst sd")
d = f"{OUT}/zh2en_B-LA2_d1"
win = WIN["zh2en"]["str"]
for k in range(N):
    jobs.append({"id": f"zh2en_B-LA2_d1_dec{k}", "gpus": 1, "vllm": True, "deps": [],
                 "cmd": f"mkdir -p {d} && {PYE} -c \"import json,sys;L=[json.dumps({{'id':w['id'],'audio':w['audio'],"
                        f"'dur_ms':w['dur_ms']}}) for w in map(json.loads, open('{win}'))];"
                        f"sys.stdout.write(chr(10).join(L[{k}::{N}])+chr(10))\" > {d}/items_s{k}.jsonl && "
                        f"CUDA_VISIBLE_DEVICES=$GPU0 VLLM_USE_FLASHINFER_SAMPLER=0 "
                        f"VLLM_CACHE_ROOT=/opt/dlami/nvme/jasonleeeli/cache/vllm_omni_030 PYTHONPATH=. {PYV} -m "
                        f"{M}.si_thinker_stream --direction zh2en --items-file {d}/items_s{k}.jsonl --windows 1 "
                        f"--policy la2 --thinker {REL} --out {d}/s{k} > {d}/infer_s{k}.log 2>&1"})
jobs.append({"id": "zh2en_B-LA2_d1_score", "gpus": 1, "vllm": False, "share": True, "need_mib": 0,
             "deps": [f"zh2en_B-LA2_d1_dec{k}" for k in range(N)],
             "cmd": f"set -e; {PYE} -c \"\nimport json, glob\nwith open('{d}/hyp.jsonl', 'w') as fh:\n"
                    f"    for f in sorted(glob.glob('{d}/s*/blocks_zh2en_w1.jsonl')):\n"
                    f"        for r in map(json.loads, open(f)):\n"
                    f"            bl = [b for b in r['blocks'] if b['text'].strip()]\n"
                    f"            fh.write(json.dumps({{'id': r['id'], 'segments': [{{'spk': 1, 'start_s': 0.0, 'end_s': 0.1, "
                    f"'text': r['translation']}}] if r['translation'].strip() else [], 'blocks': [{{'write_ms': "
                    f"b['write_ms'], 'spk': 1, 'start_s': 0.0, 'text': b['text']}} for b in bl]}}, ensure_ascii=False) + chr(10))\n\"; "
                    f"PYTHONPATH=. {PYE} -m {M}.gstar_score --windows {win} --hyp {d}/hyp.jsonl --post-key lowlat "
                    f"--out {d}/score.json > {d}/score.log 2>&1"})
print(json.dumps(jobs, indent=1))
