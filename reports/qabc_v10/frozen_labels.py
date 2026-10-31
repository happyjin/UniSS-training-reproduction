"""Whole-sentence continuations of the frozen Thinker for every read of si_qabc_rules dumps (review M6: do the stable
lengths the adapter learned from, which come from the frozen Thinker, differ from those of the adapted Thinker on the
reads the policy makes?). One engine for all dump files; one output file per dump file, same name, in --out-dir.

    python reports/qabc_v10/frozen_labels.py --out-dir <dir> --tp 2 <dump.jsonl> ...
"""
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.uniss_qwen3_omni_v1.evaluation.si_qabc_rules import load_items, make_head  # noqa: E402
from experiments.uniss_qwen3_omni_v1.evaluation.si_thinker_stream import QWEN3, THINKER_EOS, WAIT_MARK, units  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("dumps", nargs="+")
ap.add_argument("--out-dir", required=True)
ap.add_argument("--items-en2zh", default="")
ap.add_argument("--items-zh2en", default="")
ap.add_argument("--tp", type=int, default=2)
ap.add_argument("--gpu-mem", type=float, default=0.3)
a = ap.parse_args()
out_dir = Path(a.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)
todo = [Path(d) for d in a.dumps if not (out_dir / Path(d).name).exists()]
from vllm import LLM, SamplingParams  # noqa: E402
llm = LLM(model=QWEN3, gpu_memory_utilization=a.gpu_mem, max_model_len=8192, limit_mm_per_prompt={"audio": 1},
          seed=20261021, tensor_parallel_size=a.tp)
sp = SamplingParams(temperature=0.0, max_tokens=160, stop_token_ids=[THINKER_EOS], logit_bias={WAIT_MARK: -100.0})
cache = {}
for f in todo:
    d = "en2zh" if "_en2zh_" in f.name else "zh2en"
    if d not in cache:
        items = getattr(a, f"items_{d}")
        cache[d] = ({it["id"]: it for it in load_items(d, items, 0)}, make_head(d))
    items, head = cache[d]
    recs = [json.loads(l) for l in open(f, encoding="utf-8")]
    outs = llm.generate([{"prompt": head + r["committed"], "multi_modal_data": {"audio": (items[r["id"]]["audio"], 16000)}}
                         for r in recs], sp, use_tqdm=False)
    with open(out_dir / f.name, "x", encoding="utf-8") as fh:
        for r, o in zip(recs, outs):
            fh.write(json.dumps({"id": r["id"], "t": r["t"], "full_units": units(o.outputs[0].text, d.endswith("zh"))},
                                ensure_ascii=False) + "\n")
    print(f"{f.name}: {len(recs)} reads", flush=True)
