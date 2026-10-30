"""A.PCP and SIM-O for the offline CVSS-T speech of one run (paper v7 Table 1, SimulS2ST-Omni's columns).

  A.PCP   AutoPCP multilingual v2 on wav2vec2-XLSR layer 9, symmetrised (evaluation.autopcp_metrics), source speech
          against the output speech, as everywhere in this project.
  SIM-O   WavLM speaker-verification cosine between the output speech and the paired real reference target speech
          (experiments.uniss_omni_s2st_v1.evaluation.sim_o). En->Zh: the Chinese Common Voice recording; Zh->En: the
          CVSS-T voice-transferred English target. This box has wavlm-base-plus-sv only, not the paper's WavLM-Large,
          so these values are on another scale and are reported apart from the quoted column.

    PYTHONPATH=. uniss-eval/bin/python reports/uniss_si_v4/cvss_table1_metrics.py <speech run dir> <out dir> <gpu>
"""
import json
import os
import subprocess
import sys
from pathlib import Path

MODELS = "/opt/dlami/nvme/jasonleeeli/evaluation_models"
CVSS = "/opt/dlami/nvme/jasonleeeli/CVSS/canonical_16k/cvss_t_zh_en_test"


def main():
    run, out, gpu = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    out.mkdir(parents=True, exist_ok=True)
    rows = out / "rows.jsonl"
    if not rows.exists():
        recs = [json.loads(l) for l in open(run / "results.jsonl", encoding="utf-8")]
        with open(rows, "x", encoding="utf-8") as fh:
            for r in recs:
                if r.get("error") or not r.get("speech_path"):
                    continue
                src = r["source_audio_path"]
                ref = f"{CVSS}/{'source_zh' if r['tgt_lang'] == 'cmn' else 'target_en'}/{r['id']}.wav"
                assert os.path.exists(ref), ref
                fh.write(json.dumps({"id": r["id"], "mode": "si", "src_lang": r["src_lang"], "tgt_lang": r["tgt_lang"],
                                     "audio_path": r["speech_path"], "source_audio_path": src,
                                     "reference_audio_path": ref}, ensure_ascii=False) + "\n")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, PYTHONPATH=".")
    if not (out / "apcp").exists():
        subprocess.run([sys.executable, "-m", "evaluation.autopcp_metrics", "--input", str(rows), "--output-dir",
                        str(out / "apcp"), "--comparator-path", f"{MODELS}/AutoPCP-multilingual-v2", "--encoder-model",
                        f"{MODELS}/wav2vec2-large-xlsr-53", "--device", "cuda:0", "--pick-layer", "9", "--symmetrize",
                        "--batch-size", "16"], check=True, env=env)
    if not (out / "sim_o_baseplus.json").exists():
        subprocess.run([sys.executable, "-m", "experiments.uniss_omni_s2st_v1.evaluation.sim_o", "--input", str(rows),
                        "--output", str(out / "sim_o_baseplus.json"), "--model", f"{MODELS}/wavlm-base-plus-sv",
                        "--device", "cuda:0"], check=True, env=env)
    print("CVSS_TABLE1_METRICS_DONE", out, flush=True)


if __name__ == "__main__":
    main()
