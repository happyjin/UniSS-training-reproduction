"""Write one worker's X-Translator config.json: the released config.json (dc6a59a) with only the service URLs (ports),
the direction's source/target language, the run's storage dirs, the local CAM++ copy (same weights as
iic/speech_campplus_sv_zh-cn_16k-common) and the registered point's knob changed. Refuses to overwrite.
"""
import argparse
import json
from pathlib import Path

XT_CONFIG = "/opt/dlami/nvme/jasonleeeli/research_sources/baselines_20261112/X-Translator/config.json"
CAMPPLUS = "/opt/dlami/nvme/jasonleeeli/models/xtranslator/campplus"
POINTS = {"xt-default": {}, "xt-c1280": {"paraformer_chunk_size": 1280}, "xt-c5120": {"paraformer_chunk_size": 5120},
          "xt-a1.6": {}}   # xt-a1.6 changes the ASR server (serve_qwen3asr.py --chunk-size-sec 1.6), not this file
DIRS = {"en2zh": ("en", "zh"), "zh2en": ("zh", "en")}

ap = argparse.ArgumentParser()
ap.add_argument("--point", choices=sorted(POINTS), required=True)
ap.add_argument("--direction", choices=sorted(DIRS), required=True)
ap.add_argument("--asr-port", type=int, required=True)
ap.add_argument("--mt-port", type=int, required=True)
ap.add_argument("--tts-port", type=int, required=True)
ap.add_argument("--run-dir", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()

c = json.load(open(XT_CONFIG, encoding="utf-8"))
c["asr"]["qwen3_asr_url"] = f"http://127.0.0.1:{a.asr_port}/v1/recognize"
c["translation"]["lmt_url"] = f"http://127.0.0.1:{a.mt_port}/v1/chat/completions"
c["tts"]["xvoice_tts_url"] = f"http://127.0.0.1:{a.tts_port}/synthesize"
c["translation"]["source_lang"], c["translation"]["target_lang"] = DIRS[a.direction]
c["tts"]["speaker_embedding_model"] = CAMPPLUS
run = Path(a.run_dir).resolve()
c["storage"] = {"session_audio_dir": str(run / "sessions"), "log_dir": str(run / "logs")}
for k, v in POINTS[a.point].items():
    c["asr"][k] = v
with open(a.out, "x", encoding="utf-8") as fh:
    json.dump(c, fh, ensure_ascii=False, indent=2)
print(a.out)
