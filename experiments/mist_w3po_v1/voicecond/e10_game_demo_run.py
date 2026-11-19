"""E10 (LiveInterpret S5 multi-speaker, Thinker fine-tuned, Talker frozen) on a few esports windows, for a listening demo
only (user 2026-11-19: "确认目前这个LI的thinker-talker模型多说话人目前是能正常发声 … 做一个demo放几个电竞试听音频").

Wraps third_party/liveinterpret_training_pipelines/e10_multispk/run_e10.py unchanged: it only adds two esports sets
(same fields as its AMI / AliMeeting sets) and then calls run_e10.main() with the given argv. Esports data is eval/demo only.

    cd third_party/liveinterpret_training_pipelines && \
    /opt/dlami/nvme/jasonleeeli/venvs/simuls2s/bin/python /opt/dlami/nvme/jasonleeeli/projects/UniSS/experiments/mist_w3po_v1/voicecond/e10_game_demo_run.py \
        --mode e10 --set game_zh2en --windows <windows jsonl> --ids a,b,c --adapter <checkpoint> --out <dir>
"""
import sys
from pathlib import Path

E10 = Path("/opt/dlami/nvme/jasonleeeli/projects/UniSS/third_party/liveinterpret_training_pipelines/e10_multispk")
sys.path.insert(0, str(E10))
import run_e10  # noqa: E402

run_e10.SETS.update({
    "game_zh2en": dict(tgt="en", data="data/game_esports_si_v1/zh2en", sess="mix", diar="nemotron_lowlat"),
    "game_en2zh": dict(tgt="zh", data="data/game_esports_si_v1/en2zh", sess="ihm_mix", diar="nemotron_lowlat"),
})

if __name__ == "__main__":
    run_e10.main()
