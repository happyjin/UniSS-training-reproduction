import json
from pathlib import Path


def test_smoke_result_preserves_acknowledged_audio_contract():
    path = Path(
        "eval_outputs/uniss_phasea_stateful_longepisode_rl_v1/"
        "smoke_runtime_v2_attempt2/results.json"
    )
    if not path.is_file():
        return
    result = json.loads(path.read_text(encoding="utf-8"))["results"][0]
    assert result["continuous_audio_health"]["healthy"]
    assert result["tts_pending_unspoken_items"] == 0
    assert result["tts_failures"] == 0
    assert result["generated_streaming_transcription"]
    assert result["generated_streaming_translation"]

