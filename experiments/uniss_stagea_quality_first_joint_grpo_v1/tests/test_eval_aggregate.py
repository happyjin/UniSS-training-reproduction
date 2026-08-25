from experiments.uniss_stagea_quality_first_joint_grpo_v1.evaluation.aggregate import (
    _latency,
)


def test_latency_uses_emission_timestamps_and_actions():
    samples = [
        {
            "source_duration_ms": 1000,
            "tgt_lang": "eng",
            "e_s2s_free": {
                "events": [
                    {
                        "source_end_ms": 320,
                        "compute_ms": 10,
                        "chosen_continuations": ["WAIT"],
                        "mt_deltas": [],
                        "semantic_tokens": 0,
                    },
                    {
                        "source_end_ms": 640,
                        "compute_ms": 20,
                        "chosen_continuations": ["WRITE_MT", "WRITE_SEMANTIC"],
                        "mt_deltas": ["hello world"],
                        "semantic_tokens": 8,
                    },
                ]
            },
        }
    ]
    value = _latency(samples, "e_s2s_free")
    assert value["first_text_write_ms"]["p50"] == 640
    assert value["first_semantic_write_ms"]["p50"] == 640
    assert value["average_proportion"] == 0.64
    assert value["event_compute_ms"]["mean"] == 15
    assert value["generation_rtf"] == 0.03
    assert value["actions"] == {
        "wait": 1,
        "write_asr": 0,
        "write_mt": 1,
        "write_semantic": 1,
    }

