from experiments.uniss_phasea_stateful_longepisode_rl_v1.evaluation.write_stage_report import _f


def test_report_number_formatting_handles_missing_values():
    assert _f(None) == "—"
    assert _f(1.2345, 2) == "1.23"

