import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location('s5_bench', Path(__file__).parents[1] / 's5_streaming_bench.py')
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


def test_probe_balances_languages_and_covers_extremes():
    rows = [dict(id=f'{g}-{i}', dataset=g, source_sec=i + 1) for g in ('a', 'b', 'c') for i in range(100)]
    picked = bench.probe_rows(rows)
    assert len(picked) == len({r['id'] for r in picked}) == 40
    for group in ('a', 'b', 'c'):
        lengths = [r['source_sec'] for r in picked if r['dataset'] == group]
        assert min(lengths) == 1 and max(lengths) == 100


def test_actual_pcm_timing_and_offsets_are_not_reconstructed_from_tokens():
    rows = [dict(id='x', status='ok', source_sec=10)]
    events = [dict(id='x', type='pcm', sample_offset=0, num_samples=24000,
                   elapsed_sec=2, source_sec=1),
              dict(id='x', type='pcm', sample_offset=24000, num_samples=24000,
                   elapsed_sec=3.5, source_sec=2)]
    report = bench.analyze(rows, events)
    assert report['first_pcm_wall_sec']['p50'] == 2
    assert report['server_playback_proxy_gap_total_sec'] == .5
    assert report['issues'] == []
    events[1]['sample_offset'] = 24001
    assert 'discontinuous' in bench.analyze(rows, events)['issues'][0]


def test_failure_and_missing_events_remain_visible():
    report = bench.analyze([dict(id='bad', status='failed'), dict(id='empty', status='ok', source_sec=1)], [])
    assert report['n_results'] == 2 and report['n_ok'] == 1
    assert len(report['issues']) == 2 and report['pcm_samples'] == 0
