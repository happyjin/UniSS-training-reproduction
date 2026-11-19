"""Compare eager/static/graph using the same loaded model and ordered probe rows."""
import contextlib
import json
import os
import re
from pathlib import Path
import sys
import time

import omni_shared_run as shared


def main():
    argv = sys.argv[:]
    out_index = argv.index('--out-dir') + 1
    root = Path(argv[out_index])
    root.mkdir(parents=True, exist_ok=True)
    modes = os.environ.get('SIMULS2ST_PROBE_MODES', 'eager,static,graph').split(',')
    reference = None
    results = []
    # Third-party loggers may bind their StreamHandler to the current stderr on
    # first import. Keep mode files open so later modes never log to a closed file.
    capture_logs = []
    for mode in modes:
        if mode not in {'eager', 'static', 'graph'}:
            raise ValueError(mode)
        os.environ['SIMULS2ST_TALKER_ACCEL'] = mode
        sys.argv = argv[:]
        sys.argv[out_index] = str(root / mode)
        started = time.perf_counter()
        shared.START = started
        log = (root / f'{mode}.log').open('w')
        capture_logs.append(log)
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            shared.main()
        log.flush()
        elapsed = time.perf_counter() - started
        log_text = (root / f'{mode}.log').read_text()
        codes = [line for line in log_text.splitlines()
                 if line.startswith('[talker-codes]')]
        texts = {}
        for run in (root / mode).glob('run_*'):
            ids = (run / 'ids.txt').read_text().splitlines()
            predictions = [json.loads(l) for l in (run / 'text_predictions.jsonl').read_text().splitlines() if l.strip()]
            if len(ids) != len(predictions):
                raise RuntimeError(f'{run}: incomplete sidecar')
            for uid, row in zip(ids, predictions):
                texts[uid] = row.get('pred_text', '')
        if not codes or not texts or any(not text for text in texts.values()):
            raise RuntimeError(f'{mode}: missing codes/text')
        if reference is None:
            reference = (codes, texts)
        evaluation_sec = sum(float(t) for t in re.findall(
            r'\[shared-timing\] \w+\.evaluate: ([\d.]+)s', log_text))
        result = dict(mode=mode, elapsed_sec=elapsed, evaluation_sec=evaluation_sec,
                      compared_to=modes[0],
                      code_chunks=len(codes),
                      different_chunks=sum(a != b for a, b in zip(reference[0], codes)),
                      codes_equal=reference[0] == codes, texts_equal=reference[1] == texts)
        results.append(result)
        (root / 'probe_results.json').write_text(json.dumps(results, indent=2))
        print(f'[probe-result] {json.dumps(result)}', flush=True)
        # Stop before graph if static already changes semantic codes. Inspect the
        # first divergence rather than silently accepting a different trajectory.
        if not result['codes_equal'] or not result['texts_equal']:
            raise RuntimeError(f'{mode}: output differs from {modes[0]}; inspect probe logs')


if __name__ == '__main__':
    main()
