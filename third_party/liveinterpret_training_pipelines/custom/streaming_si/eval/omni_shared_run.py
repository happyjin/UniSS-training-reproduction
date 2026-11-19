"""Run both SimulEval directions sequentially with one loaded Omni inference object."""
import argparse
import gc
import inspect
import json
import logging
from pathlib import Path
import sys
import time

START = time.perf_counter()
logging.basicConfig(
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO, stream=sys.stderr,
)


def mark(stage, start=START):
    print(f'[shared-timing] {stage}: {time.perf_counter() - start:.3f}s', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--checkpoint-path', required=True)
    parser.add_argument('--latency-multiplier', required=True)
    parser.add_argument('--thinker-num-beams', default='4')
    parser.add_argument('--limit-per-direction', type=int, default=0)
    args = parser.parse_args()
    # Install the same elapsed/wall_sec and Stage 2 hooks as the legacy runner.
    import omni_simuleval_wrap  # noqa: F401
    mark('wrapper_import_and_stage2_hook')
    import src.inference.infer_omni_talker_tts as tts
    from simuleval.utils.agent import build_system_args, EVALUATION_SYSTEM_LIST
    from simuleval.evaluator import build_evaluator
    mark('all_imports')

    cls = tts.OmniTalkerSemanticTTSInference
    original = cls.from_pretrained
    signature = inspect.signature(original)
    shared = {}

    def load_once(_cls, *pos, **kw):
        bound = signature.bind(*pos, **kw)
        bound.apply_defaults()
        if shared:
            if bound.arguments != shared['arguments']:
                raise RuntimeError('Refusing to reuse inference with different loading arguments')
            model = shared['model']
            # Generation diagnostics are the only per-call field on the inference
            # container; histories and KV caches belong to each new agent's states.
            model.last_talker_generation_diagnostics = {}
            print('[shared-model] reuse loaded inference', flush=True)
            return model
        start = time.perf_counter()
        model = original(*pos, **kw)
        shared.update(arguments=bound.arguments, model=model)
        mark('from_pretrained', start)
        return model

    cls.from_pretrained = classmethod(load_once)
    rows = [json.loads(line) for line in Path(args.manifest).read_text().splitlines() if line.strip()]
    for direction, sl, tl, tok, unit in [
        ('en2zh', 'English', 'Chinese', 'zh', 'char'),
        ('zh2en', 'Chinese', 'English', '13a', 'word'),
    ]:
        selected = [r for r in rows if not r['id'].startswith('_warmup_')
                    and f"{r['src_lang']}2{r['tgt_lang']}" == direction]
        if args.limit_per_direction:
            selected = selected[:args.limit_per_direction]
        if not selected:
            continue
        warmup = dict(selected[0], id='_warmup_' + selected[0]['id'])
        selected = [warmup] + selected
        output = Path(args.out_dir) / f'run_{direction}'
        output.mkdir(parents=True, exist_ok=True)
        if (output / 'instances.log').exists():
            raise RuntimeError(f'Output already exists: {output}')
        for name, key in [('ids', 'id'), ('source', 'audio'), ('target', 'reference')]:
            (output / f'{name}.txt').write_text(''.join(str(r[key]).replace('\n', ' ') + '\n' for r in selected))
        sys.argv = ['simuleval.cli',
            '--agent', 'src/agents/simuleval_omni_talker_s2st_agent.py',
            '--source-segment-size', '1000',
            '--model-name-or-path', 'models/SimulS2ST-Omni/offline',
            '--checkpoint-path', args.checkpoint_path,
            '--source', str(output / 'source.txt'), '--target', str(output / 'target.txt'),
            '--output', str(output), '--source-lang', sl, '--target-lang', tl,
            '--latency-multiplier', args.latency_multiplier,
            '--sacrebleu-tokenizer', tok, '--eval-latency-unit', unit,
            '--history-window-turns', '28', '--history-overlap-turns', '16',
            '--prompt-source-chunks', '2', '--prompt-generated-chunks', '1',
            '--thinker-max-new-tokens', '256', '--talker-max-new-tokens', '500',
            '--thinker-no-sample', '--talker-no-sample',
            '--thinker-num-beams', args.thinker_num_beams,
            '--thinker-repetition-penalty', '1.2', '--thinker-no-repeat-ngram-size', '5',
            '--talker-repetition-penalty', '1.4', '--talker-no-repeat-ngram-size', '5',
            '--computation-aware', '--no-scoring']
        # Each build imports the agent afresh; clear registrations from the prior
        # direction while retaining imported model modules and the inference object.
        EVALUATION_SYSTEM_LIST.clear()
        start = time.perf_counter()
        system, config = build_system_args()
        mark(f'{direction}.agent_build', start)
        start = time.perf_counter()
        evaluator = build_evaluator(config)
        mark(f'{direction}.evaluator_build', start)
        start = time.perf_counter()
        evaluator(system)
        mark(f'{direction}.evaluate', start)
        count = sum(1 for _ in (output / 'instances.log').open())
        if count != len(selected):
            raise RuntimeError(f'{direction}: expected {len(selected)} instances, got {count}')
        print(f'[shared-result] {direction}: {count}/{len(selected)}', flush=True)
        system.reset()
        del evaluator, system
        start = time.perf_counter()
        gc.collect()
        mark(f'{direction}.release_agent', start)
    mark('total_before_process_exit')


if __name__ == '__main__':
    main()
