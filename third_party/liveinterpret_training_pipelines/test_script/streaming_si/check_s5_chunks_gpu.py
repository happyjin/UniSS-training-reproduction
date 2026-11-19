"""GPU integration: small chunks, EOF tails, and actual PCM accounting.

This checks transport/frontend contracts on real S5 weights. It does not claim
translation quality, acoustic seam quality, or representative benchmark RTF.
"""
import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import soundfile as sf
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('model_dir', 'adapter', 'manifest', 'cosyvoice_repo', 'cosyvoice_dir',
                'prompt_tokenizer', 'warmup_audio', 'out'):
        parser.add_argument('--' + key, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; CPU model fallback is disabled')
    from streaming_si.inference import infer_s5_streaming_dynchunk as infer
    from streaming_si.inference.infer_s2st import CampplusSpeaker, build_source_prompt, load_wav16
    from streaming_si.model.speech_decoder import resolve_speech_decoder

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    common = infer.parse_args([
        '--model_dir', args.model_dir, '--adapter', args.adapter, '--input_jsonl', args.manifest,
        '--out_dir', str(out), '--cosyvoice_repo', args.cosyvoice_repo,
        '--cosyvoice_dir', args.cosyvoice_dir, '--prompt_tokenizer', args.prompt_tokenizer,
        '--synthesis_mode', 'streaming', '--segmentation', 'none', '--retry', '0',
        '--max_audio_sec', '0', '--prompt_sec', '2', '--chunk_ms', '640'])
    torch.manual_seed(0)
    model, tokenizer, ids = infer.load_model(args.model_dir, args.adapter)
    model.to('cuda:0')
    directory, yaml = resolve_speech_decoder(args.model_dir, args.cosyvoice_dir, None)
    model.config.talker_config_yaml = yaml
    model.attach_speech_decoder(directory, device='cuda:0', cosyvoice_repo=args.cosyvoice_repo, strict=True)
    speaker = CampplusSpeaker(str(Path(directory) / 'campplus.onnx'))
    prompter = build_source_prompt(True, directory, args.prompt_tokenizer)
    processor = infer.VoxtralAudioProcessor(infer.VoxtralFeatureExtractor.from_pretrained(args.model_dir))
    warm = infer.stream_file(model, tokenizer, ids, processor,
        load_wav16(args.warmup_audio, max_sec=None), 'zh', common, speaker=speaker, prompter=prompter)
    if not warm['wav'].size:
        raise RuntimeError('warmup produced no actual PCM')
    del warm
    manifest = [json.loads(line) for line in Path(args.manifest).read_text().splitlines() if line.strip()]
    reports = []
    for chunk, after in ((640, None), (1280, None), (1920, None), (2560, None)):
        common.chunk_ms, common.chunk_ms_after_write = chunk, after
        name = str(chunk) + ('-' + str(after) if after else '')
        folder = out / name
        folder.mkdir()
        for row in manifest:
            audio = load_wav16(row['audio'], max_sec=None)
            pcm, offset = [], 0
            events_path = folder / (row['id'] + '-events.jsonl')
            with events_path.open('x') as log:
                def callback(event):
                    nonlocal offset
                    if torch.is_grad_enabled():
                        raise RuntimeError('model session escaped inference mode')
                    record = {key: value for key, value in event.items() if key != 'pcm'}
                    if event['type'] == 'pcm':
                        if event['sample_offset'] != offset:
                            raise RuntimeError('PCM gap or duplication')
                        values = np.asarray(event['pcm']).copy()
                        pcm.append(values)
                        offset += values.size
                        record['num_samples'] = int(values.size)
                    if event['source_received_sec'] + 1 / 16000 < event['source_sec']:
                        raise RuntimeError('source consumed before reception')
                    log.write(json.dumps(record, ensure_ascii=False) + '\n')
                started = time.perf_counter()
                result = infer.stream_file(model, tokenizer, ids, processor, audio,
                    row['tgt_lang'], common, speaker=speaker, prompter=prompter, on_event=callback)
            emitted = np.concatenate(pcm) if pcm else np.zeros(0, np.float32)
            if result['stop'] != 'eos' or not emitted.size:
                raise RuntimeError('session did not produce a complete nonempty output')
            if not np.array_equal(result['wav'], emitted):
                raise RuntimeError('saved output differs from actually emitted PCM')
            if abs(result['source_sec'] - audio.size / 16000) > 1 / 16000:
                raise RuntimeError('source samples lost or duplicated')
            expected = math.ceil((audio.size // processor.fe.hop_length) / 8)
            actual = sum(step['n_audio_frames'] for step in result['steps'])
            if actual != expected or result['pending_mel_frames']:
                raise RuntimeError(f'frontend tail accounting: encoded {actual}, expected {expected}')
            sf.write(str(folder / (row['id'] + '.wav')), emitted, 24000)
            result.pop('wav')
            (folder / (row['id'] + '.json')).write_text(json.dumps(result, ensure_ascii=False, indent=2))
            reports.append(dict(configuration=name, id=row['id'], source_samples=int(audio.size),
                encoded_audio_frames=actual, expected_audio_frames=expected, pcm_samples=offset,
                wall_sec=time.perf_counter() - started, status='passed'))
            (out / 'checks.json').write_text(json.dumps(reports, ensure_ascii=False, indent=2))
            print(json.dumps(reports[-1]), flush=True)
    (out / 'complete.json').write_text(json.dumps(dict(status='passed', n=len(reports),
        quality_verified=False, human_listening_verified=False), indent=2))


if __name__ == '__main__':
    main()
