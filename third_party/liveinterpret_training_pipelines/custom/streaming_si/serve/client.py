"""实时服务的测试客户端：把已切好的语音段按顺序推给 `/v1/stream`，记录逐段延迟。

段清单每行一个 JSON：{"id", "audio", "tgt_lang", 可选 "start_sec"}。
三种推流节奏：
    --realtime      按真实语速推进（段 i 在「前面各段时长之和」这一刻发出，模拟上游边切边送）
    --gap 0.0       段与段之间只隔 gap 秒（连续压测，看队列会不会堆）
    --burst         不等结果，一次性全部发出去（只看吞吐上限）

延迟口径（逐段）：
    submit_ms  客户端发出该段 -> 服务端 `accepted` 回来
    result_ms  `accepted` -> `result` 到达（含排队 + 推理 + 返回）
    e2e_ms     客户端发出该段 -> `result` 到达（对外可感知的那一个）

用法：
    python custom/streaming_si/serve/client.py --url ws://127.0.0.1:8000/v1/stream \
        --manifest /fsx/stevehguo/eval/work/manifests/all.jsonl --realtime --out_dir /tmp/serve_bench
"""
import argparse
import asyncio
import base64
import json
import os
import statistics
import time
from typing import Dict, List, Optional

import numpy as np


def _pct(values: List[float], q: float) -> Optional[float]:
    if not values:
        return None
    s = sorted(values)
    k = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return float(s[k])


def load_manifest(path: str, limit: Optional[int] = None) -> List[dict]:
    rows = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows[:limit] if limit else rows


async def run_session(url: str, rows: List[dict], *, realtime: bool, gap: float, burst: bool,
                      out_dir: Optional[str], max_seg_sec: float = 30.0, log=print) -> dict:
    import soundfile as sf
    import websockets

    os.makedirs(out_dir or '/tmp', exist_ok=True)
    results: List[dict] = []
    skipped: List[Optional[str]] = []    # 超过服务端上限、未发送的段
    t_stream0 = time.time()
    consumed = 0.0                      # 已推流的源音频秒数

    async with websockets.connect(url, max_size=None) as ws:
        async def reader():
            """收结果（JSON 头 + 可选二进制音频）。"""
            while True:
                msg = await ws.recv()
                if isinstance(msg, bytes):
                    idx = _last.get('idx')
                    if idx is not None and idx < len(results):
                        results[idx]['audio_bytes'] = len(msg)
                    continue
                obj = json.loads(msg)
                if obj.get('type') == 'done':
                    reader.done_info = obj
                    return
                if obj.get('type') == 'ready':
                    log(f'[client] 服务就绪：{ {k: v for k, v in obj.items() if k != "type"} }')
                    continue
                if obj.get('type') == 'accepted':
                    idx = obj['seq']
                    while len(results) <= idx:
                        results.append({})
                    results[idx].update(t_accepted=time.time(), source_sec=obj.get('source_sec'))
                    continue
                if 'seq' in obj and obj['seq'] >= 0:
                    idx = obj['seq']
                    while len(results) <= idx:
                        results.append({})
                    results[idx].update(t_result=time.time(), payload=obj)
                    _last['idx'] = idx

        _last: Dict[str, Optional[int]] = dict(idx=None)
        rtask = asyncio.create_task(reader())
        await ws.send(json.dumps(dict(type='config', tgt_lang=rows[0].get('tgt_lang', 'zh'))))

        for i, row in enumerate(rows):
            wav, sr = sf.read(row['audio'], dtype='float32', always_2d=False)
            if wav.ndim > 1:
                wav = wav.mean(axis=-1)
            sec = len(wav) / sr
            if sec > max_seg_sec:
                log(f'[client][WARN] {row.get("id")} 时长 {sec:.1f}s 超过服务端上限 '
                    f'{max_seg_sec:.0f}s，会被截断；这一段不计入延迟统计')
                skipped.append(row.get('id'))
                continue
            if realtime:
                # 按真实语速推进：段 i 应在「前面所有段的时长之和」这一刻发出
                wait = consumed - (time.time() - t_stream0)
                if wait > 0:
                    await asyncio.sleep(wait)
            elif gap > 0:
                await asyncio.sleep(gap)
            data = _wav_bytes(row['audio'])
            t_send = time.time()
            await ws.send(data)
            while len(results) <= i:
                results.append({})
            results[i].update(id=row.get('id'), t_send=t_send, src_sec=round(sec, 3),
                              tgt_lang=row.get('tgt_lang'), seq=i)
            consumed += sec
            if burst:
                continue

        await ws.send(json.dumps(dict(type='done')))
        try:
            await asyncio.wait_for(rtask, timeout=1800)
        except asyncio.TimeoutError:
            rtask.cancel()
    t_wall = time.time() - t_stream0

    # ---- 汇总 ----
    for r in results:
        if r.get('t_accepted') and r.get('t_send'):
            r['submit_ms'] = (r['t_accepted'] - r['t_send']) * 1000
        if r.get('t_result'):
            if r.get('t_accepted'):
                r['result_ms'] = (r['t_result'] - r['t_accepted']) * 1000
            if r.get('t_send'):
                r['e2e_ms'] = (r['t_result'] - r['t_send']) * 1000
            r['payload'] = {k: v for k, v in (r.get('payload') or {}).items() if k != 'audio_b64'}

    ok = [r for r in results if r.get('t_result')]
    e2e = [r['e2e_ms'] for r in ok if r.get('e2e_ms') is not None]
    infer = [r['payload'].get('infer_ms') for r in ok if r.get('payload', {}).get('infer_ms') is not None]
    queue = [r['payload'].get('queue_ms') for r in ok if r.get('payload', {}).get('queue_ms') is not None]
    src_total = sum(r.get('src_sec') or 0 for r in results)
    summary = dict(n=len(results), ok=len(ok), failed=len(results) - len(ok),
                   skipped=sorted({str(x) for x in skipped if x is not None}),
                   src_sec=round(src_total, 2), wall_sec=round(t_wall, 2),
                   rtf=round(t_wall / src_total, 3) if src_total else None,
                   e2e_ms=dict(p50=_pct(e2e, .5), p90=_pct(e2e, .9), max=_pct(e2e, 1.0),
                               mean=round(statistics.fmean(e2e), 1) if e2e else None),
                   infer_ms=dict(p50=_pct(infer, .5), p90=_pct(infer, .9),
                                 max=_pct(infer, 1.0), mean=round(statistics.fmean(infer), 1) if infer else None),
                   queue_ms=dict(p50=_pct(queue, .5), p90=_pct(queue, .9), max=_pct(queue, 1.0)))

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, 'client_results.jsonl'), 'w', encoding='utf-8') as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
        with open(os.path.join(out_dir, 'client_summary.json'), 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        refs = [r for r in results if r.get('payload', {}).get('text')]
        if refs:
            with open(os.path.join(out_dir, 'transcripts.jsonl'), 'w', encoding='utf-8') as f:
                for r in refs:
                    f.write(json.dumps(dict(id=r.get('id'), src_sec=r.get('src_sec'),
                                            text=r['payload'].get('text')), ensure_ascii=False) + '\n')
    return summary


def _wav_bytes(path: str) -> bytes:
    with open(path, 'rb') as f:
        return f.read()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--url', default='ws://127.0.0.1:8000/v1/stream')
    ap.add_argument('--manifest', required=True, help='每行 {"id","audio","tgt_lang"}')
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--realtime', action='store_true', help='按真实语速推流')
    ap.add_argument('--gap', type=float, default=0.0)
    ap.add_argument('--burst', action='store_true')
    ap.add_argument('--max_seg_sec', type=float, default=30.0,
                    help='服务端单段上限（与 --max_audio_sec 一致），超过的段直接跳过并告警')
    ap.add_argument('--out_dir', default=None)
    args = ap.parse_args(argv)

    rows = load_manifest(args.manifest, args.limit)
    t0 = time.time()
    summary = asyncio.run(run_session(args.url, rows, realtime=args.realtime, gap=args.gap,
                                      burst=args.burst, out_dir=args.out_dir,
                                      max_seg_sec=args.max_seg_sec))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f'[client] 总用时 {time.time() - t0:.1f}s')


if __name__ == '__main__':
    main()
