"""三个评测共用的 fluency 全量汇总、Markdown 与试听页面展示。"""
import html
import json
import os
import statistics
from collections import Counter

from schema import BOUNDARY_LABELS_ZH, BOUNDARY_TYPES

COLORS = dict(zip(BOUNDARY_TYPES, ['#c0392b', '#d35400', '#2471a3', '#1e8449']))
PHASE_LABELS = {'before_end': '源音尚未播完', 'after_end': '源音已经播完'}


def read_rows(path):
    out = {}
    if os.path.isfile(path):
        with open(path, encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    out[row['id']] = row
    return out


def divide(a, b, scale=1):
    return round(a / b * scale, 4) if b else None


def summarize(rows):
    ok = [r for r in rows if r.get('status') == 'ok' and r.get('llm_complete')]
    pauses = [p for r in ok for p in r['pauses'] if p['scored'] and p['kind'] == 'internal']
    abnormal = [p for p in pauses if p['excess_s'] > 0]
    chars = sum(r['transcript_chars'] for r in ok)
    excess = sum(p['excess_s'] for p in pauses)
    burdens = [r['abnormal_pause_burden'] for r in ok if r.get('abnormal_pause_burden') is not None]
    types = {}
    for kind in BOUNDARY_TYPES:
        group = [p for p in pauses if p['boundary_type'] == kind]
        types[kind] = dict(label=BOUNDARY_LABELS_ZH[kind], pause_count=len(group),
                           abnormal_count=sum(p['excess_s'] > 0 for p in group),
                           pause_sec=round(sum(p['duration_s'] for p in group), 3),
                           excess_sec=round(sum(p['excess_s'] for p in group), 3),
                           excess_share_pct=divide(sum(p['excess_s'] for p in group), excess, 100))
    phases = {}
    for phase in PHASE_LABELS:
        group = [p for p in abnormal if p['source_phase'] == phase]
        phases[phase] = dict(label=PHASE_LABELS[phase], abnormal_count=len(group),
                            count_share_pct=divide(len(group), len(abnormal), 100))
    before = sum(p['excess_before_source_end_s'] for p in pauses)
    after = sum(p['excess_after_source_end_s'] for p in pauses)
    phases['before_end'].update(excess_sec=round(before, 3), excess_share_pct=divide(before, excess, 100))
    phases['after_end'].update(excess_sec=round(after, 3), excess_share_pct=divide(after, excess, 100))
    dominant = max(types, key=lambda k: types[k]['excess_sec']) if excess else None
    dominant_count = max(types, key=lambda k: types[k]['abnormal_count']) if abnormal else None
    return dict(n_total=len(rows), n_ok=len(ok),
                n_skipped=sum(r.get('status') == 'skipped' for r in rows),
                skip_reasons=dict(Counter(r.get('reason', '?') for r in rows if r.get('status') == 'skipped')),
                n_error=len(rows) - len(ok) - sum(r.get('status') == 'skipped' for r in rows),
                total_chars=chars, pause_count=len(pauses), abnormal_count=len(abnormal),
                excess_sec=round(excess, 3), burden_pooled=divide(excess, chars, 100),
                burden_mean=round(statistics.mean(burdens), 4) if burdens else None,
                burden_median=round(statistics.median(burdens), 4) if burdens else None,
                burden_p90=sorted(burdens)[min(len(burdens)-1, int(len(burdens)*0.9))] if burdens else None,
                types=types, source_phases=phases,
                crosses_source_end_count=sum(p.get('crosses_source_end', False) for p in abnormal),
                dominant_type=dominant, dominant_type_by_count=dominant_count,
                tolerance_table=ok[0]['tolerance_table'] if ok else {},
                llm_model=ok[0].get('llm_model') if ok else None,
                vad_threshold=ok[0].get('vad_threshold') if ok else None,
                transcript_sources=dict(Counter(r.get('transcript_source', '?') for r in ok)),
                align_sources=dict(Counter(r.get('align_source', 'uniform') for r in ok)),
                alignment_languages=dict(Counter(r.get('alignment', {}).get('language', '?') for r in ok)),
                alignment_models=dict(Counter(r.get('alignment', {}).get('model_version', '?') for r in ok)),
                alignment_ambiguous_pauses=sum(p.get('alignment_ambiguous', False) for p in pauses),
                alignment_methods=dict(Counter(p['alignment_method'] for p in pauses if p.get('alignment_method'))),
                alignment_moved_pauses=sum('alignment_legacy_index' in p and p['alignment_legacy_index'] != p['char_index']
                                           for p in pauses))


def attach_fluency(report, root):
    all_rows = []
    found = False
    for s in report['sets']:
        path = os.path.join(root, 'runs', s['name'], os.getenv('FLUENCY_RESULTS_FILE', 'fluency.jsonl'))
        if not os.path.isfile(path):
            continue
        found = True
        rows = list(read_rows(path).values())
        summary = summarize(rows)
        summary['n_expected'] = s['n']
        summary['n_missing'] = max(0, s['n'] - len(rows))
        s['fluency'] = summary
        all_rows += rows
    if found:
        report['fluency'] = summarize(all_rows)
        report['fluency']['n_missing'] = sum(s.get('fluency', {}).get('n_missing', s['n']) for s in report['sets'])


def fmt(v):
    return '—' if v is None else f'{v:.2f}'


def markdown_section(report):
    if 'fluency' not in report:
        return []
    total = report['fluency']
    mfa = total.get('align_sources', {}).get('mfa', 0) > 0
    description = ('SageMaker 强制对齐服务（中文字符级 / 英文词级）' if mfa else '累计语音时长加权近似对齐')
    lines = ['\n## Fluency · 译音流畅度（全量）\n',
             f'- 复用 `eval_fluency_scripts`：Silero VAD 实测已有时间轴译音里的停顿 → 原评分文本的{description} → prod LLM 判定四类句法边界。',
             f'- LLM：`{total["llm_model"]}`，speech_LLM(prod)；VAD threshold={total["vad_threshold"]}，首包等待和末尾静音不计入流畅度。',
             '- **异常停顿负担（越低越好）** = Σ max(0, 停顿时长 − 对应类型容忍) / Σ译文非空白字符数 × 100（秒/100字，含标点；英文按字符而非词计数，只宜在相同方向测试集内横向比较）。另列逐句均值、中位数、P90。',
             ('- 只替换第 3 步：保留上一轮 ASR 优先的评分文本、字数分母、VAD 与四档容忍。对齐请求去标点，时间戳映射回原文本；英文仅词级，不声称能精确找出单词内部断音。ASR 标点对边界判断的潜在偏差仍存在。' if mfa else '- 本次没有精确字级真值；停顿时间来自 VAD，上下文依赖加权近似对齐，可能错位。'),
             f'- 对齐来源：`{total.get("align_sources", {})}`；语言：`{total.get("alignment_languages", {})}`；服务版本：`{total.get("alignment_models", {})}`；不确定停顿 {total.get("alignment_ambiguous_pauses", 0)} 处。强制对齐时间戳是模型估计，不是人工真值；LLM 不判断停顿的因果来源。',
             *([f'- 停顿定位 v2：默认取“起点在停顿前说出”的最后一个单元之后；英文被拉长的词看静音前后发声多少决定归属；中文按 jieba 分词不切开词、不把单独成词的助词放到停顿之后。定位方式 `{total["alignment_methods"]}`；相对 v1（间隙离停顿最近）位置改变 {total.get("alignment_moved_pauses", 0)} 处。不确定 = 距最近边界 >0.1s、或多候选且与 v1 不一致、或单元与静音大面积重叠。'] if total.get('alignment_methods') else []),
             *([f'- ⚠ 服务无法对齐而显式回退近似对齐的样本 {total.get("align_sources", {}).get("uniform_fallback", 0)} 句（`uniform_fallback`，逐句记录原因），其停顿位置仍为估算。'] if total.get('align_sources', {}).get('uniform_fallback') else []),
             '- 超额为正才算卡顿；未超过容忍的自然停顿不算卡顿。有语音且无内部停顿的样本无需对齐（`not_required`），正常以零内部停顿负担计入分母；空语音/空译文、推理失败单独计数，不赋予零负担；评测错误、缺失样本不会隐去。',
             '\n### 可配置容忍阈值（秒）\n',
             '| 类型 | 当前容忍 | 环境变量 |', '|---|---:|---|']
    envs = ['TOL_WORD_INTERNAL', 'TOL_TIGHT_STRUCTURE', 'TOL_PHRASE_BOUNDARY', 'TOL_SENTENCE_BOUNDARY']
    for kind, env in zip(BOUNDARY_TYPES, envs):
        lines.append(f'| {BOUNDARY_LABELS_ZH[kind]} | {fmt(total["tolerance_table"].get(kind))} | `{env}` |')
    lines += ['\n### 各测试集（全部推理句子）\n',
              '| 测试集 | 总句数 | 有效评分 | 空/失败跳过 | 评测错误/缺失 | 卡顿次数 | 超额总秒 | 负担：汇总/均值/中位/P90 |',
              '|---|---:|---:|---:|---:|---:|---:|---|']
    skip_notes = []
    for s in report['sets']:
        q = s.get('fluency')
        if not q:
            lines.append(f'| {s["label"]} | {s["n"]} | — | — | 未评测 | — | — | — |')
            continue
        values = '/'.join(fmt(q[k]) for k in ['burden_pooled', 'burden_mean', 'burden_median', 'burden_p90'])
        lines.append(f'| {s["label"]} | {q["n_expected"]} | {q["n_ok"]} | {q["n_skipped"]} | {q["n_error"]}/{q["n_missing"]} | {q["abnormal_count"]} | {fmt(q["excess_sec"])} | {values} |')
        if q['skip_reasons']:
            skip_notes.append(f'{s["label"]}：`{q["skip_reasons"]}`')
    if skip_notes:
        lines.append('\n跳过原因 — ' + '；'.join(skip_notes) + '。')
    for label, q in [(s['label'], s['fluency']) for s in report['sets'] if 'fluency' in s] + [('全部测试集合计', total)]:
        lines += [f'\n### {label} · 四类卡顿与位置\n',
                  '| 类型 | 检测停顿次数（含容忍内） | 卡顿次数 | 超额秒 | 超额占比 |',
                  '|---|---:|---:|---:|---:|']
        for kind in BOUNDARY_TYPES:
            t = q['types'][kind]
            lines.append(f'| {t["label"]} | {t["pause_count"]} | {t["abnormal_count"]} | {fmt(t["excess_sec"])} | {fmt(t["excess_share_pct"])}% |')
        if q['dominant_type']:
            lines.append(f'\n**主要类型（按超额时长）**：{BOUNDARY_LABELS_ZH[q["dominant_type"]]}；按卡顿次数最多：{BOUNDARY_LABELS_ZH[q["dominant_type_by_count"]]}。')
        else:
            lines.append('\n没有超过容忍的停顿。')
        lines += ['\n| 卡顿相对源音位置 | 起点次数 | 起点占比 | 超额秒（跨界拆分） | 超额占比 |', '|---|---:|---:|---:|---:|']
        for phase in PHASE_LABELS:
            t = q['source_phases'][phase]
            lines.append(f'| {t["label"]} | {t["abnormal_count"]} | {fmt(t["count_share_pct"])}% | {fmt(t["excess_sec"])} | {fmt(t["excess_share_pct"])}% |')
        if q['abnormal_count']:
            best = max(q['source_phases'], key=lambda k: q['source_phases'][k]['abnormal_count'])
            time_best = max(q['source_phases'], key=lambda k: q['source_phases'][k]['excess_sec'])
            lines.append(f'\n**集中位置**：按起点次数为“{PHASE_LABELS[best]}”；按超额时长为“{PHASE_LABELS[time_best]}”。跨越源音结束的卡顿 {q["crosses_source_end_count"]} 处。')
    lines += ['\n位置定义：时间轴 0=源音开始，源结束=`source_sec`。次数按停顿起点分类（等于源结束算结束后）；'
              '超额区间为 `[停顿起点+容忍, 停顿终点]`，遇到源结束按实际重叠秒数拆分，前后超额总和守恒。',
              '逐句明细见 `raw/<set>/fluency.jsonl`；本报告全量统计，`demo.html` 仍只展示原先挑选的句子。']
    return lines


def decorate_items(root, items):
    rows = {}
    for name in {it['set'] for it in items}:
        rows[name] = read_rows(os.path.join(root, 'runs', name, os.getenv('FLUENCY_RESULTS_FILE', 'fluency.jsonl')))
    for it in items:
        it['fluency'] = rows[it['set']].get(it['id'])


STYLE = """
*{box-sizing:border-box}body{margin:0;background:#f5f6f7;color:#1c2833;font-family:-apple-system,'PingFang SC',sans-serif;line-height:1.6}
main{max-width:1150px;margin:28px auto;padding:0 20px}h1{font-size:24px}h2{font-size:19px;margin-top:30px}.sub,.m{color:#5d6d7e;font-size:13px}
.summary{display:flex;flex-wrap:wrap;gap:26px;background:white;padding:18px;border:1px solid #dfe4e8;border-radius:10px;margin:20px 0}
.summary b{display:block;font-size:22px}.legend{display:flex;flex-wrap:wrap;gap:10px;margin:12px 0}.tag{color:white;border-radius:4px;padding:2px 8px;font-size:12px}
.c{background:white;border:1px solid #dfe4e8;border-radius:10px;padding:18px;margin:16px 0}.a{display:flex;flex-wrap:wrap;gap:16px;align-items:center}.a span{min-width:130px;font-size:13px}audio{width:min(700px,100%);height:38px;margin:8px 0}
table{border-collapse:collapse;width:100%;font-size:13px;margin:12px 0}td,th{text-align:left;padding:7px 10px;border-bottom:1px solid #eef2f5;vertical-align:top}th{color:#5d6d7e}.scroll{overflow-x:auto}
.pause{background:#fdebd0;color:#b9770e;padding:1px 5px;border-radius:4px}.ctx{background:#fbfcfc;padding:9px;border-radius:5px}.reason{font-size:12px;color:#7f8c8d}
.timeline{height:26px;background:#edf2f6;position:relative;margin:14px 0 4px;border-radius:5px}.bar{position:absolute;top:0;height:100%;opacity:.8;border-radius:3px}.srcend{position:absolute;top:-5px;height:36px;border-left:2px dashed #17202a}
button{border:1px solid #ccd3db;border-radius:5px;background:white;padding:4px 9px;color:#2471a3;cursor:pointer}.score{color:#c0392b;font-weight:600}.warn{color:#9c640c}
"""
SCRIPT = """
document.addEventListener('click',function(e){const b=e.target.closest('[data-audio]');if(!b)return;
const a=document.getElementById(b.dataset.audio);if(!a)return;document.querySelectorAll('audio').forEach(x=>{if(x!==a)x.pause()});
const start=Number(b.dataset.seek);const go=()=>{a.currentTime=start;a.play().catch(()=>{})};
if(a.readyState>=1)go();else{a.addEventListener('loadedmetadata',go,{once:true});a.load()}});
"""


def render_demo(root, items, label, note):
    e = html.escape
    h = [f'<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{e(label)} · 试听与流畅度</title><style>{STYLE}</style></head><body><main>',
         f'<h1>{e(label)} · 试听与流畅度</h1><p class="sub">{e(note)}</p>',
         '<p class="sub">停顿按 VAD 实测，四类标签来自 LLM；对齐来源见每句。首包等待不计分。点击“听此处”从停顿前 1 秒起播；标记为“不确定”的位置需试听复核。</p>']
    rp = os.path.join(root, 'report', 'report.json')
    report = {}
    if os.path.isfile(rp):
        with open(rp, encoding='utf-8') as f:
            report = json.load(f)
    q = report.get('fluency')
    if q:
        h.append(f'<div class="summary"><div>全量评分/推理句数<b>{q["n_ok"]}/{q["n_total"]}</b></div><div>超额卡顿<b>{q["abnormal_count"]} 处</b></div><div>全量异常停顿负担<b>{fmt(q["burden_pooled"])} 秒/100字</b></div><div>本页试听<b>{len(items)} 句</b></div></div>')
        h.append('<div class="legend">')
        for kind in BOUNDARY_TYPES:
            h.append(f'<span class="tag" style="background:{COLORS[kind]}">{BOUNDARY_LABELS_ZH[kind]} · 容忍 {fmt(q["tolerance_table"].get(kind))}s</span>')
        h.append('</div><p class="sub">上方统计为全部测试集，不是这几条试听。阈值由任务环境变量配置，修改后可复用分类缓存重新计分。</p>')
    cur = None
    for k, it in enumerate(items):
        if cur != it['set']:
            cur = it['set']
            h.append(f'<h2>{e(cur)}</h2>')
        aid = f'audio-{k}'
        h.append(f'<section class="c"><b>{e(it["id"])}</b> <span class="m">源 {it["source_sec"]}s · LAAL-BOW {fmt(it.get("laal_bow"))}s · 首包 {fmt(it.get("start_offset"))}s · 尾包 {fmt(it.get("end_offset"))}s</span>')
        h.append(f'<div class="a"><span>译音（保留等待）</span><audio id="{aid}" controls preload="none" src="{e(it["audio"])}"></audio></div><div class="a"><span>源音频</span><audio controls preload="none" src="{e(it["src_audio"])}"></audio></div>')
        for name, text in [('源文', it.get('src_text', '')), ('参考', it.get('reference', '')), ('模型译文', it.get('text', '')), ('译音 ASR', it.get('asr', ''))]:
            h.append(f'<div class="m">{name}：{e(text or "")}</div>')
        r = it.get('fluency')
        if not r or r.get('status') != 'ok' or not r.get('llm_complete'):
            h.append(f'<p class="warn">流畅度未评分：{e(str((r or {}).get("reason") or (r or {}).get("error") or "缺失/分类未完成"))}</p></section>')
            continue
        h.append(f'<p><span class="score">异常停顿负担 {fmt(r.get("abnormal_pause_burden"))} 秒/100字</span> · 卡顿 {r["abnormal_pause_count"]} 处 · 超额 {fmt(r["total_excess_s"])}s <span class="m">（译音文本：{e(r["transcript_source"])}，对齐：{e(r.get("align_source", "uniform"))} / {e(r.get("alignment", {}).get("resolution", "approximate"))}）</span></p>')
        dur = max(r['duration_s'], 0.001)
        h.append('<div class="timeline">')
        for p in r['pauses']:
            color = COLORS.get(p.get('boundary_type'), '#bfc9d4')
            title = f'{p["start_s"]}-{p["end_s"]}s：{p.get("boundary_type_zh") or p["kind"]}，超额 {p["excess_s"]}s'
            h.append(f'<span class="bar" style="left:{p["start_s"]/dur*100:.3f}%;width:{p["duration_s"]/dur*100:.3f}%;background:{color}" title="{e(title)}"></span>')
        if it['source_sec'] <= dur:
            h.append(f'<span class="srcend" style="left:{it["source_sec"]/dur*100:.3f}%" title="源音结束 {it["source_sec"]}s"></span>')
        h.append(f'</div><div class="m">时间轴：0–{fmt(dur)}s；黑虚线=源结束 {it["source_sec"]}s（若在译音之后则不绘制）。灰色首尾静音不计分。</div>')
        h.append('<div class="scroll"><table><thead><tr><th>试听</th><th>类型 / 上下文</th><th>停顿起止</th><th>时长 / 容忍 / 超额</th><th>相对源音位置</th></tr></thead><tbody>')
        internal = [p for p in r['pauses'] if p['scored'] and p['kind'] == 'internal']
        for p in internal:
            kind = p['boundary_type']
            ctx = e(p['context']).replace('〔停〕', f'<span class="pause">〔停 {fmt(p["duration_s"])}s〕</span>')
            if p.get('alignment_ambiguous'):
                ctx += f'<div class="warn">对齐不确定 · 距最近边界 {fmt(p.get("align_distance_s"))}s；需试听复核</div>'
            phase = PHASE_LABELS[p['source_phase']] + ('（跨越源结束）' if p['crosses_source_end'] else '')
            h.append(f'<tr><td><button data-audio="{aid}" data-seek="{max(0,p["start_s"]-1):.3f}">听此处</button></td><td><span class="tag" style="background:{COLORS[kind]}">{BOUNDARY_LABELS_ZH[kind]}</span><div class="ctx">{ctx}</div><div class="reason">LLM：{e(p["boundary_reason"])}</div></td><td>{fmt(p["start_s"])}–{fmt(p["end_s"])}s</td><td>{fmt(p["duration_s"])}/{fmt(p["tolerance_s"])}/<b>{fmt(p["excess_s"])}</b>s<br>{"卡顿" if p["excess_s"]>0 else "容忍内停顿"}</td><td>{e(phase)}</td></tr>')
        if not internal:
            h.append('<tr><td colspan="5">无达到 VAD 最短停顿阈值的句中停顿。</td></tr>')
        h.append('</tbody></table></div></section>')
    h.append(f'</main><script>{SCRIPT}</script></body></html>')
    return '\n'.join(h)


def render_job_env(path, tolerance, jobs, limit):
    import math
    import re
    import shlex
    mapping = dict(zip(BOUNDARY_TYPES, ['TOL_WORD_INTERNAL', 'TOL_TIGHT_STRUCTURE',
                                     'TOL_PHRASE_BOUNDARY', 'TOL_SENTENCE_BOUNDARY']))
    values = {}
    for item in shlex.split(tolerance):
        kind, value = item.split('=', 1)
        if kind not in mapping or not math.isfinite(float(value)) or float(value) < 0:
            raise ValueError(f'无效容忍阈值：{item}')
        values[mapping[kind]] = value
    for key, value in [('FLUENCY_JOBS', jobs), ('FLUENCY_LIMIT', limit)]:
        if value:
            if int(value) < 1:
                raise ValueError(f'{key} 必须为正整数')
            values[key] = value
    with open(path, encoding='utf-8') as f:
        text = f.read()
    for key, value in values.items():
        text, n = re.subn(rf'^    {key}: .*$', f'    {key}: {json.dumps(value)}', text, flags=re.M)
        if n != 1:
            raise ValueError(f'模板缺少 {key}')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)
    if values:
        print('[submit] fluency overrides:', values)


def refresh_outputs(root):
    import re
    import shutil
    from pathlib import Path
    report_dir = Path(root) / 'report'
    report = json.loads((report_dir / 'report.json').read_text(encoding='utf-8'))
    items = json.loads((report_dir / 'demo.json').read_text(encoding='utf-8'))
    previous = report.get('fluency', {})
    expected = sum(s['n'] for s in report['sets'])
    attach_fluency(report, root)
    q = report.get('fluency', {})
    sources = q.get('align_sources', {})
    aligned = sources.get('mfa', 0) + sources.get('not_required', 0) + sources.get('uniform_fallback', 0)
    if q.get('n_total') != expected or q.get('n_error') or q.get('n_missing') or aligned != q.get('n_ok'):
        raise ValueError('MFA 评测不完整，禁止更新报告/demo')
    filename = os.getenv('FLUENCY_RESULTS_FILE', 'fluency.jsonl')
    for s in report['sets']:
        old = read_rows(str(Path(root) / 'runs' / s['name'] / 'fluency.jsonl'))
        new = read_rows(str(Path(root) / 'runs' / s['name'] / filename))
        if set(old) != set(new):
            raise ValueError('MFA 与旧评测的样本清单不一致')
        for uid, row in new.items():
            baseline = old[uid]
            if row['status'] != baseline['status']:
                raise ValueError(f'新旧样本评分状态不一致：{uid}')
            if row['status'] == 'ok':
                if row['transcript'] != baseline['transcript'] or row['transcript_chars'] != baseline['transcript_chars']:
                    raise ValueError(f'评分文本或字数分母变化：{uid}')
                sig = lambda r: [(p['start_s'], p['end_s'], p['kind']) for p in r['pauses']]
                if sig(row) != sig(baseline):
                    raise ValueError(f'VAD 停顿变化：{uid}')
        dest = report_dir / 'raw' / s['name']
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path(root) / 'runs' / s['name'] / filename, dest / 'fluency.jsonl')
    decorate_items(root, items)
    md = (report_dir / 'REPORT.md').read_text(encoding='utf-8')
    section = '\n'.join(markdown_section(report)).strip() + '\n'
    md, count = re.subn(r'(?ms)^## Fluency .*?(?=^## |\Z)', lambda _: section, md)
    if count != 1:
        raise ValueError('找不到唯一旧 Fluency 章节')
    tag = os.getenv('FLUENCY_BACKUP_TAG', 'uniform')
    baseline_dir = report_dir / f'{tag}_backup'
    baseline_dir.mkdir(exist_ok=True)
    for name in ('REPORT.md', 'report.json', 'demo.html', 'demo.json'):
        if not (baseline_dir / name).exists():
            shutil.copyfile(report_dir / name, baseline_dir / name)
    report[f'fluency_previous_{tag}'] = previous
    (report_dir / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    title = re.search(r'<h1>(.*?)</h1>', (report_dir / 'demo.html').read_text(encoding='utf-8'))
    label = html.unescape(title[1]).removesuffix(' · 试听与流畅度') if title else report.get('model_label') or os.getenv('MODEL_LABEL', '流式译音')
    note = os.getenv('FLUENCY_DEMO_NOTE', '沿用上一轮所有试听选句与音频，仅更新 MFA 对齐后的流畅度标注。')
    page = render_demo(root, items, label, note)
    (report_dir / 'REPORT.md').write_text(md, encoding='utf-8')
    (report_dir / 'demo.json').write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding='utf-8')
    (report_dir / 'demo.html').write_text(page, encoding='utf-8')
    print(f'[fluency] report/demo refreshed: {q["n_ok"]}/{expected}; samples={len(items)}; audio untouched', flush=True)


if __name__ == '__main__':
    import sys
    if sys.argv[1:2] == ['--refresh']:
        refresh_outputs(sys.argv[2])
    else:
        render_job_env(*sys.argv[1:5])
