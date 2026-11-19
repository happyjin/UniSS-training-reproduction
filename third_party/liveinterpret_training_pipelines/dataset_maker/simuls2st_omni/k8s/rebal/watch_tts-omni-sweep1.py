#!/usr/bin/env python3
"""在 sweep Pod 内运行：把排队项移交给新 Pod，并在各 worker 做完当前项后停止它。

1) 为移交出去的排队项写占位完成标记（内容 MOVED_TO_B:<key>）。
   驱动只看标记是否存在，所以容器若被 OnFailure 自动重启，这些项不会再被处理；
   对账时按前缀排除，不会误算为已完成。
2) 每 15 秒检查：worker 的当前项一出现真实完成标记（数据已 flush），就 SIGKILL 该进程
   （不触发 finally，不会给下一项写标记），并存档它的日志（重启会截断原日志）。
"""
import hashlib
import json
import os
import shutil
import time

R = '/opt/dlami/nvme/stevehguo/tts_omni'
CUR = json.loads('''{"1": "speech_model_backup/stevehguo/data_fortrain/data_fortrain_1031version/TTS/zh/emilia/Emilia/ZH/ZH-B000406.tar", "2": "speech_model_backup/stevehguo/data_fortrain/data_fortrain_1031version/TTS/jp/audio/0103.tar", "3": "speech_model_backup/stevehguo/data_fortrain/data_fortrain_1031version/TTS/zh/emilia/Emilia/ZH/ZH-B000402.tar", "4": "speech_model_backup/stevehguo/data_fortrain/data_fortrain_1031version/TTS/zh/emilia/Emilia/ZH/ZH-B000736.tar", "5": "speech_model_backup/stevehguo/data_fortrain/data_fortrain_1031version/TTS/zh/emilia/Emilia/ZH/ZH-B000739.tar", "6": "speech_model_backup/stevehguo/data_fortrain/data_fortrain_1031version/TTS/zh/emilia/Emilia/ZH/ZH-B000063.tar", "7": "speech_model_backup/stevehguo/data_fortrain/data_fortrain_1031version/TTS/zh/emilia/Emilia/ZH/ZH-B000583.tar", "8": "speech_model_backup/stevehguo/data_fortrain/data_fortrain_1031version/TTS/ko/patterned_utterance_with_number/data/train-00089-of-00102.parquet", "9": "speech_model_backup/stevehguo/data_fortrain/data_fortrain_1031version/TTS/zh/emilia/Emilia/ZH/ZH-B000318.tar", "10": "speech_model_backup/stevehguo/data_fortrain/data_fortrain_1031version/TTS/jp/audio/0044.tar", "11": "speech_model_backup/stevehguo/data_fortrain/data_fortrain_1031version/TTS/jp/audio/0058.tar", "12": "speech_model_backup/stevehguo/data_fortrain/data_fortrain_1031version/TTS/zh/emilia/Emilia/ZH/ZH-B000076.tar", "13": "speech_model_backup/stevehguo/data_fortrain/data_fortrain_1031version/TTS/zh/emilia/Emilia/ZH/ZH-B000810.tar", "14": "speech_model_backup/stevehguo/data_fortrain/data_fortrain_1031version/TTS/jp/audio/0033.tar", "15": "speech_model_backup/stevehguo/data_fortrain/data_fortrain_1031version/TTS/zh/emilia/Emilia/ZH/ZH-B000743.tar"}''')
MOVED = json.loads('''["speech_model_backup/stevehguo/data_fortrain/data_fortrain_1031version/TTS/jp/audio/0003.tar", "speech_model_backup/stevehguo/data_fortrain/data_fortrain_1031version/TTS/jp/audio/0028.tar", "speech_model_backup/stevehguo/data_fortrain/data_fortrain_1031version/TTS/jp/audio/0040.tar", "speech_model_backup/stevehguo/data_fortrain/data_fortrain_1031version/TTS/jp/audio/0089.tar", "speech_model_backup/stevehguo/data_fortrain/data_fortrain_1031version/TTS/ko/patterned_utterance_with_number/data/train-00011-of-00102.parquet", "speech_model_backup/stevehguo/data_fortrain/data_fortrain_1031version/TTS/ko/patterned_utterance_with_number/data/train-00029-of-00102.parquet", "speech_model_backup/stevehguo/data_fortrain/data_fortrain_1031version/TTS/ko/patterned_utterance_with_number/data/train-00033-of-00102.parquet", "speech_model_backup/stevehguo/data_fortrain/data_fortrain_1031version/TTS/ko/patterned_utterance_with_number/data/train-00049-of-00102.parquet", "speech_model_backup/stevehguo/data_fortrain/data_fortrain_1031version/TTS/ko/patterned_utterance_with_number/data/train-00087-of-00102.parquet", "speech_model_backup/stevehguo/data_fortrain/data_fortrain_1031version/TTS/ko/patterned_utterance_with_number/data/validation-00003-of-00013.parquet", "speech_model_backup/stevehguo/data_fortrain/data_fortrain_1031version/TTS/zh/emilia/Emilia/ZH/ZH-B000053.tar", "speech_model_backup/stevehguo/data_fortrain/data_fortrain_1031version/TTS/zh/emilia/Emilia/ZH/ZH-B000061.tar", "speech_model_backup/stevehguo/data_fortrain/data_fortrain_1031version/TTS/zh/emilia/Emilia/ZH/ZH-B000062.tar", "speech_model_backup/stevehguo/data_fortrain/data_fortrain_1031version/TTS/zh/emilia/Emilia/ZH/ZH-B000078.tar", "speech_model_backup/stevehguo/data_fortrain/data_fortrain_1031version/TTS/zh/emilia/Emilia/ZH/ZH-B000087.tar", "speech_model_backup/stevehguo/data_fortrain/data_fortrain_1031version/TTS/zh/emilia/Emilia/ZH/ZH-B000099.tar", "speech_model_backup/stevehguo/data_fortrain/data_fortrain_1031version/TTS/zh/emilia/Emilia/ZH/ZH-B000140.tar", "speech_model_backup/stevehguo/data_fortrain/data_fortrain_1031version/TTS/zh/emilia/Emilia/ZH/ZH-B000142.tar", "speech_model_backup/stevehguo/data_fortrain/data_fortrain_1031version/TTS/zh/emilia/Emilia/ZH/ZH-B000154.tar", "speech_model_backup/stevehguo/data_fortrain/data_fortrain_1031version/TTS/zh/emilia/Emilia/ZH/ZH-B000197.tar", "speech_model_backup/stevehguo/data_fortrain/data_fortrain_1031version/TTS/zh/emilia/Emilia/ZH/ZH-B000220.tar", "speech_model_backup/stevehguo/data_fortrain/data_fortrain_1031version/TTS/zh/emilia/Emilia/ZH/ZH-B000245.tar", "speech_model_backup/stevehguo/data_fortrain/data_fortrain_1031version/TTS/zh/emilia/Emilia/ZH/ZH-B000316.tar", "speech_model_backup/stevehguo/data_fortrain/data_fortrain_1031version/TTS/zh/emilia/Emilia/ZH/ZH-B000338.tar", "speech_model_backup/stevehguo/data_fortrain/data_fortrain_1031version/TTS/zh/emilia/Emilia/ZH/ZH-B000340.tar", "speech_model_backup/stevehguo/data_fortrain/data_fortrain_1031version/TTS/zh/emilia/Emilia/ZH/ZH-B000351.tar", "speech_model_backup/stevehguo/data_fortrain/data_fortrain_1031version/TTS/zh/emilia/Emilia/ZH/ZH-B000389.tar", "speech_model_backup/stevehguo/data_fortrain/data_fortrain_1031version/TTS/zh/emilia/Emilia/ZH/ZH-B000400.tar", "speech_model_backup/stevehguo/data_fortrain/data_fortrain_1031version/TTS/zh/emilia/Emilia/ZH/ZH-B000403.tar", "speech_model_backup/stevehguo/data_fortrain/data_fortrain_1031version/TTS/zh/emilia/Emilia/ZH/ZH-B000430.tar", "speech_model_backup/stevehguo/data_fortrain/data_fortrain_1031version/TTS/zh/emilia/Emilia/ZH/ZH-B000492.tar", "speech_model_backup/stevehguo/data_fortrain/data_fortrain_1031version/TTS/zh/emilia/Emilia/ZH/ZH-B000724.tar", "speech_model_backup/stevehguo/data_fortrain/data_fortrain_1031version/TTS/zh/emilia/Emilia/ZH/ZH-B000892.tar"]''')
KILL_NOW = json.loads('''[0]''')   # 当前项已在其他节点完成（重复处理）→ 立即停止
ARCH = f'{R}/logs/archive_sweep'
os.makedirs(ARCH, exist_ok=True)


def h(k):
    return hashlib.md5(k.encode()).hexdigest()[:16]


def log(m):
    print(time.strftime('%m-%d %H:%M:%S'), m, flush=True)


def pid_of(w):
    for d in os.listdir('/proc'):
        if not d.isdigit():
            continue
        try:
            c = [x.decode(errors='ignore') for x in open(f'/proc/{d}/cmdline', 'rb').read().split(b'\0')]
        except OSError:
            continue
        if any(x.endswith('tts_omni_driver.py') for x in c) and '--worker-id' in c:
            i = c.index('--worker-id')
            if i + 1 < len(c) and c[i + 1] == str(w):
                return int(d)
    return None


n = 0
for k in MOVED:
    p = f'{R}/done/{h(k)}.done'
    if not os.path.exists(p):
        with open(p, 'w') as f:
            f.write('MOVED_TO_B:' + k)
        n += 1
log(f'写入占位完成标记 {n}/{len(MOVED)}；监视 {len(CUR)} 个 worker')


def stop(w, why):
    pid = pid_of(w)
    if pid is None:
        log(f'w{w} 进程已不在（{why}）')
        return
    os.kill(pid, 9)
    try:
        shutil.copy(f'{R}/logs/worker-{w}.log', f'{ARCH}/worker-{w}.log')
    except OSError:
        pass
    log(f'w{w} 已停止 pid={pid}（{why}）')


for w in KILL_NOW:
    stop(w, '当前项已在其他节点完成，重复处理')

pending = dict(CUR)
while pending:
    for w, k in list(pending.items()):
        if pid_of(w) is None:
            log(f'w{w} 进程已不在，移出监视')
            pending.pop(w)
            continue
        if os.path.exists(f'{R}/done/{h(k)}.done'):
            stop(w, f'当前项 {k.rsplit("/", 1)[-1]} 已完成，剩 {len(pending) - 1}')
            pending.pop(w)
    time.sleep(15)
log('全部 worker 已停止')
