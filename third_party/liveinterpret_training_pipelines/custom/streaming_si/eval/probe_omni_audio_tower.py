"""探测 **SimulS2ST-Omni / 我们的复现** 的音频编码器（`thinker.audio_tower`）各层韵律 R²，
与我们 Voxtral 各层的实测值对比（`paralinguistic_preservation.md` §5.1）。

    # 官方离线系统（offline/ 已是 merged 权重，audio_tower 含微调）
    python probe_omni_audio_tower.py extract --variant omni_off --out <工作目录>
    # 官方流式系统（offline/ 作基座 + simuls2st_adapter/thinker_lora）
    python probe_omni_audio_tower.py extract --variant omni_simul --out <工作目录>
    # 我们的复现（stage3v2 的 thinker_lora，基座同为官方 offline/）
    python probe_omni_audio_tower.py extract --variant repro --out <工作目录>
    # 出 R² 表（纯 CPU，可一次汇总多个 variant）
    python probe_omni_audio_tower.py probe --out <工作目录>

**必须测带微调的权重，不能用 raw 基座** —— 实测证据：
* `simuls2st_adapter/thinker_lora` 里有 **192 个 `audio_tower` 张量**
  （32 层 × q/k/v × lora_A/B），说明 Omni **微调了音频编码器**；
* `offline/thinker.audio_tower.layers.0.self_attn.q_proj.weight` 与
  `qwen2.5-omni-3b-raw` 的同名张量**不相同**（最大绝对差 1.95e-2），
  即 `offline/` 已是 merged 权重（官方 README：*Fully merged OmniTalker model*）；
* 我们复现的 `stage2_multitask_full` / `stage3v2_fluency_retrain` 的 `thinker_lora`
  同样各有 192 个 `audio_tower` 张量。

组合关系按官方 README（L53-55）：离线用 `offline/`；流式用 `offline/` 作基座、
`simuls2st_adapter/` 作适配器。

**口径与现有探针逐项对齐**，否则 R² 不可比：
* **同一批样本**：复用 `paraling_probe.py` 的 `prosody.pkl`（machine_0 的
  `fsq-train-00000` = eng→cmn、machine_1 的 `fsq-train-00008` = cmn→eng），
  并按其 `extract_*.npz` 的 `ok` 掩码取**完全相同**的子集；
* **同样的 7 项韵律目标**：`KEYS[1:]`，刻意剔除 `lf0_mean`（F0 均值主要随说话人变化，
  由 CAM++ spk_emb 负责，不属于要探测的副语言）；
* **同样的池化**：时间轴 mean 与 std 拼接（`_pool`），维度 = 2 × hidden；
* **同样的回归**：`RidgeCV(alphas=logspace(-1,6,15), alpha_per_target=True)`
  + `StandardScaler` + `KFold(5, shuffle, seed=0)`，R² 取折外预测。

**读数时必须考虑的限制**：
1. SimulS2ST-Omni 的底座是 **Qwen2.5-Omni**（`Qwen2_5OmniForConditionalGeneration`，
   官方 `src/inference/infer_omni_talker_tts.py` L276 等），`audio_tower` 是
   **Whisper 系** 32 层 / d=1280 编码器；**不是** Qwen3-Omni 的 AuT（那是另一个项目
   `custom/qwen_omnispeak` 用的 30B-A3B）。两者不可混称。
2. 跨模型比 R² 只反映"该表示里**线性可解码**的韵律信息量"，**不等于**下游副语言保留能力。
   我们自己的数据就是反例：Voxtral L4 有 0.286，但到 thinker 译文位置末层只剩 0.024。
   端到端判据是 AutoPCP delta。
3. 两侧 mel 前端与帧率不同（池化到定长后帧率差异被吸收，但前端差异无法完全对齐）。
"""
import argparse
import os
import pickle
import sys

import numpy as np

# 与 paraling_probe.py 完全一致的键定义（顺序也要一致，probe 阶段按下标取）
KEYS = ['lf0_mean', 'lf0_std', 'final_rise', 'en_mean', 'en_std', 'rate', 'pause', 'voiced']
NAMES = {'lf0_mean': 'F0均值', 'lf0_std': 'F0起伏', 'final_rise': '句末升降', 'en_mean': '能量均值',
         'en_std': '能量起伏', 'rate': '语速', 'pause': '停顿占比', 'voiced': '有声段占比'}
PROSODY = KEYS[1:]

_OMNI = '/fsx/stevehguo/eval/models/SimulS2ST-Omni'
_REPRO = '/fsx/stevehguo/eval/models/repro'
# variant -> (基座目录, thinker LoRA 目录或 None)
# `offline/` 是官方「Fully merged OmniTalker」，audio_tower 已含微调（与 raw 差 1.95e-2）；
# 流式系统与我们的复现都在它之上再挂 thinker_lora（其中含 192 个 audio_tower 张量）。
VARIANTS = {
    'omni_off':   (f'{_OMNI}/offline', None),
    'omni_simul': (f'{_OMNI}/offline', f'{_OMNI}/simuls2st_adapter/thinker_lora'),
    'repro':      (f'{_OMNI}/offline', f'{_REPRO}/stage3v2_fluency_retrain/thinker_lora'),
    'repro_s2':   (f'{_OMNI}/offline', f'{_REPRO}/stage2_multitask_full/thinker_lora'),
    # 仅作对照：未微调的原始基座，用来量化「Omni 的微调把音频编码器改动了多少」
    'raw':        ('/fsx/stevehguo/eval/models/qwen2.5-omni-3b-raw', None),
}


def _pool(x):
    """时间轴 mean 与 std 拼接 —— 与 paraling_probe.py 的 `_pool` 逐字相同。"""
    import torch
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    return np.concatenate([x.mean(0).float().cpu().numpy(),
                           x.std(0, unbiased=False).float().cpu().numpy()]).astype(np.float16)


def _cv_r2(X, Y, seed=0):
    """折外 R²；与 paraling_probe.py 的 `_cv_r2` 逐字相同。"""
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler
    X = X.astype(np.float64)
    pred = np.zeros_like(Y)
    for tr, te in KFold(5, shuffle=True, random_state=seed).split(X):
        sx = StandardScaler().fit(X[tr])
        sy = StandardScaler().fit(Y[tr])
        m = RidgeCV(alphas=np.logspace(-1, 6, 15), alpha_per_target=True).fit(
            sx.transform(X[tr]), sy.transform(Y[tr]))
        pred[te] = sy.inverse_transform(m.predict(sx.transform(X[te])).reshape(len(te), -1))
    return 1 - ((Y - pred) ** 2).sum(0) / ((Y - Y.mean(0)) ** 2).sum(0)


def read_tar_wav(stage_dir, ref):
    """按 `<tar 名>::<偏移>:<长度>` 从 tar 里直接取出一段音频。

    与 `paraling_probe.py` 的读取方式相同：源音频打包在 tar 内，用字节范围定位，
    避免解包 30GB。`prosody.pkl` 里记的 `src_dir` 是当时 Pod 的挂载路径
    （`/home/coder/workspace/...`），本脚本不用它，改由 `--stage_dir` 显式给。
    """
    import io
    import soundfile as sf
    name, loc = ref.split('::', 1)
    off, ln = (int(x) for x in loc.split(':'))
    with open(os.path.join(stage_dir, name), 'rb') as fp:
        fp.seek(off)
        wav, sr = sf.read(io.BytesIO(fp.read(ln)), dtype='float32')
    if wav.ndim > 1:
        wav = wav.mean(1)
    return wav, sr


def cmd_extract(args):
    import torch
    import librosa

    ref = args.ref_probe          # 现有探针的产出目录（取 prosody.pkl 与 ok 掩码）
    with open(os.path.join(ref, 'prosody.pkl'), 'rb') as f:
        data = pickle.load(f)

    from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
    base, lora = VARIANTS[args.variant]
    print(f'variant={args.variant}  基座={base}  thinker LoRA={lora or "（无，offline 已 merged）"}',
          flush=True)
    proc = Qwen2_5OmniProcessor.from_pretrained(base)
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        base, torch_dtype=torch.bfloat16, device_map=None).to(args.device)
    if lora:
        # 与官方推理一致：LoRA 挂在 thinker 上（`infer_omni_talker_tts.py` L371）。
        # merge 之后再探针，避免 PEFT 包装层影响 hidden_states 的取法。
        from peft import PeftModel
        model.thinker = PeftModel.from_pretrained(model.thinker, lora)
        n_at = sum(1 for n, _ in model.thinker.named_parameters()
                   if 'audio_tower' in n and 'lora_' in n)
        print(f'  已挂 LoRA，其中 audio_tower 上的 lora 参数 {n_at} 个', flush=True)
        if n_at == 0:
            raise SystemExit('LoRA 未命中 audio_tower —— 口径不对，停止（预期 192 个张量）')
        model.thinker = model.thinker.merge_and_unload()
    model.eval()
    tower = model.thinker.audio_tower
    n_layers = len(tower.layers)
    print(f'audio_tower: {n_layers} 层, hidden={tower.config.d_model}', flush=True)
    # 与 Voxtral 一侧同样的层间隔；末层单列
    layers = sorted(set([4, 8, 12, 16, 20, 24, 28, n_layers] if n_layers >= 28
                        else list(range(4, n_layers + 1, 4)) + [n_layers]))
    print(f'取层：{layers}', flush=True)
    del model.thinker.model          # 只需要编码器，省显存
    torch.cuda.empty_cache()

    fe = proc.feature_extractor
    # 逐层 hook：按 tower.layers 的顺序把每层输出收进 cap（每句前清空）
    cap = []

    def _mk(idx):
        def _h(_m, _i, o):
            x = o[0] if isinstance(o, tuple) else o
            cap.append(x.detach()[0] if x.dim() == 3 else x.detach())
        return _h

    for li, layer in enumerate(tower.layers):
        layer.register_forward_hook(_mk(li))

    for direction, d in data.items():
        tag = direction.replace('->', '2')
        z = np.load(os.path.join(ref, f'extract_{tag}.npz'))
        ok_ref = z['ok']
        rows = d['rows']
        n = len(ok_ref)
        res = {f'aut{l}': [] for l in layers}
        ok = np.zeros(n, dtype=bool)
        for i in range(n):
            if not ok_ref[i]:            # 严格对齐：参考里失败的行一律跳过
                continue
            try:
                y, sr = read_tar_wav(args.stage_dir, rows[i]['source_wav'])
                if sr != 16000:
                    y = librosa.resample(y, orig_sr=sr, target_sr=16000)
                # 必须 return_attention_mask：mel 被 pad 到 n_samples(30000 帧)，
                # 真实帧数只能由 mask 求和得到；直接拿 input_features 的最后一维会得到
                # padding 后的长度，导致编码器内部 chunk 切分对不上（split_with_sizes 报错）。
                inp = fe(y, sampling_rate=16000, return_tensors='pt',
                         return_attention_mask=True, padding='max_length')
                nfr = int(inp['attention_mask'].sum())
                # 编码器内部按 `input_features.split(chunk_lengths, dim=1)` 切块，
                # 即它要的是 **(mel, time)** 而不是带 batch 的 (1, mel, time)。
                feats = inp['input_features'][0, :, :nfr].to(args.device, dtype=torch.bfloat16)
                flen = torch.tensor([nfr], device=args.device, dtype=torch.long)
                # aftercnn_lens 必填（官方 thinker 侧传的是 audio_feat_lengths）：
                # 用编码器自带的 `_get_feat_extract_output_lengths` 算，别自己推公式。
                # 返回 (卷积后长度, 最终输出长度)；编码器内部按**卷积后**长度分段，
                # 传最终输出长度会少一半（实测报 split_sizes 只有一半）。
                aftercnn, _ = tower._get_feat_extract_output_lengths(flen)
                cap.clear()
                with torch.no_grad():
                    tower(feats, feature_lens=flen, aftercnn_lens=aftercnn)
                if len(cap) < max(layers):
                    raise RuntimeError(f'只捕获到 {len(cap)} 层输出，需要 {max(layers)}')
                for l in layers:
                    res[f'aut{l}'].append(_pool(cap[l - 1]))   # cap[k-1] = 第 k 层输出
                ok[i] = True
            except Exception as e:               # 单句失败不影响整批
                print(f'  [{direction}] 第 {i} 句失败：{type(e).__name__}: {e}', flush=True)
            if (i + 1) % 200 == 0:
                print(f'  [{direction}] {i + 1}/{n}  成功 {ok.sum()}', flush=True)
        od = os.path.join(args.out, args.variant)
        os.makedirs(od, exist_ok=True)
        np.savez(os.path.join(od, f'aut_{tag}.npz'),
                 ok=ok, **{k: np.stack(v) for k, v in res.items() if v})
        print(f'[{direction}] 写出 {ok.sum()}/{n} 条 -> {od}/aut_{tag}.npz', flush=True)


def cmd_probe(args):
    ref = args.ref_probe
    with open(os.path.join(ref, 'prosody.pkl'), 'rb') as f:
        data = pickle.load(f)
    idx = [KEYS.index(k) for k in PROSODY]
    summary = {}
    variants = [v for v in sorted(os.listdir(args.out))
                if os.path.isdir(os.path.join(args.out, v))] if os.path.isdir(args.out) else []
    if not variants:
        print(f'{args.out} 下没有任何 variant 子目录（先跑 extract）')
        return
    for var in variants:
        for direction, d in data.items():
            tag = direction.replace('->', '2')
            f_aut = os.path.join(args.out, var, f'aut_{tag}.npz')
            if not os.path.isfile(f_aut):
                continue
            za = np.load(f_aut)
            ok = za['ok']
            # d['src'] 是 (n, 8) 的韵律特征数组（列序 = KEYS），不是 dict 列表
            src = d['src'][:len(ok)][ok]
            Y = np.asarray(src, dtype=np.float64)[:, idx]
            print(f'\n########## {var} / {direction}: {ok.sum()} 对 ##########')
            print(f'{"表示":<26}{"7项均值":>9}' + ''.join(f'{NAMES[k]:>9}' for k in PROSODY))
            names = sorted((k for k in za.files if k.startswith('aut')),
                           key=lambda t: int(t[3:]))
            for k in names:
                r2 = _cv_r2(za[k], Y)
                label = f'{var} audio_tower L{k[3:]}'
                summary.setdefault(label, []).append(r2.mean())
                print(f'{label:<26}{r2.mean():>9.3f}' + ''.join(f'{v:>9.3f}' for v in r2), flush=True)
    if summary:
        print('\n================ 两方向平均（7 项韵律 R² 均值） ================')
        for k, v in summary.items():
            print(f'{k:<26}{np.mean(v):>9.3f}')
        print('\n对比（我们的链路，paralinguistic_preservation.md §5.1 实测）：')
        print(f'{"源手工韵律特征（上限）":<26}{0.276:>9.3f}')
        print(f'{"我们 Voxtral L4（最高）":<26}{0.286:>9.3f}')
        print(f'{"我们 Voxtral L32":<26}{0.240:>9.3f}')
        print(f'{"我们 adapter / audio_proj":<26}{0.244:>9.3f} / 0.216')
        print('\n注：跨模型比 R² 只反映「线性可解码的韵律信息量」，不等于下游副语言保留能力'
              '（我们这边 Voxtral L4 0.286 到译文位置末层只剩 0.024）。端到端判据是 AutoPCP delta。')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    for name in ('extract', 'probe'):
        p = sub.add_parser(name)
        p.add_argument('--out', required=True, help='本脚本的工作目录')
        p.add_argument('--ref_probe', default='/fsx/si_s3_work/paraling/out_s4_ckpt6000',
                       help='现有探针产出目录（取 prosody.pkl 与 ok 掩码，保证同批样本）')
        if name == 'extract':
            p.add_argument('--variant', required=True, choices=sorted(VARIANTS),
                           help='要探测的系统（见 VARIANTS）')
            p.add_argument('--stage_dir', default='/fsx/si_s3_work/paraling/stage',
                           help='源音频 tar 所在目录（src_eng2cmn.tar 等）')
            p.add_argument('--device', default='cuda:0')
    args = ap.parse_args(argv)
    (cmd_extract if args.cmd == 'extract' else cmd_probe)(args)


if __name__ == '__main__':
    main()
