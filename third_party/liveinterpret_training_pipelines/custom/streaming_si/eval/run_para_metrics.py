"""副语言指标：音色相似度（SIM-o）、情感相似度（ESim）、韵律一致性（AutoPCP）。

三项都是「源语音 vs 生成语音」的配对打分，用来看翻译后保留了多少说话人与风格信息。
同时给出**随机配对**基线：同一数据集内把生成语音与另一句的源语音配对（固定种子的循环错位），
因为这些指标在随机配对下本身就不低（emotion2vec 余弦随机配对约 0.82～0.90），只有
「配对 − 随机」才说明信息是真的被迁移了。

| 指标 | 模型 | 口径 |
|---|---|---|
| **SIM-o**（音色相似度） | `microsoft/wavlm-base-plus-sv`（WavLM + x-vector 说话人验证头） | 两段语音 embedding 的余弦；TTS / S2ST 文献里 SIM-o 的常用口径，与声码器自己用的 CAM++ 不是同一个模型，避免自说自话 |
| **ESim**（情感相似度） | `emotion2vec_plus_large`（funasr，`extract_embedding`，取 utterance 级向量） | 余弦 |
| **AutoPCP**（韵律一致性） | Meta `AutoPCP-multilingual-v2` + `facebook/wav2vec2-large-xlsr-53` 第 9 层特征（stopes 官方实现） | 模型直接预测 1～5 的韵律一致性分（为拟合人工 PCP 评分训练），跨语言可用 |

所有音频统一转 16kHz 单声道；长音频截断到 `--max_sec`（默认 30s，三类模型都是短句训练）。

用法：python run_para_metrics.py --results r.jsonl --manifest all.jsonl --out para.jsonl \
          --wavlm <dir> --emotion2vec <dir> --autopcp <model.config> [--shard i --num_shards n]
输出每行：`{id, sim_o, esim, autopcp, sim_o_rand, esim_rand, autopcp_rand}`。
"""
import argparse
import json
import os

import numpy as np
import soundfile as sf
import torch
import torchaudio

SR = 16000


def load16k(path, max_sec):
    x, sr = sf.read(path, dtype='float32', always_2d=True)
    w = torch.from_numpy(x.mean(1))
    if sr != SR:
        w = torchaudio.functional.resample(w, sr, SR)
    return w[:int(max_sec * SR)].contiguous()


class SimO:
    """WavLM + x-vector 说话人 embedding 余弦。"""

    def __init__(self, path, device):
        from transformers import AutoFeatureExtractor, WavLMForXVector
        self.fe = AutoFeatureExtractor.from_pretrained(path)
        self.m = WavLMForXVector.from_pretrained(path).to(device).eval()
        self.device = device

    @torch.inference_mode()
    def emb(self, w):
        iv = self.fe(w.numpy(), sampling_rate=SR, return_tensors='pt', padding=True)
        iv = {k: v.to(self.device) for k, v in iv.items()}
        return torch.nn.functional.normalize(self.m(**iv).embeddings[0].float(), dim=-1).cpu()


class ESim:
    """emotion2vec utterance 级 embedding 余弦。"""

    def __init__(self, path, device):
        from funasr import AutoModel
        self.m = AutoModel(model=path, device=device, disable_update=True, disable_pbar=True,
                           disable_log=True)

    def emb(self, path):
        r = self.m.generate(path, granularity='utterance', extract_embedding=True)
        v = torch.from_numpy(np.asarray(r[0]['feats'], dtype=np.float32)).reshape(-1)
        return torch.nn.functional.normalize(v, dim=-1)


class AutoPCP:
    """stopes 官方 AutoPCP：wav2vec2-xlsr 特征 + comparator，输出韵律一致性分。

    注意输出**不是 1~4 / 1~5 的量表分**，而是未过激活的 logit，见 `scores()` 的说明。

    直接用官方 `compare_audio_pairs`（默认 `pick_layer=7`、avg_pool、`symmetrize=False`），
    只把编码器与 comparator 复用一次，避免逐句重载。
    """

    def __init__(self, config_path, w2v_path, device):
        from stopes.eval.auto_pcp.audio_comparator import Comparator, encode_audios, get_model_pred
        from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
        self.encode_audios, self.get_model_pred = encode_audios, get_model_pred
        self.use_cuda = device.startswith('cuda')
        self.cmp = Comparator.load(config_path, use_gpu=self.use_cuda)
        self.fex = Wav2Vec2FeatureExtractor.from_pretrained(w2v_path)
        self.enc = Wav2Vec2Model.from_pretrained(w2v_path)
        if self.use_cuda:
            self.enc.cuda()
        self.enc.eval()

    # 官方 AutoPCP_multilingual_v2.yaml 的绑定参数（不是 compare_audio_pairs 的函数默认值！）：
    #   pick_layer: 9    symmetrize: true
    PICK_LAYER = 9

    def emb(self, paths, batch_size=8):
        return self.encode_audios(paths, fex=self.fex, model=self.enc, batch_size=batch_size,
                                  pick_layer=self.PICK_LAYER, use_cuda=self.use_cuda,
                                  progress=False, num_process=1)[:, 0]

    def scores(self, src_emb, tgt_emb, batch_size=8):
        """PCP 分 = 对称化后的 comparator 输出第 0 维，口径对齐官方 v2 config。

        刻度：comparator 是 MLP 回归器，用 `F.mse_loss` 拟合人工 PCP 标签
        （`comparator_training.py`；`contrastive_training.py` 的 `expected_mean=2.5` 即 1~4 中点），
        官方 README 明确「score 接近 4 表示高度相似，很不相似的对接近 1」。
        所以**该值就在 1~4 刻度上**，无需归一化；末层 `output_act=False` 只意味着可以外推越界
        （<1 或 >4），不代表它不在这个刻度上。

        对称化：官方 v2 config `symmetrize: true`，即 (pred(src,mt) + pred(mt,src)) / 2。
        """
        def pred(a, b):
            return self.get_model_pred(self.cmp, src=a.clone(), mt=b.clone(),
                                       use_gpu=self.use_cuda, batch_size=batch_size)[:, 0]

        p = (pred(src_emb, tgt_emb) + pred(tgt_emb, src_emb)) / 2
        return p.cpu().tolist()


def run_one(results_p, out_p, man, simo, pcp, esim, args):
    """给一个 results.jsonl 打分并写出。模型由调用方持有，便于一次加载跑多个作业。"""
    rows = [json.loads(l) for l in open(results_p, encoding='utf-8') if l.strip()]
    # 随机配对基线：同数据集内换一条**别的 talk**的源语音（同一 talk 往往是同一说话人，
    # 换同 talk 的句子测不出音色差异）。按 (talk, id) 排序后取「半程偏移」处、必要时往后找到
    # 第一条不同 talk 的，完全确定、可复现。
    # 注意：配对在**分片之前**基于全量 rows 计算，所以分片数不影响配对结果。
    by_ds = {}
    for r in rows:
        m = man[r['id']]
        by_ds.setdefault(m['dataset'], []).append((m['talk'], r['id']))
    rand_src = {}
    for items in by_ds.values():
        items.sort()
        n = len(items)
        for k, (talk, uid) in enumerate(items):
            j = (k + n // 2) % n
            for step in range(n):
                t2, u2 = items[(j + step) % n]
                if t2 != talk:
                    rand_src[uid] = u2
                    break
            else:                                  # 整个数据集只有一个 talk：退回另一句
                rand_src[uid] = items[(k + 1) % n][1]

    rows = [r for i, r in enumerate(rows) if i % args.num_shards == args.shard]
    todo, recs = [], {}
    for r in rows:
        uid = r['id']
        gen = r.get('wav')
        if not gen or not os.path.isfile(gen) or (r.get('output_sec') or 0) <= 0.05:
            recs[uid] = dict(id=uid, note='no_speech')
        else:
            recs[uid] = dict(id=uid)
            todo.append((uid, gen))

    B = args.batch
    for s in range(0, len(todo), B):
        batch = todo[s:s + B]
        try:
            gens = [g for _, g in batch]
            srcs = [man[u]['audio'] for u, _ in batch]
            rnds = [man[rand_src[u]]['audio'] for u, _ in batch]
            eg = [simo.emb(load16k(g, args.max_sec)) for g in gens]
            ea = [simo.emb(load16k(p, args.max_sec)) for p in srcs]
            er = [simo.emb(load16k(p, args.max_sec)) for p in rnds]
            pg = pa = pr = sc = sc_r = None
            if pcp is not None:
                pg, pa, pr = pcp.emb(gens, B), pcp.emb(srcs, B), pcp.emb(rnds, B)
                sc = pcp.scores(pa, pg, B)
                sc_r = pcp.scores(pr, pg, B)
            vg = va = vr = None
            if esim is not None:
                vg = [esim.emb(g) for g in gens]
                va = [esim.emb(p) for p in srcs]
                vr = [esim.emb(p) for p in rnds]
            for k, (uid, _) in enumerate(batch):
                rec = recs[uid]
                rec.update(sim_o=round(float(eg[k] @ ea[k]), 4), sim_o_rand=round(float(eg[k] @ er[k]), 4),
                           rand_ref=rand_src[uid])
                if pcp is not None:
                    rec.update(autopcp=round(float(sc[k]), 4), autopcp_rand=round(float(sc_r[k]), 4))
                if esim is not None:
                    rec.update(esim=round(float(vg[k] @ va[k]), 4), esim_rand=round(float(vg[k] @ vr[k]), 4))
        except Exception as exc:
            for uid, _ in batch:
                recs[uid]['note'] = f'{type(exc).__name__}: {exc}'
            print(f'[FAIL] {os.path.basename(out_p)} batch@{s}: {exc}', flush=True)
        print(f'[para {os.path.basename(out_p)} shard {args.shard}] '
              f'{min(s + B, len(todo))}/{len(todo)}', flush=True)

    with open(out_p, 'w', encoding='utf-8') as out:
        for r in rows:
            out.write(json.dumps(recs[r['id']], ensure_ascii=False) + '\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--results', help='单作业：结果 jsonl（与 --jobs 二选一）')
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--out', help='单作业：输出 jsonl')
    ap.add_argument('--jobs', help='批量作业清单，每行「results_path<空格>out_path」。'
                                   '模型只加载一次，顺序跑完所有作业（省掉重复加载的开销）')
    ap.add_argument('--wavlm', required=True)
    ap.add_argument('--emotion2vec', required=True)
    ap.add_argument('--skip_autopcp', action='store_true',
                    help='AutoPCP 是 gated 模型（HF 需 token），拿不到时跳过该指标')
    ap.add_argument('--autopcp', required=False)
    ap.add_argument('--w2v', required=False, help='facebook/wav2vec2-large-xlsr-53 目录（AutoPCP 的特征提取器）')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--max_sec', type=float, default=30.0)
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--num_shards', type=int, default=1)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--skip_esim', action='store_true')
    args = ap.parse_args()

    if bool(args.jobs) == bool(args.results):
        raise SystemExit('--jobs 与 --results 必须二选一')
    jobs = []
    if args.jobs:
        for l in open(args.jobs, encoding='utf-8'):
            l = l.strip()
            if l and not l.startswith('#'):
                a, b = l.split()
                jobs.append((a, b))
    else:
        if not args.out:
            raise SystemExit('--results 需要配 --out')
        jobs.append((args.results, args.out))

    man = {}
    for l in open(args.manifest, encoding='utf-8'):
        if l.strip():
            r = json.loads(l)
            man[r['id']] = r

    simo = SimO(args.wavlm, args.device)
    pcp = None if args.skip_autopcp else AutoPCP(args.autopcp, args.w2v, args.device)
    esim = None if args.skip_esim else ESim(args.emotion2vec, args.device)

    for k, (res_p, out_p) in enumerate(jobs, 1):
        if not os.path.isfile(res_p):
            print(f'[skip] {res_p} 不存在', flush=True)
            continue
        os.makedirs(os.path.dirname(out_p) or '.', exist_ok=True)
        print(f'=== [{k}/{len(jobs)}] {res_p} -> {out_p}', flush=True)
        run_one(res_p, out_p, man, simo, pcp, esim, args)
    print(f'PARA_ALL_DONE shard={args.shard} jobs={len(jobs)}', flush=True)


if __name__ == '__main__':
    main()
