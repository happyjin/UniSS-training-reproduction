"""S4 训练目标与指标（迁移自源 `trainer/train_s4_stream.py` 的 train_epoch / evaluate）。

loss（与源逐式一致）::

    loss = text_w * split_total + speech_w * weighted_speech_ce + aux
    split_total = (Σ译文CE + w_dec·(Σ决策CE + Σ步边界CE)) / (n_txt + w_dec·(n_dec + n_bnd))

`split_losses` 原样拷贝自源项目（见其 docstring）。

指标全部以**逐 token 向量**输出：swift 的 MeanMetric 对向量按元素计数累加并跨卡 all_reduce，
得到的就是源 `evaluate()` 的 token 加权均值（Σloss / Σn），而不是逐 batch 均值的均值。
决策 macro-F1 不能按 token 平均（混淆矩阵的非线性函数），因此额外输出 6 个混淆矩阵单元的
逐决策 token 指示向量 `dec_cm_<真>_<预>`：它们共享同一分母 n_dec，均值即「单元计数 / n_dec」，
F1 与尺度无关，由 `install_s4_metric_patch` 在汇总后还原为 `dec_macro_f1` 等（与源
`macro_f1_from_cm` 同式），可直接用于 `--metric_for_best_model dec_macro_f1`。
"""
import torch
import torch.nn as nn

from .s4_losses import speech_token_weights, weighted_speech_ce

CM_CLASSES = ('wait', 'write')
CM_PRED = ('wait', 'write', 'other')
CM_KEYS = tuple(f'dec_cm_{t}_{p}' for t in CM_CLASSES for p in CM_PRED)


def split_losses(logits, labels, decision_ids, loss_fct, decision_weight, boundary_id=None):
    """把文本 CE 拆成「译文」/「读写决策」/「步边界」三项并按 token 数加权合并。

    （源 train_s4_stream.split_losses 原样拷贝）

    Returns: (total, translate_loss, decision_loss, dec_hit, dec_n, bnd_loss, bnd_hit, bnd_n)
    """
    flat_lg = logits.view(-1, logits.size(-1))
    flat_y = labels.view(-1)
    sup = flat_y != -100
    dec = torch.isin(flat_y, decision_ids) & sup
    bnd = ((flat_y == boundary_id) & sup) if boundary_id is not None \
        else torch.zeros_like(sup)
    txt = sup & ~dec & ~bnd

    raw = loss_fct(flat_lg, flat_y)                  # ignore_index=-100 处已是 0
    n_txt, n_dec, n_bnd = txt.sum(), dec.sum(), bnd.sum()
    t_loss = (raw * txt).sum() / n_txt.clamp(min=1)
    d_loss = (raw * dec).sum() / n_dec.clamp(min=1)
    b_loss = (raw * bnd).sum() / n_bnd.clamp(min=1)
    denom = (n_txt + decision_weight * (n_dec + n_bnd)).clamp(min=1)
    total = (t_loss * n_txt
             + decision_weight * (d_loss * n_dec + b_loss * n_bnd)) / denom

    with torch.no_grad():
        pred = flat_lg.argmax(-1)
        hit = ((pred == flat_y) & dec).sum()
        b_hit = ((pred == flat_y) & bnd).sum()
    return total, t_loss, d_loss, hit, n_dec, b_loss, b_hit, n_bnd


def compute_s4_losses(res, labels, stream_token_ids, decision_weight, text_w, speech_w,
                      speech_loss_weights=None):
    """返回 (loss, metrics)。`labels` 为已 shift 的 Y；`stream_token_ids` = [wait, W, chunk, src_end]。"""
    wait_id, write_id, chunk_id = (int(x) for x in stream_token_ids[:3])
    labels = labels.contiguous()        # 模型内部 shift 得到的是 labels[:, 1:] 视图；split_losses 用 .view
    zero = res.aux_loss * 0
    metrics = {}
    text_loss = zero
    if res.logits is not None and res.logits.numel():
        # 不做整张 logits 的 .float() 拷贝（B=48×L=640×V=151936 时约 19GB）：与源一致，
        # 精度交给 autocast 下 cross_entropy 的 fp32 内部计算。
        logits = res.logits
        decision_ids = torch.tensor([wait_id, write_id], device=labels.device)
        flat_lg = logits.view(-1, logits.size(-1))
        flat_y = labels.reshape(-1)
        per_tok_all = nn.functional.cross_entropy(flat_lg, flat_y, reduction='none', ignore_index=-100)
        text_loss = split_losses(
            logits, labels, decision_ids, lambda *_: per_tok_all, float(decision_weight), chunk_id)[0]
        with torch.no_grad():
            sup = flat_y != -100
            dec = torch.isin(flat_y, decision_ids) & sup
            bnd = (flat_y == chunk_id) & sup
            txt = sup & ~dec & ~bnd
            per_tok = per_tok_all.detach().float()
            pred = flat_lg.argmax(-1)
            hit = (pred == flat_y).float()
            for name, mask in (('text', txt), ('dec', dec), ('bnd', bnd)):
                if bool(mask.any()):
                    metrics[f'{name}_loss'] = per_tok[mask]
                    metrics[f'{name}_acc'] = hit[mask]
            if bool(dec.any()):
                yt, yp = flat_y[dec], pred[dec]
                for t_name, t_id in zip(CM_CLASSES, (wait_id, write_id)):
                    row = (yt == t_id).float()
                    metrics[f'dec_cm_{t_name}_wait'] = row * (yp == wait_id)
                    metrics[f'dec_cm_{t_name}_write'] = row * (yp == write_id)
                    metrics[f'dec_cm_{t_name}_other'] = row * ((yp != wait_id) & (yp != write_id))

    speech_loss = zero
    if res.speech_logits is not None:
        speech_loss = weighted_speech_ce(res.speech_logits, res.speech_target, speech_loss_weights)
        with torch.no_grad():
            sl, st = res.speech_logits, res.speech_target
            w = speech_token_weights(st, speech_loss_weights, torch.float32)
            keep = w > 0
            if bool(keep.any()):
                per_tok = nn.functional.cross_entropy(
                    sl.float().transpose(1, 2), st, reduction='none', ignore_index=-100)
                top = sl.topk(min(10, sl.size(-1)), dim=-1).indices
                hit = top.eq(st.unsqueeze(-1))
                metrics['speech_loss'] = per_tok[keep]
                metrics['speech_acc'] = hit[..., 0][keep].float()
                metrics['speech_acc10'] = hit.any(-1)[keep].float()

    loss = float(text_w) * text_loss + float(speech_w) * speech_loss + res.aux_loss
    return loss, metrics


def macro_f1_from_cm(c):
    """从 2x3 混淆矩阵（行=真实 wait/write，列=预测 wait/write/other）算 per-class F1。

    （源 train_s4_stream.macro_f1_from_cm 原样拷贝；输入可以是计数，也可以是同一分母下的比例）
    """
    out, f1s = {}, []
    for i, name in enumerate(('wait', 'write')):
        tp = c[i][i]
        fn = sum(c[i]) - tp          # 误判成另一类 + 预测成 other，都算漏检
        fp = c[1 - i][i]             # 另一类被误判成本类
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        f1s.append(f1)
        out[f'{name}_f1'] = f1
        out[f'{name}_prec'] = prec
        out[f'{name}_recall'] = rec
        out[f'n_{name}'] = int(sum(c[i]))
    n = max(sum(sum(r) for r in c), 1)
    out['dec_macro_f1'] = sum(f1s) / 2
    out['dec_pred_other'] = (c[0][2] + c[1][2]) / n
    return out


def finalize_decision_metrics(logs, prefix=''):
    """把汇总后的 `dec_cm_*` 比例还原成 F1 指标（原地修改 logs）。"""
    keys = [f'{prefix}{k}' for k in CM_KEYS]
    if not all(k in logs for k in keys):
        for k in keys:
            logs.pop(k, None)
        return logs
    vals = [float(logs.pop(k)) for k in keys]
    cm = [vals[0:3], vals[3:6]]
    f1 = macro_f1_from_cm(cm)
    for name in ('dec_macro_f1', 'wait_f1', 'write_f1', 'wait_recall', 'write_recall', 'dec_pred_other'):
        logs[f'{prefix}{name}'] = round(float(f1[name]), 6)
    return logs


_PATCHED = '_streaming_si_s4_metric_patch'


def install_s4_metric_patch():
    """包一层 swift `compute_custom_metrics`：把 dec_cm_* 还原为 macro-F1（幂等）。"""
    from swift.trainers.mixin import SwiftMixin
    if getattr(SwiftMixin, _PATCHED, False):
        return
    original = SwiftMixin.compute_custom_metrics

    def compute_custom_metrics(metrics, key_prefix=''):
        return finalize_decision_metrics(original(metrics, key_prefix), key_prefix)

    SwiftMixin.compute_custom_metrics = staticmethod(compute_custom_metrics)
    setattr(SwiftMixin, _PATCHED, True)
