"""Qwen3.5 thinker 的定长缓存增量前向 + CUDA Graph（仅推理）。

`qwen3_5_compat.py` 的增量缓存（`Qwen3NextDynamicCache`）每步拼接 KV、替换状态张量，地址和形状都在变，
录不了 CUDA Graph；而 4B thinker 单次增量前向要发起 2000+ 个 kernel，墙钟里约 70～80% 是 CPU 发射开销
（实测：1 个 token 38.6ms 中 GPU 计算 ≤ 11.8ms）。这里把全部状态放进预分配的定长缓冲区、原地更新：

  * 全注意力层（8 层）：KV 缓冲 (1, n_kv, max_len, head_dim)，按位置 `index_copy_` 写入；
    注意力对整条缓冲做 sdpa，掩码 = 「键位置 <= 查询位置」，由设备上的位置张量现算（不回读 CPU）。
  * 线性注意力层（24 层）：卷积尾部 (1, conv_dim, kernel-1) 与 delta rule 递归状态 (1, H, dk, dv) fp32，
    每步 `copy_` 回原缓冲。短输入（<= `RECURRENT_MAX_LEN`）用逐步递归内核，长输入（首块 prefill）用 chunk 内核。

于是同一个输入长度的前向每次发起的 kernel 序列和张量地址都相同，可以录一次、之后重放
（流式同传里长度 1 / 9 / 10 覆盖 97.6% 的前向：逐 token 译文与决策、`<|chunk|>`+8 帧音频、再加上一步的 WAIT）。
没录图的长度走同一个 `_forward`（eager），状态完全共用。

层的参数直接取自 `Qwen3_5CompatTextModel`，不复制权重。数值与 `qwen3_5_compat` 的整段前向只差
bf16 舍入与内核差异，见 `tests/test_qwen3_5_compat.py::test_static_runner_*`。
"""
import torch
import torch.nn.functional as F

from transformers.models.qwen3_next import modeling_qwen3_next as _qn

RECURRENT_MAX_LEN = 16          # <= 此长度的线性注意力用逐步递归内核（精确、无 chunk 内的 TF32 近似）


class Qwen3_5StaticRunner:
    """Qwen3.5 文本塔的定长缓存增量前向。

    用法（每句话开始前 `reset()`；同一个 runner 同一时刻只服务一条序列）：

        runner = Qwen3_5StaticRunner(model.thinker, max_len=2048, graph_lens=(1, 9, 10))
        runner.reset()
        h = runner(embeds)          # (1, L, D) -> 末层（已过 final norm）hidden (1, L, D)，L 任意

    `graph_lens` 只在 CUDA 上生效；首次用到某个长度时预热并录图（录图前后自动清零状态，
    所以必须在 `reset()` 之后、序列开始前触发，见 `prepare()`）。
    返回的张量来自图的输出缓冲，下次调用会被覆盖；需要保留请 `clone()`。
    """

    def __init__(self, text_model, max_len=2048, graph_lens=(1, 9, 10)):
        self.m = text_model
        cfg = text_model.config
        p = next(text_model.parameters())
        self.device, self.dtype = p.device, p.dtype
        self.max_len = int(max_len)
        self.graph_lens = tuple(int(x) for x in graph_lens) if self.device.type == 'cuda' else ()
        self.pos = 0
        self.kv, self.conv, self.rec = {}, {}, {}
        for i, layer in enumerate(text_model.layers):
            if layer.layer_type == 'full_attention':
                shape = (1, cfg.num_key_value_heads, self.max_len, cfg.head_dim)
                self.kv[i] = (torch.zeros(shape, device=self.device, dtype=self.dtype),
                              torch.zeros(shape, device=self.device, dtype=self.dtype))
            else:
                la = layer.linear_attn
                self.conv[i] = torch.zeros(1, la.conv_dim, la.conv_kernel_size - 1,
                                           device=self.device, dtype=self.dtype)
                self.rec[i] = torch.zeros(1, la.num_v_heads, la.head_k_dim, la.head_v_dim,
                                          device=self.device, dtype=torch.float32)
        self.key_pos = torch.arange(self.max_len, device=self.device)
        self.graphs = {}            # L -> (graph, x_buf, pos_buf, out_buf)

    # ------------------------------------------------------------------ 状态
    def reset(self):
        with torch.inference_mode():
            for k, v in self.kv.values():
                k.zero_(); v.zero_()
            for t in list(self.conv.values()) + list(self.rec.values()):
                t.zero_()
        self.pos = 0

    @torch.inference_mode()
    def prepare(self):
        """为全部 `graph_lens` 录图（会清零状态）。在每个新 runner 第一次使用前调一次即可。"""
        for n in self.graph_lens:
            self._capture(n)
        self.reset()

    # ------------------------------------------------------------------ 层
    def _attn(self, i, layer, x, cos, sin, cpos):
        a = layer.self_attn
        B, L, _ = x.shape
        hd = a.head_dim
        q, gate = torch.chunk(a.q_proj(x).view(B, L, -1, hd * 2), 2, dim=-1)
        gate = gate.reshape(B, L, -1)
        q = a.q_norm(q).transpose(1, 2)
        k = a.k_norm(a.k_proj(x).view(B, L, -1, hd)).transpose(1, 2)
        v = a.v_proj(x).view(B, L, -1, hd).transpose(1, 2)
        q, k = _qn.apply_rotary_pos_emb(q, k, cos, sin)
        kb, vb = self.kv[i]
        kb.index_copy_(2, cpos, k)
        vb.index_copy_(2, cpos, v)
        mask = (self.key_pos[None, :] <= cpos[:, None])[None, None]            # (1,1,L,max_len)
        o = F.scaled_dot_product_attention(q, kb, vb, attn_mask=mask, scale=a.scaling,
                                           enable_gqa=kb.size(1) != q.size(1))
        o = o.transpose(1, 2).reshape(B, L, -1) * torch.sigmoid(gate)
        return a.o_proj(o)

    def _linear(self, i, la, x):
        B, L, _ = x.shape
        mixed = la.in_proj_qkv(x).transpose(1, 2)                              # (B, C, L)
        full = torch.cat([self.conv[i], mixed], dim=-1)
        self.conv[i].copy_(full[:, :, -(la.conv_kernel_size - 1):])
        mixed = F.silu(F.conv1d(full, la.conv1d.weight, la.conv1d.bias, groups=la.conv_dim)).transpose(1, 2)
        z = la.in_proj_z(x).reshape(B, L, -1, la.head_v_dim)
        beta = la.in_proj_b(x).sigmoid()
        g = -la.A_log.float().exp() * F.softplus(la.in_proj_a(x).float() + la.dt_bias)
        q, k, v = torch.split(mixed, [la.key_dim, la.key_dim, la.value_dim], dim=-1)
        q = q.reshape(B, L, -1, la.head_k_dim)
        k = k.reshape(B, L, -1, la.head_k_dim)
        v = v.reshape(B, L, -1, la.head_v_dim)
        rep = la.num_v_heads // la.num_k_heads
        if rep > 1:
            q = q.repeat_interleave(rep, dim=2)
            k = k.repeat_interleave(rep, dim=2)
        cuda = x.is_cuda
        if L <= RECURRENT_MAX_LEN:
            rule = _qn.fused_recurrent_gated_delta_rule if (cuda and _qn.fused_recurrent_gated_delta_rule) \
                else _qn.torch_recurrent_gated_delta_rule
        else:
            rule = _qn.chunk_gated_delta_rule if (cuda and _qn.chunk_gated_delta_rule) \
                else _qn.torch_chunk_gated_delta_rule
        core, state = rule(q, k, v, g=g, beta=beta, initial_state=self.rec[i], output_final_state=True,
                           use_qk_l2norm_in_kernel=True)
        self.rec[i].copy_(state)
        core = la.norm(core.reshape(-1, la.head_v_dim), z.reshape(-1, la.head_v_dim))
        return la.out_proj(core.reshape(B, L, -1))

    def _forward(self, x, cpos):
        """x: (1, L, D)；cpos: (L,) long，设备上的绝对位置。原地更新全部状态，返回末层 hidden。"""
        cos, sin = self.m.rotary_emb(x, cpos[None])
        h = x
        for i, layer in enumerate(self.m.layers):
            r = h
            y = layer.input_layernorm(h)
            y = self._linear(i, layer.linear_attn, y) if layer.layer_type == 'linear_attention' \
                else self._attn(i, layer, y, cos, sin, cpos)
            h = r + y
            h = h + layer.mlp(layer.post_attention_layernorm(h))
        return self.m.norm(h)

    # ------------------------------------------------------------------ 图
    def _capture(self, n):
        if n in self.graphs:
            return
        D = self.m.config.hidden_size
        xb = torch.zeros(1, n, D, device=self.device, dtype=self.dtype)
        pb = torch.arange(n, device=self.device)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):                 # 预热：triton 自动调优、cuBLAS 选算法
            for _ in range(3):
                self._forward(xb, pb)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = self._forward(xb, pb)
        self.graphs[n] = (g, xb, pb, out)

    @torch.inference_mode()
    def __call__(self, x):
        n = int(x.size(1))
        if self.pos + n > self.max_len:
            raise RuntimeError(f'thinker 序列超过缓存上限 {self.max_len}（已有 {self.pos}，追加 {n}）')
        if n in self.graph_lens:
            if n not in self.graphs:
                raise RuntimeError(f'长度 {n} 的图还没录：请在 reset() 之前调用 prepare()')
            g, xb, pb, out = self.graphs[n]
            xb.copy_(x)
            torch.add(self.key_pos[:n], self.pos, out=pb)
            g.replay()
        else:
            out = self._forward(x.to(self.dtype), self.key_pos[self.pos:self.pos + n])
        self.pos += n
        return out
