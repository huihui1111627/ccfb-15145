"""小型 Transformer 语言模型的张量并行实现。

数据流（SP 始终开启，序列轴为第 0 维）：
  词嵌入（all-reduce + reduce-scatter）
  -> [LN -> Dropout -> 列切QKV -> 按头注意力 -> 行切O -> 残差+Dropout
      LN -> Dropout -> 列切FC1 -> GELU -> Dropout -> 行切FC2 -> 残差+Dropout] x L
  -> 末层 LN
  -> 词表切输出 logits（共享嵌入权重）-> 分片交叉熵
每个 SP 段（LN/Dropout/GELU/残差/按头注意力内部）都不通信；
通信只发生在列/行切线性的配对边界与词表操作上。
"""
from __future__ import annotations

import numpy as np

from .layers import (ColumnLinear, Dropout, LayerNorm, MultiHeadAttention,
                     RowLinear, gelu_backward, gelu_forward)
from .mesh import DeviceMesh
from .module import Module
from .tensor import split_axis
from .vocab import ParallelCrossEntropy, TiedVocab


class TransformerBlock(Module):
    def __init__(self, mesh: DeviceMesh, tag: str, cfg: dict, params: dict,
                 idx: int):
        super().__init__(mesh, tag)
        h, heads, ff = cfg["hidden"], cfg["heads"], cfg["ff"]
        p = mesh.world_size
        self.ln1 = self.add_module(LayerNorm(
            mesh, f"{tag}.ln1", h,
            params[f"blocks.{idx}.ln1.gamma"],
            params[f"blocks.{idx}.ln1.beta"]))
        self.drop1 = self.add_module(Dropout(
            mesh, f"{tag}.drop1", cfg["dropout"], "local"))
        qkv_w = split_axis(params[f"blocks.{idx}.attn.qkv.weight"],
                           1, p)
        qkv_b = split_axis(params[f"blocks.{idx}.attn.qkv.bias"], 0, p)
        o_w = split_axis(params[f"blocks.{idx}.attn.out.weight"], 0, p)
        self.attn = self.add_module(MultiHeadAttention(
            mesh, f"{tag}.attn", h, heads, qkv_w, qkv_b, o_w))
        self.drop_attn = self.add_module(Dropout(
            mesh, f"{tag}.drop_attn", cfg["dropout"], "local"))

        self.ln2 = self.add_module(LayerNorm(
            mesh, f"{tag}.ln2", h,
            params[f"blocks.{idx}.ln2.gamma"],
            params[f"blocks.{idx}.ln2.beta"]))
        self.drop2 = self.add_module(Dropout(
            mesh, f"{tag}.drop2", cfg["dropout"], "local"))
        fc1_w = split_axis(params[f"blocks.{idx}.mlp.fc1.weight"], 1, p)
        fc1_b = split_axis(params[f"blocks.{idx}.mlp.fc1.bias"], 0, p)
        self.fc1 = self.add_module(ColumnLinear(
            mesh, f"{tag}.fc1", fc1_w, fc1_b))
        self.drop_gelu = self.add_module(Dropout(
            mesh, f"{tag}.drop_gelu", cfg["dropout"], "local"))
        fc2_w = split_axis(params[f"blocks.{idx}.mlp.fc2.weight"], 0, p)
        self.fc2 = self.add_module(RowLinear(
            mesh, f"{tag}.fc2", fc2_w))
        self.drop_mlp = self.add_module(Dropout(
            mesh, f"{tag}.drop_mlp", cfg["dropout"], "local"))
        self._cache = None

    def forward(self, x_local, training: bool):
        # 注意力子层
        n1 = self.ln1.forward(x_local)
        d1 = self.drop1.forward(n1, training)
        a = self.attn.forward(d1)
        a = self.drop_attn.forward(a, training)
        r1 = [xl + al for xl, al in zip(x_local, a)]
        # MLP 子层
        n2 = self.ln2.forward(r1)
        d2 = self.drop2.forward(n2, training)
        f = self.fc1.forward(d2)
        g_out, g_cache = [], []
        for r, fr in enumerate(f):
            y, c = gelu_forward(fr)
            g_out.append(y)
            g_cache.append(c)
            self.act(r, "gelu", y)
        g_out = self.drop_gelu.forward(g_out, training)
        m = self.fc2.forward(g_out)
        m = self.drop_mlp.forward(m, training)
        r2 = [rl + ml for rl, ml in zip(r1, m)]
        self._cache = (x_local, r1, g_cache)
        return r2

    def backward(self, gx_local):
        x_local, r1, g_cache = self._cache
        # MLP 子层反向
        g_m = self.drop_mlp.backward(gx_local)
        g_g = self.fc2.backward(g_m)
        g_g = self.drop_gelu.backward(g_g)
        g_f = [gelu_backward(g, c) for g, c in zip(g_g, g_cache)]
        g_d2 = self.fc1.backward(g_f)
        g_n2 = self.drop2.backward(g_d2)
        g_r1 = self.ln2.backward(g_n2)
        g_r1 = [a + b for a, b in zip(g_r1, gx_local)]  # 残差
        # 注意力子层反向
        g_a = self.drop_attn.backward(g_r1)
        g_d1 = self.attn.backward(g_a)
        g_n1 = self.drop1.backward(g_d1)
        g_x = self.ln1.backward(g_n1)
        return [a + b for a, b in zip(g_x, g_r1)]


class ParallelTransformer(Module):
    def __init__(self, mesh: DeviceMesh, cfg: dict, params: dict):
        super().__init__(mesh, "model")
        self.cfg = cfg
        p = mesh.world_size
        h, V = cfg["hidden"], cfg["vocab"]
        # 词表除不尽时把权重补齐到 p 的倍数，padding 行不参与数值计算。
        pad_v = int(np.ceil(V / p) * p)
        w_emb = np.zeros((pad_v, h), dtype=np.float64)
        w_emb[:V] = params["embed.weight"]
        emb_shards = split_axis(w_emb, 0, p)
        self.vocab = self.add_module(TiedVocab(
            mesh, "vocab", emb_shards, V))
        self.blocks = [
            self.add_module(TransformerBlock(
                mesh, f"block{i}", cfg, params, i))
            for i in range(cfg["layers"])
        ]
        self.ln_f = self.add_module(LayerNorm(
            mesh, "ln_f", h, params["ln_f.gamma"], params["ln_f.beta"]))
        self.ce = self.add_module(ParallelCrossEntropy(mesh, "ce"))
        self._cache = None

    def forward(self, token_ids: np.ndarray, training: bool = False) -> float:
        p = self.mesh.world_size
        assert (token_ids.shape[0] - 1) % p == 0, (
            "标签长度 S-1 必须能被卡数整除（配置生成处保证）")
        x = self.vocab.embed_forward(token_ids)   # 已是 (S-1)/P 的 SP 段
        for blk in self.blocks:
            x = blk.forward(x, training)
        x = self.ln_f.forward(x)
        # 输出投影前退出 SP：all-gather 序列，隐藏态在各卡复制
        h_rep_full = self.mesh.all_gather(x, axis=0,
                                          layer="output.ag")
        h_rep = [h_rep_full.copy() for _ in range(p)]
        self._cache = (token_ids, h_rep)
        labels = token_ids[1:]
        self.ce.set_meta(self.vocab.bounds, labels)
        logits = self.vocab.logits_forward(h_rep)
        return self.ce.forward(logits)

    def backward(self):
        token_ids, h_rep = self._cache
        p = self.mesh.world_size
        dz = self.ce.backward()                       # 纯词表切分，无通信
        dh_rep = self.vocab.logits_backward(dz, h_rep)  # all-reduce 后复制
        # 退出复制态：reduce-scatter 回 SP 本地段（副本相同，只分散）
        dh = self.mesh.reduce_scatter(dh_rep, axis=0,
                                      layer="output.rs.bwd",
                                      reduce=False)
        g = self.ln_f.backward(dh)
        for blk in reversed(self.blocks):
            g = blk.backward(g)
        self.vocab.embed_backward(g)
