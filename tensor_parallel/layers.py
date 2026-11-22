"""张量并行各层的前向与手写反向。

约定：序列轴为第 0 维。序列并行（SP）下残差流每卡持有 (T/P, H) 的本地
激活；复制态为 (T, H)，各卡内容相同。

Megatron 风格的列-行配对（题设第 1、6 条），通信全部放在列切层边界，
列-行之间不通信：
- 列切线性：forward 入口 all-gather 序列、出口 reduce-scatter 回 SP；
  backward 入口 all-gather、出口 reduce-scatter（反向配对）。
- 行切线性：输入输出都是 SP 本地段，前向、反向均无通信。
- 注意力夹在列切 QKV 与行切 O 之间，工作在完整序列上、按头切分。
"""
from __future__ import annotations

import numpy as np

from .mesh import DeviceMesh
from .module import Module
from .tensor import ShardedTensor

_PI_SQRT2 = np.sqrt(2.0 / np.pi)
_GELU_C = 0.044715


# ---------------------------------------------------------------------- #
# 随机丢弃：两套随机状态（题设第 7 条）
# ---------------------------------------------------------------------- #
class Dropout(Module):
    def __init__(self, mesh: DeviceMesh, tag: str, p: float, stream: str):
        """stream='local'   按序列切开的部分，各卡掩码不同；
        stream='replica' 未切开的复制部分，各卡掩码相同。"""
        super().__init__(mesh, tag)
        self.p = p
        self.stream = stream
        self._mask = None

    def forward(self, x_list, training: bool):
        if not training or self.p == 0.0:
            self._mask = None
            return [x.copy() for x in x_list]
        masks, out = [], []
        for r in range(self.mesh.world_size):
            rng = self.mesh.rng(r, self.stream)
            m = (rng.random(x_list[r].shape) >= self.p).astype(
                x_list[r].dtype)
            masks.append(m)
            out.append(x_list[r] * m / (1.0 - self.p))
        self._mask = masks
        return out

    def backward(self, g_list):
        if self._mask is None:
            return [g.copy() for g in g_list]
        return [g * m / (1.0 - self.p)
                for g, m in zip(g_list, self._mask)]


# ---------------------------------------------------------------------- #
# 层归一化：复制参数；SP 下每卡归一化自己的序列段，参数梯度 all-reduce
# ---------------------------------------------------------------------- #
class LayerNorm(Module):
    def __init__(self, mesh: DeviceMesh, tag: str, hidden: int,
                 gamma: np.ndarray, beta: np.ndarray):
        super().__init__(mesh, tag)
        self.hidden = hidden
        self.gamma = self.register(ShardedTensor(
            [gamma.copy() for _ in range(mesh.world_size)], mesh,
            f"{tag}.gamma", replicated=True))
        self.beta = self.register(ShardedTensor(
            [beta.copy() for _ in range(mesh.world_size)], mesh,
            f"{tag}.beta", replicated=True))
        self._cache = None

    def forward(self, x_list):
        cache, out = [], []
        for r, x in enumerate(x_list):
            mu = x.mean(axis=-1, keepdims=True)
            xc = x - mu
            rstd = 1.0 / np.sqrt(
                (xc * xc).mean(axis=-1, keepdims=True) + 1e-5)
            xhat = xc * rstd
            y = xhat * self.gamma.shards[r] + self.beta.shards[r]
            cache.append((xhat, rstd))
            self.act(r, "xhat", xhat)
            out.append(y)
        self._cache = cache
        return out

    def backward(self, gy_list):
        gx_list, dgamma_loc, dbeta_loc = [], [], []
        for r, gy in enumerate(gy_list):
            xhat, rstd = self._cache[r]
            dbeta = gy.sum(axis=0)
            dgamma = (gy * xhat).sum(axis=0)
            dgh = gy * self.gamma.shards[r]
            n = self.hidden
            dx = (1.0 / n) * rstd * (
                n * dgh
                - dgh.sum(axis=-1, keepdims=True)
                - xhat * (dgh * xhat).sum(axis=-1, keepdims=True))
            gx_list.append(dx)
            dgamma_loc.append(dgamma)
            dbeta_loc.append(dbeta)
        dgamma_all = self.mesh.all_reduce(
            dgamma_loc, self.tag + ".dgamma")
        dbeta_all = self.mesh.all_reduce(
            dbeta_loc, self.tag + ".dbeta")
        self.gamma.add_grad(dgamma_all)
        self.beta.add_grad(dbeta_all)
        return gx_list


# ---------------------------------------------------------------------- #
# 列切线性：forward 入口 all-gather 序列（SP->复制），输出完整序列的特征
# 分片；backward 出口 reduce-scatter（复制->SP）。配对，无出口额外通信。
# ---------------------------------------------------------------------- #
class ColumnLinear(Module):
    def __init__(self, mesh: DeviceMesh, tag: str,
                 w_shards: list[np.ndarray],
                 b_shards: list[np.ndarray] | None):
        super().__init__(mesh, tag)
        self.weight = self.register(ShardedTensor(
            w_shards, mesh, f"{tag}.weight", shard_axis=1))
        self.bias = None
        if b_shards is not None:
            self.bias = self.register(ShardedTensor(
                b_shards, mesh, f"{tag}.bias", shard_axis=0))
        self._x_full = None

    def forward(self, x_sp):
        # SP -> 复制：all-gather 序列
        x_full = self.mesh.all_gather(x_sp, axis=0,
                                      layer=self.tag + ".ag")
        self._x_full = [self.act(r, "x_full", x_full)
                        for r in range(self.mesh.world_size)]
        out = []
        for r in range(self.mesh.world_size):
            y = x_full @ self.weight.shards[r]
            if self.bias is not None:
                y = y + self.bias.shards[r]
            out.append(y)
        return out  # 完整序列 (T, out/P)，各卡特征分片不同

    def backward(self, gy_full):
        p = self.mesh.world_size
        gw_list, gb_list, gx_partials = [], [], []
        for r in range(p):
            x = self._x_full[r]
            gw_list.append(x.T @ gy_full[r])
            if self.bias is not None:
                gb_list.append(gy_full[r].sum(axis=0))
            gx_partials.append(gy_full[r] @ self.weight.shards[r].T)
        self.weight.add_grad(gw_list)
        if self.bias is not None:
            self.bias.add_grad(gb_list)
        # 反向出口 reduce-scatter：各卡输入梯度分片求和后分散回 SP
        return self.mesh.reduce_scatter(gx_partials, axis=0,
                                        layer=self.tag + ".rs.bwd",
                                        reduce=True)


# ---------------------------------------------------------------------- #
# 行切线性：输入是完整序列的特征分片 (T, in/P)，各卡不同；前向出口
# reduce-scatter 把部分和沿序列分散回 SP（替代全归约，题设第 6 条），
# 反向入口 all-gather 还原完整序列输出梯度，配对。
# ---------------------------------------------------------------------- #
class RowLinear(Module):
    def __init__(self, mesh: DeviceMesh, tag: str,
                 w_shards: list[np.ndarray]):
        super().__init__(mesh, tag)
        self.weight = self.register(ShardedTensor(
            w_shards, mesh, f"{tag}.weight", shard_axis=0))
        self._x_full = None

    def forward(self, x_full):
        # x_full[r]: (T, in/P)，本卡特征分片、完整序列
        self._x_full = [self.act(r, "x_full", x_full[r])
                        for r in range(self.mesh.world_size)]
        partials = [x_full[r] @ self.weight.shards[r]
                    for r in range(self.mesh.world_size)]
        # 各卡部分和 -> 归约后沿序列分散回 SP
        return self.mesh.reduce_scatter(partials, axis=0,
                                        layer=self.tag + ".rs",
                                        reduce=True)

    def backward(self, gy_sp):
        p = self.mesh.world_size
        # 反向入口 all-gather：SP 输出梯度 -> 完整序列 (T, out)，各卡相同
        gy_full = self.mesh.all_gather(gy_sp, axis=0,
                                       layer=self.tag + ".ag.bwd")
        gw_list, gx_list = [], []
        for r in range(p):
            # W_r(in/P,out)：梯度用完整序列的本地特征分片
            gw_list.append(self._x_full[r].T @ gy_full)
            # 输入梯度 (T, in/P)，完整序列、特征分片，交回列切反向
            gx_list.append(gy_full @ self.weight.shards[r].T)
        self.weight.add_grad(gw_list)
        return gx_list


# ---------------------------------------------------------------------- #
# GELU（tanh 近似）
# ---------------------------------------------------------------------- #
def gelu_forward(x: np.ndarray):
    u = _PI_SQRT2 * (x + _GELU_C * x ** 3)
    t = np.tanh(u)
    return 0.5 * x * (1.0 + t), (x, t)


def gelu_backward(g: np.ndarray, cache) -> np.ndarray:
    x, t = cache
    u = _PI_SQRT2 * (x + _GELU_C * x ** 3)
    du = _PI_SQRT2 * (1.0 + 3.0 * _GELU_C * x ** 2)
    return g * (0.5 * (1.0 + t)
                + 0.5 * x * (1.0 - t * t) * du)


# ---------------------------------------------------------------------- #
# 多头注意力：按头切分（题设第 2 条），每卡只算自己那几个头。
# QKV 列切输出完整序列的本卡头 QKV，注意力在完整序列上计算、无通信，
# 上下文交给行切 O（其出口 reduce-scatter 回 SP）。
# ---------------------------------------------------------------------- #
class MultiHeadAttention(Module):
    def __init__(self, mesh: DeviceMesh, tag: str, hidden: int, n_heads: int,
                 qkv_w_shards, qkv_b_shards, o_w_shards):
        super().__init__(mesh, tag)
        assert hidden % n_heads == 0
        assert n_heads % mesh.world_size == 0
        self.hidden = hidden
        self.n_heads = n_heads
        self.heads_per_rank = n_heads // mesh.world_size
        self.head_dim = hidden // n_heads
        self.qkv = self.add_module(ColumnLinear(
            mesh, f"{tag}.qkv", qkv_w_shards, qkv_b_shards))
        self.o_proj = self.add_module(RowLinear(
            mesh, f"{tag}.out", o_w_shards))
        self._cache = None

    def forward(self, x_sp):
        p = self.mesh.world_size
        qkv_full = self.qkv.forward(x_sp)     # (T, 3H/P) 完整序列
        outs, cache = [], []
        for r in range(p):
            s = qkv_full[r].shape[0]
            q, k, v = [t.squeeze(2) for t in np.split(
                qkv_full[r].reshape(s, self.heads_per_rank, 3,
                                    self.head_dim), 3, axis=2)]
            scores = np.einsum("qhd,khd->qhk", q, k) \
                / np.sqrt(self.head_dim)
            probs = self._softmax(scores)
            ctx = np.einsum("qhk,khd->qhd", probs, v)
            ctx2 = ctx.reshape(s, self.heads_per_rank * self.head_dim)
            cache.append((q, k, v, probs))
            self.act(r, "attn_probs", probs)
            outs.append(ctx2)
        self._cache = cache
        return self.o_proj.forward(outs)      # 行切出口 RS 回 SP

    def backward(self, gy_sp):
        p = self.mesh.world_size
        g_ctx_full = self.o_proj.backward(gy_sp)  # 完整序列特征分片
        g_qkv_full = []
        for r in range(p):
            q, k, v, probs = self._cache[r]
            g_ctx = g_ctx_full[r].reshape(q.shape)
            g_probs = np.einsum("qhd,khd->qhk", g_ctx, v)
            g_scores = self._softmax_backward(g_probs, probs) \
                / np.sqrt(self.head_dim)
            g_q = np.einsum("qhk,khd->qhd", g_scores, k)
            g_k = np.einsum("qhk,qhd->khd", g_scores, q)
            g_v = np.einsum("qhk,qhd->khd", probs, g_ctx)
            g_qkv_full.append(
                np.stack([g_q, g_k, g_v], axis=2).reshape(
                    q.shape[0], -1))
        return self.qkv.backward(g_qkv_full)  # 列切反向出口 RS 回 SP

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        m = x.max(axis=-1, keepdims=True)
        e = np.exp(x - m)
        return e / e.sum(axis=-1, keepdims=True)

    @staticmethod
    def _softmax_backward(g: np.ndarray, p: np.ndarray) -> np.ndarray:
        dot = (g * p).sum(axis=-1, keepdims=True)
        return p * (g - dot)
