"""词表并行：词嵌入、输出投影（共享同一份切分权重）与分片交叉熵。

题设第 3、4、5 条：
- 嵌入按词表切，查不到的位置填零，all-reduce 求和得到完整嵌入；
- 输出分数按词表切，交叉熵不收齐分数：先跨卡求全局最大值与归一化项，
  再在各卡本地算损失与梯度，数值稳定不溢出；
- 嵌入与输出层共用权重，两处梯度累加到同一个 ShardedTensor。

输出投影要求输入在序列维复制（模型在投影前 all-gather、反向
reduce-scatter 配对），因此 logits 是纯词表一维切分：每卡持有全部
token 在本卡词表段上的分数，max/sumexp 可在对齐的 token 位置上归约。
"""
from __future__ import annotations

import numpy as np

from .mesh import DeviceMesh
from .module import Module
from .tensor import ShardedTensor

NEG_INF = np.finfo(np.float64).min / 4.0  # 留出做差不溢出的余量


class TiedVocab(Module):
    """词嵌入与输出 logits 的共享权重 W（按词表维切分）。"""

    def __init__(self, mesh: DeviceMesh, tag: str,
                 w_shards: list[np.ndarray], vocab_size: int):
        super().__init__(mesh, tag)
        self.weight = self.register(
            ShardedTensor(w_shards, mesh, f"{tag}.weight",
                          shard_axis=0))
        self.vocab_size = vocab_size
        self.shard_cols = w_shards[0].shape[0]
        self.padded_vocab = self.shard_cols * mesh.world_size
        # 分片“先补齐再等分”，本地索引按等分起点；padding 行由 -inf /
        # 掩码保证不参与计算。
        self.bounds = [(r * self.shard_cols,
                        min((r + 1) * self.shard_cols, vocab_size))
                       for r in range(mesh.world_size)]
        self._token_ids = None

    # ---------------- 嵌入 ---------------- #
    def embed_forward(self, token_ids: np.ndarray) -> list[np.ndarray]:
        """token_ids: (S,)，各卡相同。

        all-reduce 求和得到全部 S 个位置的完整嵌入后，丢弃最后一个无标签
        位置，再 reduce-scatter 成 ( (S-1)/P, H ) 的 SP 本地段。
        """
        p = self.mesh.world_size
        s, h = token_ids.shape[0], self.weight.shards[0].shape[1]
        assert (s - 1) % p == 0
        self._token_ids = token_ids
        partials = []
        for r in range(p):
            v0, v1 = self.bounds[r]
            emb = np.zeros((s, h), dtype=self.weight.dtype)
            hit = (token_ids >= v0) & (token_ids < v1)
            if hit.any():
                emb[hit] = self.weight.shards[r][token_ids[hit] - v0]
            partials.append(emb)
        full_list = self.mesh.all_reduce(
            partials, layer=self.tag + ".embed.ar")
        full = full_list[0][:-1]  # 最后一个位置没有下一 token 标签
        full_replicas = [full.copy() for _ in range(p)]
        # 进入 SP：副本相同，只分散不再求和。
        return self.mesh.reduce_scatter(full_replicas, axis=0,
                                        layer=self.tag + ".embed.rs",
                                        reduce=False)

    def embed_backward(self, g_local: list[np.ndarray]):
        """配对反向：先 all-gather 还原 (S-1, H)，补末位零梯度再散加。"""
        g_full = self.mesh.all_gather(g_local, axis=0,
                                      layer=self.tag + ".embed.ag.bwd")
        g_full = np.vstack(
            [g_full, np.zeros((1, g_full.shape[1]))])  # 末位无标签
        p = self.mesh.world_size
        ids = self._token_ids
        dw = []
        for r in range(p):
            v0, v1 = self.bounds[r]
            grad = np.zeros_like(self.weight.shards[r])
            hit = (ids >= v0) & (ids < v1)
            if hit.any():
                np.add.at(grad, ids[hit] - v0, g_full[hit])
            dw.append(grad)
        self.weight.add_grad(dw)

    # ---------------- 输出 logits（输入在序列维复制）---------------- #
    def logits_forward(self, h_rep: list[np.ndarray]) -> list[np.ndarray]:
        """每卡 (T, Vpad/P)：全部 token、本卡词表段；padding 列置 -inf。"""
        out = []
        for r, h in enumerate(h_rep):
            z = h @ self.weight.shards[r].T
            v0, v1 = self.bounds[r]
            real_cols = v1 - v0
            if real_cols < z.shape[1]:
                z[:, real_cols:] = NEG_INF
            out.append(z)
        return out

    def logits_backward(self, dz_vocab: list[np.ndarray],
                        h_rep: list[np.ndarray]):
        """dz_vocab：每卡全部 token、本卡词表段。

        隐藏态梯度要对全词表求和，各卡本地部分积跨卡 all-reduce；
        权重 W 按词表切，各卡梯度只覆盖自己的词表行，无需通信。
        """
        dh_partial, dw = [], []
        for r, dz in enumerate(dz_vocab):
            dh_partial.append(dz @ self.weight.shards[r])
            g = dz.T @ h_rep[r]
            v0, v1 = self.bounds[r]
            real_rows = v1 - v0
            if real_rows < g.shape[0]:
                g[real_rows:, :] = 0.0  # padding 行不产生梯度
            dw.append(g)
        self.weight.add_grad(dw)  # 与嵌入梯度累加到同一份权重
        # 全词表贡献求和，结果各卡相同（复制态）
        return self.mesh.all_reduce(dh_partial,
                                    layer=self.tag + ".logits.dh.ar")


class ParallelCrossEntropy(Module):
    """词表并行交叉熵：不收齐分数，跨卡只传标量统计量。

    输入是纯词表一维切分（每卡全部 token × 本卡词表段），token 位置天然
    对齐。前向 3 次 all-reduce（max / sumexp / 损失分量和），反向零通信，
    每卡只对自己词表段内的标签构造 one-hot 项。
    """

    def __init__(self, mesh: DeviceMesh, tag: str):
        super().__init__(mesh, tag)
        self._cache = None
        self._bounds = None
        self._labels = None

    def set_meta(self, bounds, labels):
        self._bounds = bounds
        self._labels = labels  # 完整标签 (T,)，各卡相同

    def forward(self, logits_vocab: list[np.ndarray]) -> float:
        p = self.mesh.world_size
        z0 = logits_vocab[0]
        s_total = z0.shape[0]
        labels = self._labels

        # 1) 各卡本地最大值 -> 全局最大值（位置对齐，归约结果各卡相同）
        local_max = [z.max(axis=-1, keepdims=True) for z in logits_vocab]
        global_max = self.mesh.all_reduce(
            local_max, layer=self.tag + ".max", op="max")[0]
        # 2) 平移指数和 -> 全局归一化项
        local_exp, softmax_vocab = [], []
        for r, z in enumerate(logits_vocab):
            e = np.exp(z - global_max)
            esum = e.sum(axis=-1, keepdims=True)
            local_exp.append(esum)
        global_exp = self.mesh.all_reduce(
            local_exp, layer=self.tag + ".sumexp")[0]
        for r, z in enumerate(logits_vocab):
            softmax_vocab.append(np.exp(z - global_max) / global_exp)
        # 3) 损失。每卡的 (max+logsumexp) 项完全相同（重复项），只在
        #    rank0 计一次；-z_label 按词表分片，各卡贡献自己命中的部分，
        #    再 all-reduce 求和。
        base = (global_max.squeeze(-1)
                + np.log(global_exp.squeeze(-1)))      # (T,)，各卡相同
        local_loss = []
        for r, z in enumerate(logits_vocab):
            v0, v1 = self._bounds[r]
            in_range = (labels >= v0) & (labels < v1)
            z_label = np.zeros(s_total)
            if in_range.any():
                z_label[in_range] = z[in_range, labels[in_range] - v0]
            contrib = -z_label
            if r == 0:
                contrib = contrib + base
            local_loss.append(np.asarray(contrib.sum()).reshape(1))
        global_loss_sum = self.mesh.all_reduce(
            local_loss, layer=self.tag + ".loss_sum")[0]
        self._cache = (softmax_vocab, s_total)
        return float(global_loss_sum.sum() / s_total)

    def backward(self) -> list[np.ndarray]:
        softmax_vocab, s_total = self._cache
        p = self.mesh.world_size
        labels = self._labels
        out = []
        for r in range(p):
            v0, v1 = self._bounds[r]
            d = softmax_vocab[r] / s_total
            in_range = (labels >= v0) & (labels < v1)
            if in_range.any():
                d[in_range, labels[in_range] - v0] -= 1.0 / s_total
            out.append(d)
        return out  # 纯词表切分，反向无通信
