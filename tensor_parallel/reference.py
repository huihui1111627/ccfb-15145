"""单卡不切分的参考模型：同样的架构、同样手写反向，作为数值基准。

参考模型与并行模型共用同一份完整参数字典；梯度测试时关闭随机丢弃，
随机数流测试在专门的单测里对照 replica / local 两套状态。
"""
from __future__ import annotations

import numpy as np

from .layers import gelu_backward, gelu_forward


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    m = x.max(axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=axis, keepdims=True)


def ln_forward(x, gamma, beta, eps=1e-5):
    mu = x.mean(axis=-1, keepdims=True)
    xc = x - mu
    rstd = 1.0 / np.sqrt((xc * xc).mean(axis=-1, keepdims=True) + eps)
    xhat = xc * rstd
    return xhat * gamma + beta, (x, xhat, rstd)


def ln_backward(gy, x, xhat, rstd, gamma):
    dbeta = gy.sum(axis=0)
    dgamma = (gy * xhat).sum(axis=0)
    dgh = gy * gamma
    n = x.shape[-1]
    dx = (1.0 / n) * rstd * (
        n * dgh
        - dgh.sum(axis=-1, keepdims=True)
        - xhat * (dgh * xhat).sum(axis=-1, keepdims=True)
    )
    return dx, dgamma, dbeta


class RefTransformer:
    def __init__(self, cfg: dict, params: dict):
        self.cfg = cfg
        self.params = params
        self._cache = None

    # -------------------------------------------------------------- forward
    def forward(self, ids: np.ndarray) -> float:
        p = self.params
        labels = ids[1:]
        # 与并行侧对齐：最后一个 token 没有下一 token 标签，嵌入后即丢弃，
        # 整条网络只在 S-1 个位置上运行。
        x = p["embed.weight"][ids[:-1]]      # (S-1, H)
        block_caches = []
        for i in range(self.cfg["layers"]):
            x, c = self._block_forward(x, i)
            block_caches.append(c)
        y_f, ln_f_cache = ln_forward(
            x, p["ln_f.gamma"], p["ln_f.beta"])
        logits = y_f @ p["embed.weight"].T   # (S-1, V)
        z = logits
        m = z.max(axis=-1, keepdims=True)
        e = np.exp(z - m)
        lse = m.squeeze(-1) + np.log(e.sum(axis=-1))
        loss = float((lse - z[np.arange(len(labels)), labels]).mean())
        self._cache = (ids[:-1], labels, block_caches, ln_f_cache, y_f)
        return loss

    def _block_forward(self, x, i):
        p, c = self.params, self.cfg
        pre = f"blocks.{i}."
        n1, ln1_cache = ln_forward(
            x, p[pre + "ln1.gamma"], p[pre + "ln1.beta"])
        qkv = n1 @ p[pre + "attn.qkv.weight"] + p[pre + "attn.qkv.bias"]
        s, h = n1.shape
        hd = h // c["heads"]
        # 交错布局 (s, heads, 3, hd) -> q,k,v
        q, k, v = [t.squeeze(2) for t in np.split(
            qkv.reshape(s, c["heads"], 3, hd), 3, axis=2)]
        scores = np.einsum("qhd,khd->qhk", q, k) / np.sqrt(hd)
        probs = softmax(scores)
        ctx = np.einsum("qhk,khd->qhd", probs, v).reshape(s, h)
        a = ctx @ p[pre + "attn.out.weight"]
        r1 = x + a
        n2, ln2_cache = ln_forward(
            r1, p[pre + "ln2.gamma"], p[pre + "ln2.beta"])
        f = n2 @ p[pre + "mlp.fc1.weight"] + p[pre + "mlp.fc1.bias"]
        g, gcache = gelu_forward(f)
        m_out = g @ p[pre + "mlp.fc2.weight"]
        out = r1 + m_out
        cache = (ln1_cache, n1, q, k, v, probs, ctx,
                 ln2_cache, n2, gcache, g, r1, x)
        return out, cache

    # -------------------------------------------------------------- backward
    def backward(self) -> dict:
        cfg, params = self.cfg, self.params
        ids_eff, labels, caches, ln_f_cache, y_f = self._cache
        n_tok = len(labels)
        grads = {k: np.zeros_like(v) for k, v in params.items()}

        logits = y_f @ params["embed.weight"].T
        sm = softmax(logits)
        dz = sm / n_tok
        dz[np.arange(n_tok), labels] -= 1.0 / n_tok
        # 输出投影对共享嵌入权重的梯度
        grads["embed.weight"] += dz.T @ y_f
        gy = dz @ params["embed.weight"]

        gx, dg, db = ln_backward(gy, *ln_f_cache,
                                 params["ln_f.gamma"])
        grads["ln_f.gamma"] += dg
        grads["ln_f.beta"] += db

        for i in reversed(range(cfg["layers"])):
            gx = self._block_backward(gx, i, grads, caches[i])

        # 嵌入层散加梯度，与输出投影累加到同一份权重；末位 token 梯度为 0
        np.add.at(grads["embed.weight"], ids_eff, gx)
        return grads

    def _block_backward(self, g_out, i, grads, cache):
        cfg, params = self.cfg, self.params
        pre = f"blocks.{i}."
        (ln1_cache, n1, q, k, v, probs, ctx,
         ln2_cache, n2, gcache, g_act, r1, x) = cache
        s, h = x.shape
        hd = h // cfg["heads"]

        # ---- MLP 子层（含残差）----
        g_r1 = g_out.copy()
        grads[pre + "mlp.fc2.weight"] += g_act.T @ g_out
        g_g = g_out @ params[pre + "mlp.fc2.weight"].T
        g_f = gelu_backward(g_g, gcache)
        grads[pre + "mlp.fc1.bias"] += g_f.sum(axis=0)
        grads[pre + "mlp.fc1.weight"] += n2.T @ g_f
        g_n2 = g_f @ params[pre + "mlp.fc1.weight"].T
        g_ln2, dg, db = ln_backward(g_n2, *ln2_cache,
                                    params[pre + "ln2.gamma"])
        grads[pre + "ln2.gamma"] += dg
        grads[pre + "ln2.beta"] += db
        g_r1 = g_r1 + g_ln2   # 残差回传

        # ---- 注意力子层（含残差）----
        g_x = g_r1.copy()
        grads[pre + "attn.out.weight"] += ctx.T @ g_r1
        g_ctx = g_r1 @ params[pre + "attn.out.weight"].T
        g_ctx4 = g_ctx.reshape(s, cfg["heads"], hd)
        g_probs = np.einsum("qhd,khd->qhk", g_ctx4, v)
        dot = (g_probs * probs).sum(axis=-1, keepdims=True)
        g_scores = probs * (g_probs - dot) / np.sqrt(hd)
        g_q = np.einsum("qhk,khd->qhd", g_scores, k)
        g_k = np.einsum("qhk,qhd->khd", g_scores, q)
        g_v = np.einsum("qhk,qhd->khd", probs, g_ctx4)
        g_qkv = np.stack([g_q, g_k, g_v], axis=2).reshape(s, 3 * h)
        grads[pre + "attn.qkv.bias"] += g_qkv.sum(axis=0)
        grads[pre + "attn.qkv.weight"] += n1.T @ g_qkv
        g_n1 = g_qkv @ params[pre + "attn.qkv.weight"].T
        g_ln1, dg, db = ln_backward(g_n1, *ln1_cache,
                                    params[pre + "ln1.gamma"])
        grads[pre + "ln1.gamma"] += dg
        grads[pre + "ln1.beta"] += db
        return g_x + g_ln1
