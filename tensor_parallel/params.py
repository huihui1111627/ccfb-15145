"""确定性地生成完整参数字典与测试配置。"""
from __future__ import annotations

import numpy as np

from .tensor import kaiming_uniform


def make_params(cfg: dict, seed: int = 2024) -> dict:
    rng = np.random.default_rng(seed)
    h, ff, V, L = cfg["hidden"], cfg["ff"], cfg["vocab"], cfg["layers"]
    p: dict[str, np.ndarray] = {}
    p["embed.weight"] = kaiming_uniform(rng, (V, h), h)
    p["ln_f.gamma"] = rng.normal(1.0, 0.05, size=h)
    p["ln_f.beta"] = np.zeros(h)
    for i in range(L):
        pre = f"blocks.{i}."
        p[pre + "ln1.gamma"] = rng.normal(1.0, 0.05, size=h)
        p[pre + "ln1.beta"] = np.zeros(h)
        p[pre + "ln2.gamma"] = rng.normal(1.0, 0.05, size=h)
        p[pre + "ln2.beta"] = np.zeros(h)
        p[pre + "attn.qkv.weight"] = kaiming_uniform(
            rng, (h, 3 * h), h)
        p[pre + "attn.qkv.bias"] = np.zeros(3 * h)
        p[pre + "attn.out.weight"] = kaiming_uniform(
            rng, (h, h), h)
        p[pre + "mlp.fc1.weight"] = kaiming_uniform(rng, (h, ff), h)
        p[pre + "mlp.fc1.bias"] = np.zeros(ff)
        p[pre + "mlp.fc2.weight"] = kaiming_uniform(rng, (ff, h), ff)
    return p


def make_configs(n: int = 36, seed: int = 7) -> list[dict]:
    """随机生成几十种 (层数, 头数, 词表, 卡数, 序列长) 的合法组合。"""
    rng = np.random.default_rng(seed)
    configs = []
    seen = set()
    tries = 0
    while len(configs) < n and tries < n * 30:
        tries += 1
        P = int(rng.choice([1, 2, 3, 4]))
        L = int(rng.integers(1, 4))
        # 头数必须 >= 卡数且能被卡数整除（按头切分的前提）
        heads_choices = [h for h in (2, 4, 8)
                         if h >= P and h % P == 0]
        if not heads_choices:
            continue
        heads = int(rng.choice(heads_choices))
        h = int(rng.choice([16, 24, 32]))
        ff_mult = int(rng.choice([2, 4]))
        ff = h * ff_mult
        # 词表刻意包含不能被 P 整除的情况
        V = int(rng.integers(11, 60))
        # S-1 必须能被 P 整除（SP 等分的前提）
        s_label = P * int(rng.integers(2, 6))
        S = s_label + 1
        dropout = 0.0
        key = (L, heads, h, ff, V, P, S)
        if key in seen:
            continue
        seen.add(key)
        configs.append({
            "layers": L, "heads": heads, "hidden": h, "ff": ff,
            "vocab": V, "world_size": P, "seq_len": S,
            "dropout": dropout,
        })
    return configs
