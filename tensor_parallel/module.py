"""模块基类：参数注册、激活显存记账。

序列并行（题设第 6 条）的“全收集 / 先归约再分散”由相邻的列切 / 行切
线性层在各自的 forward/backward 中配对完成（见 layers.py）：
- 列切线性 forward 入口 all-gather，backward 出口 reduce-scatter；
- 行切线性 forward 出口 reduce-scatter，backward 入口 all-gather。
LayerNorm / Dropout 始终工作在按序列切分的激活上，不额外通信。
"""
from __future__ import annotations

import numpy as np

from .mesh import DeviceMesh
from .tensor import ShardedTensor


class Activation:
    """前向产生、保留到反向结束的本地激活，纳入显存记账。"""

    def __init__(self, mesh: DeviceMesh, rank: int, name: str, value):
        self.mesh = mesh
        self.rank = rank
        self.name = name
        self.value = value
        mesh.mem.alloc_act(rank, name, np.asarray(value).size)


class Module:
    def __init__(self, mesh: DeviceMesh, tag: str):
        self.mesh = mesh
        self.tag = tag
        self._params: list[ShardedTensor] = []
        self._children: list["Module"] = []
        # 激活清单：(rank, name, shape, nelement)，供统计页核对。
        self.act_manifest: list[tuple[int, str, tuple, int]] = []

    def register(self, st: ShardedTensor) -> ShardedTensor:
        self._params.append(st)
        return st

    def add_module(self, m: "Module") -> "Module":
        self._children.append(m)
        return m

    def parameters(self) -> list[ShardedTensor]:
        out = list(self._params)
        for c in self._children:
            out.extend(c.parameters())
        return out

    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()

    def act(self, rank: int, name: str, value) -> np.ndarray:
        """登记一个保留到反向的激活并返回其数值。"""
        v = np.asarray(value)
        self.mesh.mem.alloc_act(rank, f"{self.tag}/{name}", v.size)
        self.act_manifest.append((rank, f"{self.tag}/{name}",
                                  tuple(v.shape), int(v.size)))
        return v

    def forward(self, *args, **kwargs):  # pragma: no cover - 接口
        raise NotImplementedError

    def backward(self, *args, **kwargs):  # pragma: no cover - 接口
        raise NotImplementedError
