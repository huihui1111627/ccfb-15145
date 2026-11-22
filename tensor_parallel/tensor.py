"""分片张量 ShardedTensor 与参数初始化。

一个 ShardedTensor 在每张卡上持有一个本地 numpy 数组。
- 按词表/列/头切分：各卡分片语义不同；词表除不尽时先补齐再等分。
- 复制参数（LayerNorm 的 gamma/beta 等）：各卡持有相同副本。
参数同时在 MemoryTracker 里登记权重与梯度占用。
"""
from __future__ import annotations

import numpy as np

from .mesh import DeviceMesh


class ShardedTensor:
    def __init__(self, shards: list[np.ndarray], mesh: DeviceMesh,
                 name: str, kind: str = "param", replicated: bool = False,
                 shard_axis: int | None = None):
        assert len(shards) == mesh.world_size
        self.shards = [np.asarray(s) for s in shards]
        self.mesh = mesh
        self.name = name
        self.kind = kind            # param / grad / activation
        self.replicated = replicated
        self.shard_axis = shard_axis
        self.grad: ShardedTensor | None = None
        if kind == "param":
            for r in range(mesh.world_size):
                mesh.mem.alloc_param(r, name, self.shards[r].size)
        elif kind == "grad":
            for r in range(mesh.world_size):
                mesh.mem.alloc_grad(r, name, self.shards[r].size)

    @property
    def shape(self):
        return self.shards[0].shape

    @property
    def dtype(self):
        return self.shards[0].dtype

    def zero_grad(self):
        if self.grad is not None:
            for g in self.grad.shards:
                g.fill(0.0)

    def alloc_grad(self, name: str | None = None) -> "ShardedTensor":
        """建立与本参数同形的零梯度并记账。共享参数只登记一份显存。"""
        if self.grad is None:
            gname = name or f"grad::{self.name}"
            gshards = [np.zeros_like(s) for s in self.shards]
            self.grad = ShardedTensor(
                gshards, self.mesh, gname, kind="grad",
                replicated=self.replicated,
                shard_axis=self.shard_axis)
        return self.grad

    def add_grad(self, grad_shards: list[np.ndarray]):
        """累加梯度（词嵌入与输出层共享权重时用到）。"""
        g = self.alloc_grad()
        for r in range(self.mesh.world_size):
            g.shards[r] += grad_shards[r]
        return g


def split_evenly(total: int, world_size: int) -> list[tuple[int, int]]:
    """返回每卡 (start, end)，除不尽时最后一卡稍大（真实数据边界）。"""
    base = total // world_size
    bounds = []
    for r in range(world_size):
        start = r * base if r < world_size - 1 else (world_size - 1) * base
        end = (r + 1) * base if r < world_size - 1 else total
        bounds.append((start, end))
    return bounds


def split_axis(array: np.ndarray, axis: int, world_size: int,
               pad_to: int | None = None) -> list[np.ndarray]:
    """沿 axis 切成 world_size 份。

    pad_to 给定时先把该维补齐到 world_size*pad_to，再做**等分**。
    词表除不尽的场景必须走这条路径：各卡本地索引按等分起点计算，
    若用 array_split 切真实行数会造成末卡索引错位。
    """
    if pad_to is not None:
        target = world_size * pad_to
        if array.shape[axis] < target:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, target - array.shape[axis])
            array = np.pad(array, pad_widths)
        return list(np.split(array, world_size, axis=axis))
    return list(np.array_split(array, world_size, axis=axis))


def init_param(full: np.ndarray, mesh: DeviceMesh, name: str,
               axis: int | None, pad_to: int | None = None) -> ShardedTensor:
    """把一个完整参数按 axis 切开（axis=None 表示复制）。"""
    if axis is None:
        shards = [full.copy() for _ in range(mesh.world_size)]
        return ShardedTensor(shards, mesh, name, replicated=True)
    shards = split_axis(full, axis, mesh.world_size, pad_to)
    return ShardedTensor(shards, mesh, name, replicated=False)


def kaiming_uniform(rng: np.random.Generator, shape, fan_in: int) -> np.ndarray:
    bound = (3.0 / fan_in) ** 0.5
    return rng.uniform(-bound, bound, size=shape)
