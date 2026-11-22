"""模拟设备网格、集合通信、双路随机数与显存记账。

用进程内的 numpy 数组模拟多张设备卡：每张“卡”是 DeviceMesh 中一个 rank，
卡间通信全部由本模块的集合操作模拟，并按环形（ring）算法的经典通信量
公式记账，便于和手算公式逐项核对。
"""
from __future__ import annotations

import numpy as np

# 记账单位：1 个 float64 元素 8 字节。测试统一用 float64。
_BYTES_PER_ELEMENT = 8.0


class CommEvent:
    """一次集合通信事件的记录。"""

    def __init__(self, name: str, layer: str, shape, nelement_per_rank: int,
                 bytes_per_rank: float, total_bytes: float):
        self.name = name
        self.layer = layer
        self.shape = tuple(shape)
        self.nelement_per_rank = nelement_per_rank
        self.bytes_per_rank = bytes_per_rank
        self.total_bytes = total_bytes

    def as_dict(self):
        return {
            "name": self.name,
            "layer": self.layer,
            "shape": list(self.shape),
            "elements_per_rank": self.nelement_per_rank,
            "bytes_per_rank": self.bytes_per_rank,
            "total_bytes": self.total_bytes,
        }


class MemoryTracker:
    """按 rank 记录权重 / 梯度 / 激活占用与峰值。"""

    def __init__(self, world_size: int):
        self.world_size = world_size
        # 每类是 dict[name -> bytes]，free 时按名字删除，重名复用安全。
        self.params = [dict() for _ in range(world_size)]
        self.grads = [dict() for _ in range(world_size)]
        self.acts = [dict() for _ in range(world_size)]
        self.peak = [0.0 for _ in range(world_size)]

    def _bump_peak(self, rank: int):
        cur = (sum(self.params[rank].values())
               + sum(self.grads[rank].values())
               + sum(self.acts[rank].values()))
        self.peak[rank] = max(self.peak[rank], cur)

    def alloc_param(self, rank: int, name: str, n: int):
        self.params[rank][name] = n * _BYTES_PER_ELEMENT
        self._bump_peak(rank)

    def alloc_grad(self, rank: int, name: str, n: int):
        self.grads[rank][name] = n * _BYTES_PER_ELEMENT
        self._bump_peak(rank)

    def alloc_act(self, rank: int, name: str, n: int):
        self.acts[rank][name] = n * _BYTES_PER_ELEMENT
        self._bump_peak(rank)

    def free_act(self, rank: int, name: str):
        self.acts[rank].pop(name, None)

    def current(self, rank: int) -> dict:
        return {
            "params": sum(self.params[rank].values()),
            "grads": sum(self.grads[rank].values()),
            "acts": sum(self.acts[rank].values()),
        }

    def snapshot(self) -> dict:
        return {
            "params_per_rank": [sum(d.values()) for d in self.params],
            "grads_per_rank": [sum(d.values()) for d in self.grads],
            "peak_per_rank": list(self.peak),
        }


class DeviceMesh:
    """world_size 张模拟卡组成的通信域。"""

    def __init__(self, world_size: int, base_seed: int = 1234):
        assert world_size >= 1
        self.world_size = world_size
        self.base_seed = base_seed
        # 两套随机状态：
        #   replica_rng —— 未按序列切分的部分，各卡相同
        #   local_rng   —— 按序列切分的部分，各卡不同
        self.replica_rng = [np.random.default_rng(base_seed)
                            for _ in range(world_size)]
        self.local_rng = [np.random.default_rng(base_seed + 1000 + r)
                          for r in range(world_size)]
        self.events: list[CommEvent] = []
        self.mem = MemoryTracker(world_size)
        self._step = 0

    # ------------------------------------------------------------------ #
    # 随机数
    # ------------------------------------------------------------------ #
    def rng(self, rank: int, stream: str) -> np.random.Generator:
        if stream == "replica":
            return self.replica_rng[rank]
        if stream == "local":
            return self.local_rng[rank]
        raise ValueError(stream)

    # ------------------------------------------------------------------ #
    # 记账。所有集合操作的分片在参与维度上等大，统一按环形公式：
    #   all-reduce:    2*(p-1)/p * n            （n=每卡参与元素数）
    #   all-gather:    (p-1)/p * n_full         （n_full=聚合后元素数）
    #   reduce-scatter: (p-1)/p * n_full         （n_full=归约前每卡元素数）
    # ------------------------------------------------------------------ #
    def _bill(self, name, layer, shape, nelement_per_rank, factor_num,
              factor_den):
        bytes_per_rank = (nelement_per_rank * factor_num / factor_den
                          * _BYTES_PER_ELEMENT)
        total = bytes_per_rank * self.world_size
        ev = CommEvent(name, layer, shape, nelement_per_rank,
                       bytes_per_rank, total)
        self.events.append(ev)
        return ev

    def _ar_bill(self, layer, n):
        return self._bill("all_reduce", layer, (), n,
                          2 * (self.world_size - 1), self.world_size)

    def _ag_bill(self, layer, n_full):
        # 每卡持有 n_full/p 个元素；ring all-gather 每卡流量 (p-1)/p*n_full
        return self._bill("all_gather", layer, (), n_full,
                          self.world_size - 1, self.world_size)

    def _rs_bill(self, layer, n_full):
        return self._bill("reduce_scatter", layer, (), n_full,
                          self.world_size - 1, self.world_size)

    # ------------------------------------------------------------------ #
    # 集合操作（数值语义 + 通信量记录）
    # ------------------------------------------------------------------ #
    def all_reduce(self, shards: list[np.ndarray], layer: str,
                   op: str = "sum") -> list[np.ndarray]:
        """各卡得到相同的归约结果。"""
        assert len(shards) == self.world_size
        n = shards[0].size
        self._ar_bill(layer, n)
        total = np.sum(np.stack(shards, axis=0), axis=0)
        if op == "sum":
            out = total
        elif op == "max":
            out = np.max(np.stack(shards, axis=0), axis=0)
        else:
            raise ValueError(op)
        return [out.copy() for _ in range(self.world_size)]

    def all_gather(self, shards: list[np.ndarray], axis: int,
                   layer: str) -> np.ndarray:
        """沿 axis 拼接各卡分片，每张卡拿到完整结果。"""
        assert len(shards) == self.world_size
        n_full = sum(s.size for s in shards)
        self._ag_bill(layer, n_full)
        return np.concatenate(shards, axis=axis)

    def reduce_scatter(self, replicas: list[np.ndarray], axis: int,
                       layer: str, reduce: bool = True) -> list[np.ndarray]:
        """先归约再沿 axis 分散（允许该维不能被 world_size 整除）。

        reduce=True  用于行切线性出口：各卡持有不同的部分和，先求和；
        reduce=False 用于 SP 入口：各卡持有相同副本，只分散不再求和
        （ring 通信量公式相同，数值语义不同）。
        """
        assert len(replicas) == self.world_size
        n_full = replicas[0].size
        self._rs_bill(layer, n_full)
        if reduce:
            total = np.sum(np.stack(replicas, axis=0), axis=0)
        else:
            total = replicas[0]
        return list(np.array_split(total, self.world_size, axis=axis))

    def broadcast(self, shards: list[np.ndarray], src: int,
                  layer: str) -> list[np.ndarray]:
        """模拟广播：记账按 (p-1)/p*n，结果各卡一致。"""
        assert len(shards) == self.world_size
        n = shards[src].size
        self._bill("broadcast", layer, (), n,
                   self.world_size - 1, self.world_size)
        return [shards[src].copy() for _ in range(self.world_size)]

    # ------------------------------------------------------------------ #
    def begin_step(self):
        self._step += 1
        self.events.clear()

    def comm_summary(self) -> dict:
        agg: dict[str, dict] = {}
        for ev in self.events:
            d = agg.setdefault(ev.name, {"events": 0,
                                         "bytes_per_rank": 0.0,
                                         "total_bytes": 0.0})
            d["events"] += 1
            d["bytes_per_rank"] += ev.bytes_per_rank
            d["total_bytes"] += ev.total_bytes
        return agg

    def comm_events_dict(self) -> list[dict]:
        return [ev.as_dict() for ev in self.events]
