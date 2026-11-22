"""训练循环：纯 SGD 更新分片参数，并记录损失曲线。"""
from __future__ import annotations

import numpy as np

from .mesh import DeviceMesh
from .model import ParallelTransformer


def sgd_step(model: ParallelTransformer, lr: float):
    for st in model.parameters():
        if st.grad is None:
            continue
        for r in range(model.mesh.world_size):
            st.shards[r] -= lr * st.grad.shards[r]


def train(cfg: dict, params: dict, steps: int = 60, lr: float = 0.05,
          seed: int = 0, dropout: float = 0.0) -> dict:
    """固定一批随机 token，反复训练，返回损失曲线与每步通信/显存快照。"""
    rng = np.random.default_rng(seed)
    ids = rng.integers(0, cfg["vocab"], size=cfg["seq_len"])
    cfg = dict(cfg)
    cfg["dropout"] = dropout

    mesh = DeviceMesh(cfg["world_size"], base_seed=123)
    model = ParallelTransformer(mesh, cfg, params)

    losses, comm_bytes, peaks = [], [], []
    for step in range(steps):
        model.zero_grad()
        mesh.begin_step()
        loss = model.forward(ids, training=True)
        model.backward()
        sgd_step(model, lr)
        losses.append(loss)
        summary = mesh.comm_summary()
        comm_bytes.append({
            op: summary.get(op, {}).get("bytes_per_rank", 0.0)
            for op in ("all_reduce", "all_gather", "reduce_scatter")
        })
        peaks.append(list(mesh.mem.peak))
    return {
        "losses": losses,
        "comm_bytes_per_step": comm_bytes[0],
        "peak_bytes": peaks[-1],
        "events": mesh.comm_events_dict(),
        "final_peak_snapshot": mesh.mem.snapshot(),
    }
