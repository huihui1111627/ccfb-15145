"""生成可视化报告所需的数据 JSON（切分方式、显存、通信量、损失曲线）。"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tensor_parallel import make_configs, make_params
from tensor_parallel import stats as stats_mod
from tensor_parallel.train import train


def build_report_data(out_path: str):
    # 选一个有代表性的配置做详细展示
    cfg = {
        "layers": 2, "heads": 4, "hidden": 24, "ff": 48,
        "vocab": 31, "world_size": 2, "seq_len": 9, "dropout": 0.0,
    }
    params = make_params(cfg, seed=1)
    train_out = train(cfg, params, steps=80, lr=0.1)

    plan = stats_mod.layer_plan(cfg)
    weight = stats_mod.weight_bytes_per_rank(cfg)
    activ = stats_mod.activation_bytes_per_rank(cfg)
    comm = stats_mod.communication_formula(cfg)

    # 多张卡的显存明细
    per_rank_mem = []
    for r in range(cfg["world_size"]):
        per_rank_mem.append({
            "rank": r,
            "param_bytes": weight["param_bytes"],
            "grad_bytes": weight["grad_bytes"],
            "activation_bytes": activ["bytes"],
            "peak_bytes": (weight["param_bytes"] + weight["grad_bytes"]
                           + activ["bytes"]),
        })

    # 逐层激活明细（rank0 口径）
    act_items = [{"name": n, "elements": e, "bytes": e * 8.0}
                 for n, e in activ["items"]]

    # 每步各类集合操作的通信量（实测，与公式一致）
    comm_events = train_out["events"]

    data = {
        "config": cfg,
        "layer_plan": plan,
        "memory": {
            "weight_breakdown_elems": weight["breakdown_elems"],
            "per_rank": per_rank_mem,
            "activation_items": act_items,
        },
        "communication": {
            "formula": comm,
            "events": comm_events,
        },
        "loss_curve": train_out["losses"],
    }
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"报告数据已写入 {out_path}")

    # 额外：跨多个随机配置汇总通信/显存，供页面横向对比
    configs = make_configs(n=24)
    sweep = []
    for c in configs:
        w = stats_mod.weight_bytes_per_rank(c)
        a = stats_mod.activation_bytes_per_rank(c)
        cm = stats_mod.communication_formula(c)
        sweep.append({
            "config": {k: c[k] for k in
                       ("layers", "heads", "hidden", "ff",
                        "vocab", "world_size", "seq_len")},
            "peak_bytes_rank0": (w["param_bytes"] + w["grad_bytes"]
                                 + a["bytes"]),
            "comm_bytes_per_rank": sum(cm["bytes_per_rank"].values()),
            "comm_events": cm["events"],
        })
    sweep_path = os.path.join(os.path.dirname(out_path), "sweep.json")
    with open(sweep_path, "w") as f:
        json.dump(sweep, f, indent=2, ensure_ascii=False)
    print(f"配置扫描数据已写入 {sweep_path}")


if __name__ == "__main__":
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "report_data.json")
    build_report_data(out)
