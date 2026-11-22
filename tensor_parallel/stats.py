"""按公式计算切分方式、每卡显存与每步集合通信量。

环形（ring）集合操作的每卡通信量（元素数，字节 = 元素 ×8）：
    all-reduce     2(P-1)/P · n
    all-gather     (P-1)/P · n_full
    reduce-scatter (P-1)/P · n_full
其中 n 为每卡参与元素数，n_full 为聚合/归约后的完整元素数。
"""
from __future__ import annotations

import numpy as np

BYTES = 8.0


def ar_elements(n: int, p: int) -> float:
    return 2.0 * (p - 1) / p * n


def ag_elements(n_full: int, p: int) -> float:
    return (p - 1) / p * n_full


def rs_elements(n_full: int, p: int) -> float:
    return (p - 1) / p * n_full


def layer_plan(cfg: dict) -> list[dict]:
    """逐层切分方式描述，供页面第一块展示。"""
    L, h, ff, heads, V, P = (cfg["layers"], cfg["hidden"], cfg["ff"],
                             cfg["heads"], cfg["vocab"], cfg["world_size"])
    vpad = int(np.ceil(V / P))
    plan = [
        {"layer": "词嵌入/输出权重（共享）",
         "sharding": f"按词表切：每卡 {vpad} 行（含 padding）× H={h}",
         "comm_fwd": "嵌入 all-reduce + reduce-scatter；输出前 all-gather；CE 3 次 all-reduce",
         "comm_bwd": "输出 dh all-reduce + reduce-scatter；嵌入 all-gather"},
        {"layer": "LayerNorm ×(2L+1) / Dropout / GELU / 残差",
         "sharding": "按序列长度切：每卡 (S-1)/P 个 token；gamma/beta 复制",
         "comm_fwd": "无（LN 复制参数梯度在反向 all-reduce）",
         "comm_bwd": "每个 LN 2 次 all-reduce（dgamma、dbeta）"},
    ]
    for i in range(L):
        plan += [
            {"layer": f"block{i} QKV（列切）",
             "sharding": f"W {h}×{3*h//P}、偏置 {3*h//P}；按头切 {heads//P}/{heads} 头",
             "comm_fwd": "入口 all-gather 序列",
             "comm_bwd": "出口 reduce-scatter"},
            {"layer": f"block{i} 注意力内部",
             "sharding": f"每卡 {heads//P} 个头，完整序列上算 softmax",
             "comm_fwd": "无",
             "comm_bwd": "无"},
            {"layer": f"block{i} 输出 O（行切）",
             "sharding": f"W {h//P}×{h}，无偏置",
             "comm_fwd": "出口 reduce-scatter（替代全归约）",
             "comm_bwd": "入口 all-gather"},
            {"layer": f"block{i} FC1（列切）",
             "sharding": f"W {h}×{ff//P}、偏置 {ff//P}",
             "comm_fwd": "入口 all-gather",
             "comm_bwd": "出口 reduce-scatter"},
            {"layer": f"block{i} FC2（行切）",
             "sharding": f"W {ff//P}×{h}，无偏置",
             "comm_fwd": "出口 reduce-scatter",
             "comm_bwd": "入口 all-gather"},
        ]
    return plan


def weight_bytes_per_rank(cfg: dict) -> dict:
    """每卡权重与梯度的元素数/字节（padding 计入显存）。"""
    L, h, ff, V, P = (cfg["layers"], cfg["hidden"], cfg["ff"],
                      cfg["vocab"], cfg["world_size"])
    vpad = int(np.ceil(V / P))
    elems = {}
    elems["vocab.weight"] = vpad * h              # 共享，一份权重一份梯度
    # 2L 个块内 LN + 1 个末层 LN，每个含 gamma、beta 两个复制向量
    elems["layernorm"] = 2 * (2 * L + 1) * h
    per_block = (
        h * (3 * h // P) + (3 * h // P)           # qkv weight + bias
        + (h // P) * h                            # out weight
        + h * (ff // P) + (ff // P)               # fc1 weight + bias
        + (ff // P) * h                           # fc2 weight
    )
    elems["blocks"] = L * per_block
    total = sum(elems.values())
    return {
        "breakdown_elems": elems,
        "param_elems": total,
        "grad_elems": total,
        "param_bytes": total * BYTES,
        "grad_bytes": total * BYTES,
    }


def activation_bytes_per_rank(cfg: dict) -> dict:
    """前向结束时每卡保留到反向的激活，按模块枚举。"""
    L, h, ff, heads, P = (cfg["layers"], cfg["hidden"], cfg["ff"],
                          cfg["heads"], cfg["world_size"])
    T = cfg["seq_len"] - 1
    sp = T // P
    hd = h // heads
    hpr = heads // P
    items: list[tuple[str, int]] = []
    for i in range(L):
        t = f"block{i}"
        items.append((f"{t}.ln1.xhat", sp * h))
        items.append((f"{t}.qkv.x_full", T * h))          # 列切缓存完整输入
        items.append((f"{t}.attn.probs", T * hpr * T))    # 完整序列注意力
        items.append((f"{t}.attn.out.x_full", T * (h // P)))
        items.append((f"{t}.ln2.xhat", sp * h))
        items.append((f"{t}.fc1.x_full", T * h))
        items.append((f"{t}.gelu", T * (ff // P)))
        items.append((f"{t}.fc2.x_full", T * (ff // P)))
    items.append(("ln_f.xhat", sp * h))          # 末层 LN 的归一化缓存
    total = sum(n for _, n in items)
    return {"items": items, "elems": total, "bytes": total * BYTES}


def communication_formula(cfg: dict) -> dict:
    """每种集合操作每步的次数与每卡通信元素数/字节。"""
    L, h, ff, V, P = (cfg["layers"], cfg["hidden"], cfg["ff"],
                      cfg["vocab"], cfg["world_size"])
    T = cfg["seq_len"] - 1
    sp = T // P
    counts = {"all_reduce": 0, "all_gather": 0, "reduce_scatter": 0}
    elems = {k: 0.0 for k in counts}

    def ar(n):
        counts["all_reduce"] += 1
        elems["all_reduce"] += ar_elements(n, P)

    def ag(n_full):
        counts["all_gather"] += 1
        elems["all_gather"] += ag_elements(n_full, P)

    def rs(n_full):
        counts["reduce_scatter"] += 1
        elems["reduce_scatter"] += rs_elements(n_full, P)

    # ---- 词嵌入 ----
    ar(cfg["seq_len"] * h)     # 部分嵌入求和
    rs(T * h)                  # 进入 SP（只分散，记账同 RS）
    ag(T * h)                  # 嵌入反向 all-gather

    for _ in range(L):
        # QKV 列切：前向 AG，反向 RS
        ag(T * h)
        rs(T * h)
        # O 行切：前向 RS，反向 AG
        rs(T * h)
        ag(T * h)
        # LN1 反向 2 次 all-reduce
        ar(h); ar(h)
        # FC1 列切
        ag(T * h)
        rs(T * h)
        # FC2 行切
        rs(T * h)
        ag(T * h)
        # LN2 反向 2 次
        ar(h); ar(h)

    # 末层 LN 反向 2 次
    ar(h); ar(h)

    # ---- 输出投影与 CE ----
    ag(T * h)                  # 输出前 SP->复制 all-gather
    ar(T)                      # CE max：每卡持有全部 T 个位置的本地统计
    ar(T)                      # CE sumexp
    ar(1)                      # CE 损失分量和
    ar(T * h)                  # 输出 dh 全词表求和
    rs(T * h)                  # 复制->SP 分散

    bytes_per_rank = {k: v * BYTES for k, v in elems.items()}
    return {
        "events": counts,
        "elements_per_rank": elems,
        "bytes_per_rank": bytes_per_rank,
        "total_bytes_cluster": {k: v * P for k, v in bytes_per_rank.items()},
    }
