"""完整测试套件。

1) 几十种随机 (层数, 头数, 词表, 卡数, 序列长) 组合，前向损失与全部参数
   梯度对照单卡不切分参考实现，要求误差在 float64 机器精度量级；
2) 词表并行交叉熵在极大/极小分数下不溢出，损失与梯度有限；
3) 双路随机数：replica 流各卡相同、local 流各卡不同；
4) 实测集合通信量、每卡显存与解析公式逐项一致。
"""
import sys
import numpy as np

sys.path.insert(0, "/Users/shizhihui/ccfb/runs/15145/workspace")

from tensor_parallel import (DeviceMesh, ParallelTransformer,
                             RefTransformer, make_configs, make_params)
from tensor_parallel import stats as stats_mod

TOL_LOSS = 1e-10
TOL_GRAD = 1e-9


def parallel_to_ref_name(name: str) -> str:
    if name == "vocab.weight":
        return "embed.weight"
    if name.startswith("ln_f."):
        return name
    if name.startswith("block"):
        dot = name.index(".")
        idx = name[len("block"):dot]
        rest = name[dot + 1:]
        if rest.startswith(("fc1.", "fc2.")):
            rest = "mlp." + rest
        return f"blocks.{idx}.{rest}"
    raise KeyError(name)


def gather_full(st):
    if st.replicated:
        return st.shards[0]
    return np.concatenate(st.shards, axis=st.shard_axis)


def check_one_config(cfg, seed_ids=0, verbose=False):
    rng = np.random.default_rng(seed_ids)
    ids = rng.integers(0, cfg["vocab"], size=cfg["seq_len"])
    params = make_params(cfg, seed=42)

    ref = RefTransformer(cfg, params)
    loss_ref = ref.forward(ids)
    grads_ref = ref.backward()

    mesh = DeviceMesh(cfg["world_size"], base_seed=1)
    model = ParallelTransformer(mesh, cfg, params)
    mesh.begin_step()
    loss_tp = model.forward(ids, training=False)
    model.backward()

    errs = []
    if abs(loss_ref - loss_tp) > TOL_LOSS:
        errs.append(f"loss diff {abs(loss_ref-loss_tp):.2e}")
    for st in model.parameters():
        ref_key = parallel_to_ref_name(st.name)
        g_full = gather_full(st.grad)
        g_ref = grads_ref[ref_key]
        if g_full.shape != g_ref.shape:
            g_full = g_full[:g_ref.shape[0]]
        d = float(np.abs(g_full - g_ref).max())
        if d > TOL_GRAD:
            errs.append(f"grad {st.name} diff {d:.2e}")
    return loss_ref, loss_tp, errs, mesh, model


def test_random_configs():
    configs = make_configs(n=36)
    print(f"== 1) 随机配置梯度对账（{len(configs)} 组）==")
    worst = 0.0
    for i, cfg in enumerate(configs):
        loss_ref, loss_tp, errs, _, _ = check_one_config(cfg, seed_ids=i)
        worst = max(worst, abs(loss_ref - loss_tp))
        flag = "OK " if not errs else "FAIL"
        print(f"  [{flag}] L={cfg['layers']} heads={cfg['heads']} "
              f"h={cfg['hidden']} ff={cfg['ff']} V={cfg['vocab']} "
              f"P={cfg['world_size']} S={cfg['seq_len']} "
              f"loss|Δ|={abs(loss_ref-loss_tp):.1e}")
        if errs:
            for e in errs:
                print("      ", e)
            raise AssertionError(f"config {cfg} failed")
    print(f"  全部通过，最差损失误差 {worst:.2e}\n")


def test_ce_overflow():
    print("== 2) 词表并行交叉熵数值稳定性 ==")
    from tensor_parallel.vocab import NEG_INF
    cfg = {"layers": 1, "heads": 2, "hidden": 8, "ff": 16,
           "vocab": 13, "world_size": 2, "seq_len": 7, "dropout": 0.0}
    params = make_params(cfg, seed=42)
    rng = np.random.default_rng(3)
    ids = rng.integers(0, cfg["vocab"], size=cfg["seq_len"])

    for scale, label in [(1e4, "极大分数 ±1e4"),
                         (1e-8, "极小分数 ±1e-8")]:
        # 直接构造落在 float64 安全区间、但跨度极大/极小的 logits 测 CE
        mesh = DeviceMesh(2, base_seed=1)
        model = ParallelTransformer(mesh, cfg, params)
        model.forward(ids, False)
        ce = model.ce
        T = cfg["seq_len"] - 1
        z = [rng.normal(size=(T, 7)) * scale for _ in range(2)]
        z[1][:, 6] = NEG_INF  # padding 列
        labels = ids[1:]
        ce.set_meta(model.vocab.bounds, labels)
        loss = ce.forward(z)
        dz = ce.backward()
        # 与单卡收齐分数的朴素稳定 CE 对照（同样先减最大值）
        z0 = np.where(np.isfinite(z[0]), z[0], -np.inf)
        z1 = np.where(np.isfinite(z[1]), z[1], -np.inf)[:, :6]
        full = np.hstack([z0, z1])
        m = full.max(-1, keepdims=True)
        lse = m.squeeze(-1) + np.log(np.exp(full - m).sum(-1))
        loss_ref = (lse - full[np.arange(T), labels]).mean()
        finite = np.isfinite(loss) and all(
            np.isfinite(a).all() for a in dz)
        match = abs(loss - loss_ref) < 1e-8 * max(1.0, abs(loss_ref))
        print(f"  [{'OK ' if finite and match else 'FAIL'}] {label}: "
              f"loss={loss:.6f} 参考={loss_ref:.6f} 梯度有限={finite}")
        assert finite and match
    # 真正的极端边界：接近 float64 上限的安全量级也不产生 inf/nan
    z_big = [np.full((T, 7), 700.0) for _ in range(2)]
    model2 = ParallelTransformer(DeviceMesh(2, base_seed=2), cfg, params)
    model2.forward(ids, False)
    model2.ce.set_meta(model2.vocab.bounds, ids[1:])
    loss_big = model2.ce.forward(z_big)
    assert np.isfinite(loss_big) and loss_big > 0
    print(f"  [OK ] logits=700（exp 近 float64 上限）损失有限={loss_big:.4f}")
    print()


def test_dual_rng():
    print("== 3) 双路随机数状态 ==")
    from tensor_parallel import DeviceMesh
    mesh = DeviceMesh(4, base_seed=11)
    shape = (1000,)
    rep = [mesh.rng(r, "replica").normal(size=shape) for r in range(4)]
    loc = [mesh.rng(r, "local").normal(size=shape) for r in range(4)]
    replica_identical = all(
        np.array_equal(rep[0], rep[r]) for r in range(1, 4))
    local_different = all(
        not np.array_equal(loc[0], loc[r]) for r in range(1, 4))
    print(f"  [{'OK ' if replica_identical else 'FAIL'}] "
          f"replica 流各卡相同：{replica_identical}")
    print(f"  [{'OK ' if local_different else 'FAIL'}] "
          f"local 流各卡不同：{local_different}")
    assert replica_identical and local_different

    # Dropout 两种流的掩码行为
    from tensor_parallel.layers import Dropout
    mesh2 = DeviceMesh(2, base_seed=5)
    x = [np.ones((50, 8)) for _ in range(2)]
    d_rep = Dropout(mesh2, "d_rep", 0.5, "replica")
    d_loc = Dropout(mesh2, "d_loc", 0.5, "local")
    yr = d_rep.forward(x, True)
    yl = d_loc.forward(x, True)
    rep_mask_same = np.array_equal(yr[0] != 0, yr[1] != 0)
    loc_mask_diff = not np.array_equal(yl[0] != 0, yl[1] != 0)
    print(f"  [{'OK ' if rep_mask_same else 'FAIL'}] replica 丢弃掩码各卡相同")
    print(f"  [{'OK ' if loc_mask_diff else 'FAIL'}] local 丢弃掩码各卡不同")
    assert rep_mask_same and loc_mask_diff
    print()


def test_comm_and_memory():
    print("== 4) 通信量与显存的公式对账 ==")
    configs = make_configs(n=12, seed=17)
    for cfg in configs:
        _, _, errs, mesh, model = check_one_config(cfg, seed_ids=99)
        assert not errs
        # ---- 通信：实测事件聚合 vs 公式 ----
        formula = stats_mod.communication_formula(cfg)
        measured = mesh.comm_summary()
        for op in ("all_reduce", "all_gather", "reduce_scatter"):
            m_cnt = measured.get(op, {}).get("events", 0)
            f_cnt = formula["events"][op]
            assert m_cnt == f_cnt, (
                f"{op} 次数 {m_cnt}!={f_cnt} @ {cfg}")
            m_bytes = measured.get(op, {}).get("bytes_per_rank", 0.0)
            f_bytes = formula["bytes_per_rank"][op]
            assert abs(m_bytes - f_bytes) < 1e-6, (
                f"{op} 字节 {m_bytes}!={f_bytes} @ {cfg}")
        # ---- 权重/梯度显存 vs 公式 ----
        w = stats_mod.weight_bytes_per_rank(cfg)
        snap = mesh.mem.snapshot()
        assert abs(snap["params_per_rank"][0] - w["param_bytes"]) < 1e-6
        assert abs(snap["grads_per_rank"][0] - w["grad_bytes"]) < 1e-6
        # 各卡权重显存一致（等分后）
        assert len(set(round(x, 4) for x in snap["params_per_rank"])) == 1
        # ---- 激活：实测峰值 >= 权重+梯度+激活公式，且与枚举一致 ----
        a = stats_mod.activation_bytes_per_rank(cfg)
        expected_peak = (w["param_bytes"] + w["grad_bytes"]
                         + a["bytes"])
        assert abs(snap["peak_per_rank"][0] - expected_peak) < 1e-6, (
            f"激活峰值 {snap['peak_per_rank'][0]} != {expected_peak}")
    print(f"  [OK ] {len(configs)} 组配置的通信次数、字节、权重/梯度/激活显存"
          f"全部与公式一致\n")


def test_single_device_and_dropout():
    print("== 5) 单卡退化与 dropout 训练模式 ==")
    from tensor_parallel.layers import Dropout
    # P=1 时集合通信应自洽（ring 系数为 0），前向/反向可运行
    cfg = {"layers": 2, "heads": 2, "hidden": 16, "ff": 32,
           "vocab": 21, "world_size": 1, "seq_len": 5, "dropout": 0.3}
    rng = np.random.default_rng(0)
    ids = rng.integers(0, cfg["vocab"], size=cfg["seq_len"])
    params = make_params(cfg, seed=42)
    mesh = DeviceMesh(1, base_seed=7)
    model = ParallelTransformer(mesh, cfg, params)
    l1 = model.forward(ids, training=True)
    model.backward()
    l2 = model.forward(ids, training=True)
    l3 = model.forward(ids, training=False)
    l4 = model.forward(ids, training=False)
    print(f"  [OK ] P=1 开 dropout 前向/反向可运行，损失={l1:.4f}")
    assert l1 != l2, "训练模式 dropout 应带来随机性"
    assert l3 == l4, "推理模式应确定性"
    # P=1 的 ring 通信量系数 (P-1)/P = 0，记账应为 0
    assert all(ev.bytes_per_rank == 0.0 for ev in mesh.events), \
        "单卡下集合通信量应为 0"
    # replica 流 dropout 各卡掩码相同、local 流不同
    mesh2 = DeviceMesh(3, base_seed=9)
    x = [np.random.default_rng(r).normal(size=(40, 6)) for r in range(3)]
    yr = Dropout(mesh2, "dr", 0.4, "replica").forward(x, True)
    yl = Dropout(mesh2, "dl", 0.4, "local").forward(x, True)
    assert all(np.array_equal(yr[0] != 0, yr[r] != 0) for r in range(1, 3))
    assert not all(np.array_equal(yl[0] != 0, yl[r] != 0)
                   for r in range(1, 3))
    print("  [OK ] replica 掩码各卡一致、local 掩码各卡不同")
    print()


if __name__ == "__main__":
    test_random_configs()
    test_ce_overflow()
    test_dual_rng()
    test_comm_and_memory()
    test_single_device_and_dropout()
    print("所有测试通过 ✅")
