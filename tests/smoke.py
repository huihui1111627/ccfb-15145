"""冒烟测试：损失 + 全部参数梯度与单卡参考逐项对比。"""
import sys
import numpy as np

sys.path.insert(0, "/Users/shizhihui/ccfb/runs/15145/workspace")

from tensor_parallel import (DeviceMesh, ParallelTransformer,
                             RefTransformer, make_params)


def parallel_to_ref_name(name: str) -> str:
    """模型里的参数名 -> 参考参数字典键。"""
    if name == "vocab.weight":
        return "embed.weight"
    if name.startswith("ln_f."):
        return name
    # block0.xxx -> blocks.0.xxx；fc1/fc2 补 mlp. 前缀
    if name.startswith("block"):
        dot = name.index(".")
        idx = name[len("block"):dot]
        rest = name[dot + 1:]
        if rest.startswith("fc1.") or rest.startswith("fc2."):
            rest = "mlp." + rest
        return f"blocks.{idx}.{rest}"
    raise KeyError(name)


def gather_full(st):
    if st.replicated:
        return st.shards[0]
    return np.concatenate(st.shards, axis=st.shard_axis)


def run(cfg, seed_ids=0, seed_params=42):
    rng = np.random.default_rng(seed_ids)
    ids = rng.integers(0, cfg["vocab"], size=cfg["seq_len"])
    params = make_params(cfg, seed=seed_params)

    ref = RefTransformer(cfg, params)
    loss_ref = ref.forward(ids)
    grads_ref = ref.backward()

    mesh = DeviceMesh(cfg["world_size"], base_seed=1)
    model = ParallelTransformer(mesh, cfg, params)
    mesh.begin_step()
    loss_tp = model.forward(ids, training=False)
    model.backward()
    return model, grads_ref, loss_ref, loss_tp


def main():
    cfg = {"layers": 1, "heads": 2, "hidden": 8, "ff": 16,
           "vocab": 13, "world_size": 2, "seq_len": 7, "dropout": 0.0}
    model, grads_ref, loss_ref, loss_tp = run(cfg)
    print(f"loss ref={loss_ref:.12f} tp={loss_tp:.12f} "
          f"diff={abs(loss_ref - loss_tp):.2e}")

    worst = 0.0
    checked = 0
    for st in model.parameters():
        ref_key = parallel_to_ref_name(st.name)
        g_full = gather_full(st.grad)
        g_ref = grads_ref[ref_key]
        if g_full.shape != g_ref.shape:  # 去词表 padding 行
            assert st.name == "vocab.weight"
            g_full = g_full[:g_ref.shape[0]]
        d = float(np.abs(g_full - g_ref).max())
        worst = max(worst, d)
        checked += 1
        status = "OK  " if d < 1e-9 else "FAIL"
        print(f"  [{status}] {st.name:34s} -> {ref_key:28s} {d:.2e}")
    print(f"checked {checked} params, worst grad diff={worst:.2e}")
    assert worst < 1e-9, "gradient mismatch"


if __name__ == "__main__":
    main()
