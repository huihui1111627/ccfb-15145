# 张量并行训练框架（numpy 模拟多卡）

面向玻璃温室花卉育种语言模型的小型张量并行训练框架。用进程内的 numpy 数组
模拟多张设备卡，卡间通信用带通信量记账的模拟集合操作实现，并附带一个展示
切分方式、显存、通信量与损失曲线的自包含 HTML 页面。

## 设计与八条要求的对应

| 要求 | 实现位置 |
| --- | --- |
| 1. 线性层列切/行切配对，中间不通信，出口一次归约 | `layers.py` `ColumnLinear` / `RowLinear`，列切入口 all-gather、行切出口 reduce-scatter |
| 2. 注意力按头切分，每卡只算自己的头 | `layers.py` `MultiHeadAttention`，QKV 交错布局 `(T, 头, 3, 头维)` |
| 3. 词嵌入按词表切，查不到填零，求和得完整嵌入 | `vocab.py` `TiedVocab.embed_forward`（all-reduce 求和） |
| 4. 输出层按词表切，CE 不收齐分数，跨卡求 max/sumexp | `vocab.py` `ParallelCrossEntropy`，3 次 all-reduce，log-sum-exp 防溢出 |
| 5. 输出层与词嵌入共享权重，梯度累加 | 同一个 `ShardedTensor`，`embed_backward` 与 `logits_backward` 都 `add_grad` |
| 6. LN/Dropout 按序列长度切，归约↔分散配对 | 全程序列并行（SP），列切/行切边界的 all-gather 与 reduce-scatter 配对 |
| 7. 丢弃随机数两套状态 | `mesh.py` `replica_rng`（各卡相同）/ `local_rng`（各卡不同），`layers.py` `Dropout` |
| 8. 手写反向，前/反向通信配对 | 每个层都实现 `forward`/`backward`，AG↔RS、all-reduce 成对出现 |

## 目录结构

```
tensor_parallel/
  mesh.py       模拟设备网格、集合通信（环形记账）、双路 RNG、显存记账
  tensor.py     ShardedTensor、等分/补齐切分
  module.py     模块基类、激活显存登记
  layers.py     Dropout/LN/列切线性/行切线性/GELU/按头注意力
  vocab.py      共享词表权重（嵌入+输出）、词表并行交叉熵
  model.py      Transformer 块与整体模型的前向/反向编排
  reference.py  单卡不切分参考模型（同样手写反向），作为数值基准
  params.py     确定性参数初始化、随机合法配置生成
  stats.py      切分方式、显存、通信量的解析公式
  train.py      SGD 训练循环与损失曲线
tests/
  test_framework.py  完整测试套件（梯度对账/数值稳定/RNG/通信显存公式/单卡）
  smoke.py           最小配置的快速梯度冒烟测试
reports/
  generate_data.py  生成报告 JSON
  build_html.py     生成自包含 report.html
  report.html       可视化页面（双击打开）
```

## 运行

```bash
# 完整测试（36 组随机配置的损失+全梯度对账，及通信/显存公式核对）
python3 -I tests/test_framework.py

# 冒烟测试
python3 -I tests/smoke.py

# 生成可视化页面
python3 -I reports/generate_data.py
python3 -I reports/build_html.py
# 浏览器打开 reports/report.html
```

所有计算使用 float64，并行与单卡参考的损失、参数梯度误差在 1e-15 量级。

## 关键数值技巧

- **词表除不尽**：权重先补齐到卡数的等倍再等分，padding 行在 logits 置
  `-inf`、在梯度处掩码为零，本地索引按等分起点计算，避免末卡错位。
- **交叉熵防溢出**：先跨卡 all-reduce 全局最大值，平移后再求指数和；
  重复的 `max+logsumexp` 项只在 rank0 计一次，`-z_label` 按词表分片贡献。
- **二维切分规避**：输出投影前先 all-gather 序列使隐藏态复制，logits 退化为
  纯词表一维切分，CE 才能只靠标量统计量归约；反向配对 reduce-scatter。
- **通信量公式（环形，每卡口径）**：all-reduce `2(P-1)/P·n`，
  all-gather / reduce-scatter `(P-1)/P·n_full`，字节 = 元素 × 8。
