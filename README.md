# CRV — 容器平台自定义资源多版本管理组件

一个用 Go 实现的、仿真真实容器平台（CRD 风格）的多版本自定义资源管理系统。
集群里同时存在几十个资源类型、每种多个版本，不同版本的客户端混合读写，
对象始终以指定的**存储版本**落盘，请求版本与存储版本之间通过转换服务
**链式转换**，并保证往返无损。

## 需求与实现对照

| 需求 | 实现 |
| --- | --- |
| 每种资源指定一个存储版本，对象以存储版本落盘 | `pkg/storage`：Create/Update 先经转换服务转到 `Kind.StorageVersion` 再序列化落盘；Get/List 再转回请求版本 |
| 新版本字段转老版本不丢，藏在注解里，老客户端读改写不冲掉新字段 | `pkg/convert`：未知字段 sweep 进 `crv.io/stash.<version>` 注解，重新升级时按 Schema 类型恢复；保留型客户端（K8s 风格，原样回写未知注解）的 RMW 不丢字段 |
| 转换函数按相邻版本两两定义，任意版本链式转换；测试保证全组合往返一致 | `schema.ConversionRule` 只声明在相邻版本边上；`Engine.Convert` 沿版本链逐边转换；`TestRoundTripAllPairs` 覆盖所有版本组合 |
| 存储版本切换后后台逐个重写，迁移期读写不受影响，记录进度，迁完才能移除老版本 | `pkg/migrator`：`SwitchStorage` 立即对新写生效，后台用乐观 CAS（`MigrateOne`）逐个重写，进度可查；`CanRemoveVersion` 闸门要求零存量对象且无进行中迁移 |
| 转换服务故障/变慢时读写明确失败，不写半转换数据 | `convert.Service` 支持 Down/Slow/FailRate 故障注入；引擎在 scratch 副本上转换，失败则输入对象和存储都不变；存储层转换先于任何状态变更 |

## 四种版本演化规则（相邻版本边）

- `rename`：字段改名（反向自动还原）
- `add`：新版本新增字段，带默认值（降级时用户改过的值进 stash，未改的默认值不产生冗余注解）
- `split`：单字符串按分隔符拆成多个字段（`SplitN` 保留末段中多余的分隔符，合并时精确还原）
- `wrap_list`：单值变列表（多元素列表降级时整体进 stash，单元素折叠回标量）

## 目录结构

```
pkg/types       通用对象模型（Object/ObjectMeta），JSON 编解码
pkg/schema      资源定义注册表：版本、字段、相邻版本转换规则
pkg/convert     转换引擎（链式、注解 stash、类型还原）+ 转换服务（故障/指标）
pkg/storage     存储层：存储版本落盘、乐观并发、迁移 CAS 原语
pkg/migrator    后台存储版本迁移、进度、移除闸门
pkg/client      版本化客户端（保留型 / 激进型两种人格）
pkg/simulator   36 种资源 × 2~5 版本的确定性仿真与混合版本流量
pkg/acceptance  端到端验收场景
cmd/simulator   CLI 验收入口 + Web 控制台
```

## 运行

```bash
# 完整验收（几十种资源、混合流量、迁移、故障、闸门）
go run ./cmd/simulator -seed 42 -objects 5 -traffic 3000

# Web 控制台：资源版本/存储版本、迁移进度、对象多版本视图、故障注入
go run ./cmd/simulator -serve -addr :8080
```

控制台页面：
- 资源清单表：每种资源的全部版本、当前存储版本、对象按落盘版本的分布、相邻版本演化规则
- 迁移面板：选择资源和目标版本切换存储版本，实时进度条和重试计数
- 故障面板：对任意资源注入 Down / 200ms 延迟，或清除
- 对象视图：同一对象在每个服务版本下的完整形态，stash 注解高亮，以及实际落盘信封

## 测试

```bash
go test ./...                 # 单元 + 集成 + 仿真测试
go test -race ./...           # 竞态检测
```

关键测试：
- `convert.TestRoundTripAllPairs`：3 版本 × 所有版本组合往返无损
- `convert.TestSplitPreservesExtraSeparators`：多分隔符 split 精确还原
- `convert.TestMultiValueListSurvivesDowngrade`：多元素列表降级不丢
- `storage.TestConversionServiceFailureWritesNothing`：故障期零坏数据
- `migrator.TestMigrationSurvivesConversionOutageAndWritesNoBadData`：迁移中故障、退避重试、恢复后 drain
- `simulator.TestMixedTrafficLosesNoFields` / `TestTrafficUnderConversionFaults`：36 种资源混合流量
- `client.TestPreservingClientKeepsNewFields`：老客户端 RMW 不冲新字段

## 无损保证的边界

保留型客户端（不识别的注解原样回写，K8s 的真实行为）下，老客户端的
读改写不会冲掉任何新字段。若客户端**主动剥离**自己不认识的注解
（`client.Aggressive`），stash 随之丢失——系统不会产生半转换坏数据，
对象在它仍支持的所有版本下依然一致可读，`client_test.go` 明确记录了
这条边界。
