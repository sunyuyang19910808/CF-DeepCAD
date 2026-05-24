# Constraint-Fused DeepCAD Simplify Modify2 任务清单

> 来源文档：`Constraint-Fused DeepCAD Simplify Modify2 DDD技术方案.md`
>
> 目标：将最新 DDD 方案拆解为可执行、可记录、可验证、可串行推进的任务清单。实现口径与最新 DDD 保持一致：**仅将约束范围从五类收敛为四类，系统移除 `Collinear`，其余上下文划分、模块边界、训练/推理主路径与原始 Constraint-Fused DeepCAD 保持一致。**

## 0. DDD 映射速览

| DDD 限界上下文 | 本清单主要落点 |
| --- | --- |
| Sketch Preparation | P1-02、P1-03、P3-02 |
| Constraint-Fused Encoding | P1-04、P2-01、P2-02、P2-03、`ConstraintFusionDomainService` |
| Generation | P2-05、P3-01 |
| Training Orchestration | P2-06、P3-02、P3-03、P3-04、P4 系列 |

## 1. 使用规则

1. 本清单默认按顺序执行，不建议跳过 Gate 直接进入后续任务。
2. 每个可测试任务实现完成后，必须先完成验证并记录结果，再进入下一任务。
3. 若验证失败，应在当前任务下记录阻塞原因与修复动作。
4. 测试与阶段验收应优先依赖仓库 `data/` 下真实 DeepCAD 数据，而不是只依赖合成张量。
5. 实现边界遵循最新 DDD：不改 DeepCAD 命令表示、不新增人工标注、推理主路径为 latent-only。

## 2. 标记约定

- 完成状态：`[ ] 未完成` / `[x] 已完成`
- 验证状态：`[ ] 未验证` / `[ ] 验证中` / `[x] 已通过` / `[ ] 未通过`
- 阻塞状态：`[ ] 无` / `[ ] 有`

## 3. 总控看板

| ID | 阶段 | 任务 | 可测试 | 前置 | 完成 | 验证 |
| --- | --- | --- | --- | --- | --- | --- |
| P0-01 | Phase 0 | 范围冻结、目录映射与配置基线 | 否 | 无 | `[ ]` | `N/A` |
| P1-01 | Phase 1 | 领域实体、值对象与聚合根骨架 | 是 | P0-01 | `[ ]` | `[ ]` |
| P1-02 | Phase 1 | 离线约束提取与标准化 | 是 | P1-01 | `[ ]` | `[ ]` |
| P1-03 | Phase 1 | Batch Assembler、标签与监督张量 | 是 | P1-02 | `[ ]` | `[ ]` |
| P1-04 | Phase 1 | 命令级约束标记嵌入 | 是 | P1-03 | `[ ]` | `[ ]` |
| G1 | Gate 1 | 四类约束数据闭环放行 | 是 | P1-04 | `[ ]` | `[ ]` |
| P2-01 | Phase 2 | Constraint Token Encoder 与 Segment Embedding | 是 | G1 | `[ ]` | `[ ]` |
| P2-02 | Phase 2 | EncoderFused 联合编码链路 | 是 | P2-01 | `[ ]` | `[ ]` |
| P2-03 | Phase 2 | Pooling 与 Bottleneck | 是 | P2-02 | `[ ]` | `[ ]` |
| P2-04 | Phase 2 | 约束重建头与重建损失 | 是 | P2-03 | `[ ]` | `[ ]` |
| P2-05 | Phase 2 | Decoder 适配器与约束预测头 | 是 | P2-04 | `[ ]` | `[ ]` |
| P2-06 | Phase 2 | 可微几何约束评估器、总损失与单 Batch 训练用例 | 是 | P2-05 | `[ ]` | `[ ]` |
| G2 | Gate 2 | 四类约束模型主链路放行 | 是 | P2-06 | `[ ]` | `[ ]` |
| P3-01 | Phase 3 | latent-only 推理用例 | 是 | G2 | `[ ]` | `[ ]` |
| P3-02 | Phase 3 | 约束满足率评估用例 | 是 | P3-01 | `[ ]` | `[ ]` |
| P3-03 | Phase 3 | Repository、数据映射与 checkpoint 抽象 | 是 | P3-02 | `[ ]` | `[ ]` |
| P3-04 | Phase 3 | 配置、入口脚本、日志监控 | 是 | P3-03 | `[ ]` | `[ ]` |
| P3-05 | Phase 3 | 实验追踪与论文证据链 | 是 | P3-04 | `[ ]` | `[ ]` |
| G3 | Gate 3 | 训练、推理、评估闭环放行 | 是 | P3-05 | `[ ]` | `[ ]` |
| P4-01 | Phase 4 | Pooling / Bottleneck / 损失权重调优 | 是 | G3 | `[ ]` | `[ ]` |
| P4-02 | Phase 4 | 可选解码器 Cross-Attn 与 dropout schedule 消融 | 是 | G3 | `[ ]` | `[ ]` |
| P4-03 | Phase 4 | 与原始五类版对比实验与论文导出 | 是 | P4-01 | `[ ]` | `[ ]` |
| G4 | Gate 4 | 最终验收放行 | 是 | P4-03 | `[ ]` | `[ ]` |

## 4. 全局记录位

- 当前阶段：`Phase 0`
- 当前进行中的任务 ID：`P0-01`
- 最近一次通过的 Gate：`____`
- 设计基线：`论文尝试/约束感知生成/5_constraint_fused_deepcad_simplify_modify2/Constraint-Fused DeepCAD Simplify Modify2 DDD技术方案.md`
- 参考架构：`论文尝试/约束感知生成/2_constraint_fused_deepcad/Constraint-Fused DeepCAD DDD技术方案.md`
- 建议实现目录：`constraint_fused_deepcad/`
- 真实数据根路径 `DATA_ROOT`：`data`
- 最近一次验证时间：`____`
- 最近一次失败任务：`____`
- 最近一次失败原因：`____`

## 5. 任务明细

### P0-01 范围冻结、目录映射与配置基线

- 完成状态：`[ ] 未完成`  `[ ] 已完成`
- 可测试：否
- 前置任务：无
- 目标：确认最新 Modify2 DDD 的唯一业务差异是去掉 `Collinear`，并把实现边界、目录映射、配置基线固定下来。
- 实现清单：
- 明确保留 4 个限界上下文与原始 Fused 一致的模块边界
- 明确约束范围为 `Horizontal / Vertical / Parallel / Perpendicular`
- 明确 `constraint_tags=(S,N,4)`、`pair_gt=(N,L,L,2)`、`n_constraint_types=5`
- 对齐最新 DDD 中的目录建议、入口脚本与配置项
- 产出物：
- 范围冻结说明
- 目录映射说明
- 配置基线说明

### P1-01 领域实体、值对象与聚合根骨架

- 完成状态：`[ ] 未完成`  `[ ] 已完成`
- 验证状态：`[ ] 未验证`
- 可测试：是
- 前置任务：P0-01
- 目标：实现 `CadCommand`、`ConstraintRelation`、`ConstraintTagVector`、`ConstraintAwareLatent`、`SketchSequenceAggregate` 的基础结构。
- 实现清单：
- 定义实体字段与基础约束
- 固定 `constraint_tags/unary_gt/pair_gt` 维度
- 聚合根 `validate()` 只接受四类真实约束
- 建议验证：
- 最小样本下字段完整、shape 正确
- 非法 type id、非法 line index 能被拒绝
- 验证证据：
- 测试文件/命令：`____`
- 结果摘要：`____`

### P1-02 离线约束提取与标准化

- 完成状态：`[ ] 未完成`  `[ ] 已完成`
- 验证状态：`[ ] 未验证`
- 可测试：是
- 前置任务：P1-01
- 目标：实现 `ConstraintExtractor`，从 DeepCAD JSON 命令解析几何并自动推导四类约束字典，再映射为标准 `ConstraintRelation`。
- 实现清单：
- 保留 `horizontal / vertical / parallel / perpendicular` 四类提取
- 删除 `Collinear` 判定、输出键与映射逻辑
- 保持 `ANGLE_THRESH` 等阈值口径稳定
- 建议验证：
- 人工构造案例验证四类提取结果正确
- 真实样本中不出现 `collinear` 字段
- **真实数据**：抽样 ≥32 条样本跑通“读 JSON → 线段恢复 → 约束字典 → `ConstraintRelation`”
- 验证证据：
- 测试文件/命令：`____`
- 结果摘要：`____`

### P1-03 Batch Assembler、标签与监督张量

- 完成状态：`[ ] 未完成`  `[ ] 已完成`
- 验证状态：`[ ] 未验证`
- 可测试：是
- 前置任务：P1-02
- 目标：实现 `ConstraintBatchAssembler`，生成 `constraint_tags`、`constraint_tokens`、`unary_gt`、`pair_gt` 与 mask。
- 实现清单：
- `constraint_tags` 输出 `(S,N,4)`
- `unary_gt` 输出 `(N,L,2)`
- `pair_gt` 输出 `(N,L,L,2)`
- token padding 使用 `NONE=4`
- 建议验证：
- `build_constraint_tags()` 对 unary/pair 映射正确
- `pair_gt.shape[-1] == 2`
- **真实数据**：组 ≥2 个 batch，检查 shape 与最新 DDD 的 `DATA SHAPES` 一致
- 验证证据：
- 测试文件/命令：`____`
- 结果摘要：`____`

### P1-04 命令级约束标记嵌入

- 完成状态：`[ ] 未完成`  `[ ] 已完成`
- 验证状态：`[ ] 未验证`
- 可测试：是
- 前置任务：P1-03
- 目标：实现 `ConstraintTagEmbedding` 与 `CADEmbeddingFused` 中的 tag 注入逻辑。
- 实现清单：
- `ConstraintTagEmbedding(4, d_model)`
- tag 与 command / arg / group embedding 的融合
- 保持位置编码与原始命令表示兼容
- 建议验证：
- tag embedding 前向 shape 正确
- 无约束样本与有约束样本都能稳定前向
- 验证证据：
- 测试文件/命令：`____`
- 结果摘要：`____`

### G1 四类约束数据闭环放行

- 完成状态：`[ ] 未完成`  `[ ] 已完成`
- 验证状态：`[ ] 未验证`
- 前置任务：P1-04
- 放行标准：
- `Collinear` 已从 type、extractor、batch 契约中移除
- `constraint_tags/unary_gt/pair_gt` 维度符合最新 DDD
- 从原始输入到 command embedding 的数据流已闭合
- 结论：`____`

### P2-01 Constraint Token Encoder 与 Segment Embedding

- 完成状态：`[ ] 未完成`  `[ ] 已完成`
- 验证状态：`[ ] 未验证`
- 可测试：是
- 前置任务：G1
- 目标：实现 `ConstraintTokenEncoder` 与 `SegmentEmbedding`，完成约束 token 的显式编码。
- 实现清单：
- `n_types=5`（4 类真实约束 + `NONE/PAD`）
- `type_embed`、`line_embed`、`pair_fuse`、`out_proj`、`norm`
- 命令段 / 约束段二值 segment 标记
- 建议验证：
- 输出 shape 正确
- padding token 不越界
- 多约束共享线段场景编码稳定
- 验证证据：
- 测试文件/命令：`____`
- 结果摘要：`____`

### P2-02 EncoderFused 联合编码链路

- 完成状态：`[ ] 未完成`  `[ ] 已完成`
- 验证状态：`[ ] 未验证`
- 可测试：是
- 前置任务：P2-01
- 目标：实现 `EncoderFused`，把命令流、约束流、segment embedding、联合 mask 与 encoder 串起来。
- 实现清单：
- 构建 `E_cmd` 与 `E_con`
- 拼接 `E_joint` 与 `mask_joint`
- 接入 `TransformerEncoder`
- 输出 `memory` 与 `z_pre`
- 建议验证：
- 不同 `S/T_c` 下 shape 稳定
- batch 内混合长度样本时 mask 行为正确
- 验证证据：
- 测试文件/命令：`____`
- 结果摘要：`____`

### P2-03 Pooling 与 Bottleneck

- 完成状态：`[ ] 未完成`  `[ ] 已完成`
- 验证状态：`[ ] 未验证`
- 可测试：是
- 前置任务：P2-02
- 目标：实现 `MaskedMeanPooling + BottleneckAdapter`，保持与 decoder 输入接口对齐。
- 实现清单：
- 实现 `MaskedMeanPooling`
- 实现 `BottleneckAdapter`
- 预留双流池化扩展接口
- 建议验证：
- 全 padding 边界不除零
- 输出维度满足 decoder 输入要求
- 重复前向结果稳定
- 验证证据：
- 测试文件/命令：`____`
- 结果摘要：`____`

### P2-04 约束重建头与重建损失

- 完成状态：`[ ] 未完成`  `[ ] 已完成`
- 验证状态：`[ ] 未验证`
- 可测试：是
- 前置任务：P2-03
- 目标：实现 `ConstraintReconHead` 与 `weighted_bce()`，为 latent 注入四类约束重建监督。
- 实现清单：
- unary 输出 2 维
- pair 输出 2 维
- `pos_weight` 支持稀疏标签加权
- 建议验证：
- `unary_pred.shape[-1] == 2`
- `pair_pred.shape[-1] == 2`
- loss 数值有限且不 shape 错配
- 验证证据：
- 测试文件/命令：`____`
- 结果摘要：`____`

### P2-05 Decoder 适配器与约束预测头

- 完成状态：`[ ] 未完成`  `[ ] 已完成`
- 验证状态：`[ ] 未验证`
- 可测试：是
- 前置任务：P2-04
- 目标：实现 `ConstraintAwareDecoderAdapter` 与 `ConstraintPredHead`，默认保持 latent-only 主路径。
- 实现清单：
- 对接现有 DeepCAD 同构 decoder
- 约束预测头输出 4 维
- 为可选 Cross-Attn 预留接口，但默认关闭
- 建议验证：
- 仅依赖 latent 即可完成前向
- `constraint_pred_logits.shape[-1] == 4`
- 推理接口不强制外部 `C`
- 验证证据：
- 测试文件/命令：`____`
- 结果摘要：`____`

### P2-06 可微几何约束评估器、总损失与单 Batch 训练用例

- 完成状态：`[ ] 未完成`  `[ ] 已完成`
- 验证状态：`[ ] 未验证`
- 可测试：是
- 前置任务：P2-05
- 目标：实现 `DifferentiableSketchInterpreter`、`DifferentiableConstraintEvaluator`、`LossComposer` 与 `TrainConstraintFusedBatchUseCase`，打通单次训练闭环。
- 实现清单：
- 几何解释链路：`cmd/arg logits -> soft dequantization -> interpreter -> evaluator`
- 残差只保留 H/V/Parallel/Perpendicular
- 总损失：`L_cmd + α·L_constraint_pred + β·L_constraint_recon + γ·L_geom_constraint`
- 单 batch 前向、反向与记录逻辑
- 建议验证：
- 1 个 batch 前向与反向可运行
- `geom_loss` 数值有限
- 无 `Collinear` 残差分支
- **真实数据**：使用 `DATA_ROOT` 取 ≥1 个 batch 完成前向+反向
- 验证证据：
- 测试文件/命令：`____`
- 结果摘要：`____`

### G2 四类约束模型主链路放行

- 完成状态：`[ ] 未完成`  `[ ] 已完成`
- 验证状态：`[ ] 未验证`
- 前置任务：P2-06
- 放行标准：
- Encoder / Pooling / Bottleneck / Recon / Decoder / Geom loss 主链路已打通
- 所有核心输出维度与最新 DDD 一致
- 不存在 `Collinear` 相关 head 或 loss
- 结论：`____`

### P3-01 latent-only 推理用例

- 完成状态：`[ ] 未完成`  `[ ] 已完成`
- 验证状态：`[ ] 未验证`
- 可测试：是
- 前置任务：G2
- 目标：实现 `GenerateFromLatentUseCase`，保证仅基于 latent 生成 CAD 序列。
- 实现清单：
- 支持 encoder 输出 latent 的推理
- 支持随机 latent 的推理
- decoder 统一只消费 `z`
- 建议验证：
- 自编码路径可生成
- 随机 latent 可生成合法序列
- **真实数据**：对 ≥8 条样本走“编码 → latent-only 解码”全链路
- 验证证据：
- 测试文件/命令：`____`
- 结果摘要：`____`

### P3-02 约束满足率评估用例

- 完成状态：`[ ] 未完成`  `[ ] 已完成`
- 验证状态：`[ ] 未验证`
- 可测试：是
- 前置任务：P3-01
- 目标：实现 `EvaluateConstraintSatisfactionUseCase`，建立四类约束一致的评估口径。
- 实现清单：
- 从解码结果恢复线段与关系
- 统计 `R_h`、`R_v`、`parallel_recall`、`perpendicular_recall`
- 结合几何指标与拓扑规则
- 建议验证：
- summary 中不再有 `collinear` 字段
- 四类指标稳定输出
- **真实数据**：在 val 或等价切分上完成 ≥100 条样本评估
- 验证证据：
- 测试文件/命令：`____`
- 结果摘要：`____`

### P3-03 Repository、数据映射与 checkpoint 抽象

- 完成状态：`[ ] 未完成`  `[ ] 已完成`
- 验证状态：`[ ] 未验证`
- 可测试：是
- 前置任务：P3-02
- 目标：按 DDD 分层完成 `SketchRepository`、`ModelCheckpointRepository` 与基础设施落地。
- 实现清单：
- 数据映射为 `SketchSequenceAggregate`
- checkpoint 保存、加载与恢复
- application / domain / infrastructure 依赖方向明确
- 建议验证：
- `SketchRepository -> Aggregate` 不丢字段
- checkpoint 保存与恢复成功
- **真实数据**：连续加载 ≥200 个样本不崩溃
- 验证证据：
- 测试文件/命令：`____`
- 结果摘要：`____`

### P3-04 配置、入口脚本、日志监控

- 完成状态：`[ ] 未完成`  `[ ] 已完成`
- 验证状态：`[ ] 未验证`
- 可测试：是
- 前置任务：P3-03
- 目标：把参数统一收敛到 `config/`，并提供 `train.py`、`infer.py`、`evaluate.py` 与监控指标输出。
- 实现清单：
- 集中式配置
- 训练、推理、评估入口
- 记录 `L_cmd`、重建分项、`L_geom_constraint`、约束满足率等指标
- 建议验证：
- 三个入口脚本均可加载配置并启动
- 日志与监控字段完整
- **真实数据**：各入口在 `DATA_ROOT` 上至少完成一次短跑
- 验证证据：
- 测试文件/命令：`____`
- 结果摘要：`____`

### P3-05 实验追踪与论文证据链

- 完成状态：`[ ] 未完成`  `[ ] 已完成`
- 验证状态：`[ ] 未验证`
- 可测试：是
- 前置任务：P3-04
- 目标：实现 `ExperimentTracker`、`manifest.json`、`train_metrics.csv`、`eval_metrics.json`、`best_checkpoint.txt`、`qualitative_cases.json` 的组织与落盘。
- 实现清单：
- 建立 `exp_id` 主键关联规则
- 保存配置快照、环境信息与代码版本
- 记录训练曲线、评估指标、最优 checkpoint 与定性案例
- 建议验证：
- 单次实验可生成完整证据目录
- 不同实验 `exp_id` 不互相覆盖
- 验证证据：
- 测试文件/命令：`____`
- 结果摘要：`____`

### G3 训练、推理、评估闭环放行

- 完成状态：`[ ] 未完成`  `[ ] 已完成`
- 验证状态：`[ ] 未验证`
- 前置任务：P3-05
- 放行标准：
- 训练、推理、评估三条主链路均已打通
- `SketchRepository -> Aggregate -> Encoder -> Decoder` 全链路不丢字段
- 监控与证据链可支撑复现实验与论文记录
- 结论：`____`

### P4-01 Pooling / Bottleneck / 损失权重调优

- 完成状态：`[ ] 未完成`  `[ ] 已完成`
- 验证状态：`[ ] 未验证`
- 可测试：是
- 前置任务：G3
- 目标：围绕 `pooling_strategy`、bottleneck 宽度、`alpha`、`beta`、`gamma`、`pos_weight` 做受控实验，定位稳定区间。
- 实现清单：
- `masked_mean` 与可选双流池化实验
- bottleneck 与损失权重对比
- 记录约束满足率与收敛稳定性变化
- 建议验证：
- `L_cmd` 与约束损失可协同下降
- 至少完成 1 组对照实验
- 验证证据：
- 实验记录路径：`____`
- 结果摘要：`____`

### P4-02 可选解码器 Cross-Attn 与 dropout schedule 消融

- 完成状态：`[ ] 未完成`  `[ ] 已完成`
- 验证状态：`[ ] 未验证`
- 可测试：是
- 前置任务：G3
- 目标：按最新 DDD 中的可选增强路径，实现训练期 Cross-Attn 与 `training_dropout` 渐进关闭消融。
- 实现清单：
- 将 Cross-Attn 置于 decoder 层栈内
- 配置项对齐：`enable_decoder_cross_attn`、`constraint_cross_attn_dropout`
- 推理阶段关闭 Cross-Attn，保持 latent-only 主路径
- 建议验证：
- 开关 Cross-Attn 时前向均稳定
- `training_dropout -> 1` 时输出不依赖外部 `C`
- 验证证据：
- 测试文件/命令：`____`
- 结果摘要：`____`

### P4-03 与原始五类版对比实验与论文导出

- 完成状态：`[ ] 未完成`  `[ ] 已完成`
- 验证状态：`[ ] 未验证`
- 可测试：是
- 前置任务：P4-01
- 目标：形成面向结论输出的主表、对比实验与论文导出结果。
- 实现清单：
- 与原始五类 Fused 版统一口径对比
- 比较“复杂度 / 稳定性 / 指标”
- 输出主表、消融表与定性案例
- 建议验证：
- 至少完成 1 组统一口径对比实验
- 明确回答去掉 `Collinear` 后是否更稳
- 结果可从 `exp_id` 回溯到 manifest、checkpoint 与 eval 指标
- 验证证据：
- 实验记录路径：`____`
- 结果摘要：`____`

### G4 最终验收放行

- 完成状态：`[ ] 未完成`  `[ ] 已完成`
- 验证状态：`[ ] 未验证`
- 前置任务：P4-03
- 最终验收标准：
- 四类约束主链路已完整闭环
- `Collinear` 已从提取、标签、重建、评估与 summary 中彻底删除
- latent-only 推理稳定成立
- 已形成与原始五类版的统一口径对比结论
- 论文导出与实验追踪链路完整
- 结论：`____`

## 6. 建议实验记录模板

| 实验名 | 版本定位 | 约束范围 | `R_h` | `R_v` | `parallel_recall` | `perpendicular_recall` | 结论 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fused_full_v1 | 原始五类版 | H/V/∥/⊥/collinear | ____ | ____ | ____ | ____ | ____ |
| fused_modify2_v1 | 四类约束版 | H/V/∥/⊥ | ____ | ____ | ____ | ____ | ____ |

## 7. 建议验收口径

建议把最终验收压缩为五个问题：

1. `Collinear` 是否已从提取、标签、重建、评估和 summary 中彻底删除。
2. 四类约束链路是否已完整打通。
3. latent-only 推理是否仍然成立。
4. 最新 DDD 中的模块边界是否已按章节完整落地。
5. 相比原始五类版，四类约束版是否更适合作为后续主线。
