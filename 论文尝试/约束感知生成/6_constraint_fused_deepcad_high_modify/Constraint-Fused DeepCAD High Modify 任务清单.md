# Constraint-Fused DeepCAD High Modify 任务清单

> 来源文档：`Constraint-Fused DeepCAD High Modify DDD技术方案.md`
>
> 目标：将 High Modify DDD 方案拆解为可执行、可记录、可验证、可串行推进的任务清单。实现口径与最新 High Modify DDD 保持一致：**保持生成闭包 `P(S | z)`（decoder 主路径只依赖 `z`）；约束监督全部后移至 decoder 层后表示；默认 `dim_z=512` 与命令/约束分离池化；四类约束定义与数据契约不变；不引入推理期必需的外部约束输入。**

## 0. DDD 映射速览


| DDD 限界上下文                               | 本清单主要落点                          |
| --------------------------------------- | -------------------------------- |
| Sketch Preparation Context              | P0-01、P1-01                      |
| Constraint-Fused Encoding Context       | P1-02                             |
| Latent Bottleneck Context               | P1-02、G1                         |
| Latent-Only Generation Context          | P1-03、G1、P2-01                  |
| Decoder-Side Constraint Supervision Context | P2-01、P2-02、P2-03、P2-04       |
| Training Orchestration（训练编排）          | P2-03、P2-04、P3-01、P3-02、P4 系列 |


## 1. 使用规则

1. 本清单默认按顺序执行，不建议跳过 Gate 直接进入后续任务。
2. 每个可测试任务完成后，必须先完成验证并记录结果，再进入下一任务。
3. High Modify 首版**不改变**四类约束语义与 `SketchPreparation` 输出字段契约；评估口径与仓库现有 `summary.json` / `per_sample_counts.csv` 字段应对齐或可映射说明。
4. 所有实验应优先复用仓库 `data/` 下真实 DeepCAD 数据，而不是只依赖合成张量。
5. 实验结果必须最终落到统一口径的 `per_sample_counts.csv` 与 `summary.json`，不能只看训练 loss。
6. **架构红线**：主解码路径不得将 `constraint_memory` 作为 `decoder.forward` 的必需输入；约束相关损失不得直接基于 `z` 计算（见项目 `CF-DeepCADAgreement`）。
7. 若某任务导致 `cmd_acc` / `args_acc` 或 `ratio_h`、`ratio_v`、`n_parse_fail_pred` 明显恶化，应记录原因并优先回退或降低 `alpha` / `beta` / `gamma`。

## 2. 标记约定

- 完成状态：`[ ] 未完成` / `[x] 已完成`
- 验证状态：`[ ] 未验证` / `[ ] 验证中` / `[x] 已通过` / `[ ] 未通过`
- 阻塞状态：`[ ] 无` / `[ ] 有`

## 3. 总控看板


| ID    | 阶段      | 任务                                                | 可测试 | 前置    | 完成    | 验证    |
| ----- | ------- | ------------------------------------------------- | --- | ----- | ----- | ----- |
| P0-01 | Phase 0 | 范围冻结、架构红线确认与独立包目录映射                           | 否   | 无     | `[ ]` | `N/A` |
| P1-01 | Phase 1 | 基线对照与消融口径锁定（A0–A4、指标字段）                        | 是   | P0-01 | `[ ]` | `[ ]` |
| P1-02 | Phase 1 | `SegmentSeparatedPooling` + `Bottleneck512` 编码侧扩容 | 是   | P1-01 | `[ ]` | `[ ]` |
| P1-03 | Phase 1 | `LatentOnlyDecoderAdapter`：移除主路径 `constraint_memory` | 是   | P1-02 | `[ ]` | `[ ]` |
| G1    | Gate 1  | Latent 与 latent-only 解码最小闭环放行                    | 是   | P1-03 | `[ ]` | `[ ]` |
| P2-01 | Phase 2 | `DecoderLineFeatureGather`（hidden → line feature）   | 是   | G1    | `[ ]` | `[ ]` |
| P2-02 | Phase 2 | `DecoderOutputConstraintReconHead`（decoder-side unary/pair） | 是   | P2-01 | `[ ]` | `[ ]` |
| P2-03 | Phase 2 | `train_use_case` 全链路接线与 `LossComposer` 对齐        | 是   | P2-02 | `[ ]` | `[ ]` |
| P2-04 | Phase 2 | 配置默认值、开关、回退策略与 Stage 0–2 训练策略参数                | 是   | P2-03 | `[ ]` | `[ ]` |
| G2    | Gate 2  | High Modify 主训练链路放行                            | 是   | P2-04 | `[ ]` | `[ ]` |
| P3-01 | Phase 3 | 短程训练验证与平台损失复测（主任务优先 + warmup）                  | 是   | G2    | `[ ]` | `[ ]` |
| P3-02 | Phase 3 | 测试集重建与统一口径评估                                   | 是   | P3-01 | `[ ]` | `[ ]` |
| P3-03 | Phase 3 | 与 Low Risk / Modify2 / simplify / 原始 DeepCAD 对比与消融摘要 | 是   | P3-02 | `[ ]` | `[ ]` |
| G3    | Gate 3  | High Modify 实验闭环放行                             | 是   | P3-03 | `[ ]` | `[ ]` |
| P4-01 | Phase 4 | 消融：`dim_z` 256 vs 512（A1 vs A2）                    | 是   | G3    | `[ ]` | `[ ]` |
| P4-02 | Phase 4 | 消融：pooling `masked_mean` vs `segment_separated`（A2 vs A3） | 是   | G3    | `[ ]` | `[ ]` |
| P4-03 | Phase 4 | 消融：soft geometry 开关（A3 vs A4）与论文证据链归档            | 是   | P4-02 | `[ ]` | `[ ]` |
| G4    | Gate 4  | 最终验收放行                                          | 是   | P4-03 | `[ ]` | `[ ]` |


## 4. 全局记录位

- 当前阶段：`____`
- 当前进行中的任务 ID：`____`
- 最近一次通过的 Gate：`____`
- 设计基线：`论文尝试/约束感知生成/6_constraint_fused_deepcad_high_modify/Constraint-Fused DeepCAD High Modify DDD技术方案.md`
- 参考架构（若已生成 HTML）：`论文尝试/约束感知生成/6_constraint_fused_deepcad_high_modify/Constraint-Fused DeepCAD High Modify Model Architecture.html`
- 建议实现目录：`constraint_fused_deepcad_high_modify/`（与技术方案 §5 目录草案一致）
- 真实数据根路径 `DATA_ROOT`：`data`
- 对照基线 checkpoint（Low Risk 或 Modify2，择一冻结）：`____`
- High Modify 主实验 checkpoint：`____`
- 最近一次验证时间：`____`
- 最近一次失败任务：`____`
- 最近一次失败原因：`____`

## 5. 任务明细

### P0-01 范围冻结、架构红线确认与独立包目录映射

- 完成状态：`[ ] 未完成`  `[ ] 已完成`
- 可测试：否
- 前置任务：无
- 目标：明确 High Modify 为**独立方案**（非 Low Risk 增量说明），冻结与 DDD 一致的交付边界与代码落点。
- 实现清单：
- 冻结四项老师输入对应的架构决策：latent-only decoder、`ConstraintReconHead` 输入改为 decoder line features、`dim_z=512` 默认、默认 `segment_separated` pooling
- 明确**非目标**：推理期硬约束投影、改变四类约束定义、约束损失直接压 `z`、辅助头替代主解码路径
- 按技术方案 §5 建立 `constraint_fused_deepcad_high_modify/` 包骨架（`application/`、`encoding/`、`generation/`、`config/`、`sketch_preparation/`、`domain/` 等）
- 产出物：
- 范围与红线说明（含与 Low Risk 差异对照表）
- 目录映射与模块归属说明

### P1-01 基线对照与消融口径锁定（A0–A4、指标字段）

- 完成状态：`[ ] 未完成`  `[ ] 已完成`
- 验证状态：`[ ] 未验证`
- 可测试：是
- 前置任务：P0-01
- 目标：固定对照实验矩阵与评估字段，使后续 High Modify 结果可与技术方案 §4.3 消融表逐项对齐。
- 实现清单：
- 锁定评估字段：`cmd_acc`、`args_acc`、`ratio_h`、`ratio_v`、`parallel_recall_index_aligned`、`perpendicular_recall_index_aligned`、`unary_recon_loss`、`pair_recon_loss`、`geom_loss`（若启用）
- 记录 `n_parse_fail_pred`、`n_samples_extrude_count_mismatch` 等稳定性指标
- 明确 A0（Low Risk 式）、A1（z-only + decoder recon）、A2（512 + masked mean）、A3（完整 High Modify）、A4（无 soft geom）的配置快照字段名
- 建议验证：
- 能列出每个 Ai 对应的 `pooling_strategy`、`dim_z`、`recon_input`、`decoder` 输入组合
- 验证证据：
- 测试文件/命令：`____`
- 结果摘要：`____`

### P1-02 `SegmentSeparatedPooling` + `Bottleneck512` 编码侧扩容

- 完成状态：`[ ] 未完成`  `[ ] 已完成`
- 验证状态：`[ ] 未验证`
- 可测试：是
- 前置任务：P1-01
- 目标：在 `EncoderFused` 输出 `command_memory` / `constraint_memory` 后，用分离池化得到 `z_pre: 1×B×512`，再经 bottleneck 得到单 token `z: 1×B×512`。
- 实现清单：
- 实现 `encoding/pooling.py` 中 `SegmentSeparatedPooling`（或等价模块），默认替代 joint masked mean
- 实现或扩展 `encoding/bottleneck.py` 支持 `pooled_dim → dim_z`（512）
- 保持 encoder 侧仍可消费约束 token；`constraint_memory` **仅**用于 pooling / 诊断，不进入 decoder 主路径
- 建议验证：
- `z.shape == (1, B, 512)` 与文档一致
- 变长 batch、空/少约束 token 时数值稳定、无 NaN
- **真实数据**：`DATA_ROOT` 下 ≥1 batch 前向通过
- 验证证据：
- 测试文件/命令：`____`
- 结果摘要：`____`

### P1-03 `LatentOnlyDecoderAdapter`：移除主路径 `constraint_memory`

- 完成状态：`[ ] 未完成`  `[ ] 已完成`
- 验证状态：`[ ] 未验证`
- 可测试：是
- 前置任务：P1-02
- 目标：主生成路径严格为 `decoder(z)`，删除或禁用 `OptionalConstraintCrossAttn` 等对 `constraint_memory` 的依赖；训练与推理调用一致。
- 实现清单：
- `generation/decoder_adapter.py`：`forward(self, z)` 为对外主接口；`command_logits`、`args_logits`、`hidden_states`、`constraint_pred_logits` 输出契约明确
- 移除文档 §5.2 所列旧签名中对 `constraint_memory` / `constraint_mask` 的传入
- 建议验证：
- 随机 `z` 与 encoder 得到的 `z` 两条路径均可前向
- 无可达代码路径在 logits 计算前强制要求 `constraint_memory`
- 验证证据：
- 测试文件/命令：`____`
- 结果摘要：`____`

### G1 Latent 与 latent-only 解码最小闭环放行

- 完成状态：`[ ] 未完成`  `[ ] 已完成`
- 验证状态：`[ ] 未验证`
- 前置任务：P1-03
- 放行标准：
- `z` 为单 token 512 维（或已文档化偏差）
- decoder 主路径仅依赖 `z`，满足 `P(S | z)`
- 编码—池化—瓶颈—解码前向在真实 batch 上稳定
- 结论：`____`

### P2-01 `DecoderLineFeatureGather`（hidden → line feature）

- 完成状态：`[ ] 未完成`  `[ ] 已完成`
- 验证状态：`[ ] 未验证`
- 可测试：是
- 前置任务：G1
- 目标：从 `hidden_states (B,S,D)` 按 `line_cmd_mask`、`line_index_map` 聚合为 `decoder_line_features (B,L,D)`，供 recon head 使用。
- 实现清单：
- 实现 `gather_decoder_line_features`（可向量化优化，语义与技术方案 §3.6 一致）
- 与同 batch 的 `line_mask`、`max_lines` 对齐，避免越界
- 建议验证：
- 多 line、单 line 多 token 聚合时 shape 与数值正确
- 验证证据：
- 测试文件/命令：`____`
- 结果摘要：`____`

### P2-02 `DecoderOutputConstraintReconHead`（decoder-side unary/pair）

- 完成状态：`[ ] 未完成`  `[ ] 已完成`
- 验证状态：`[ ] 未验证`
- 可测试：是
- 前置任务：P2-01
- 目标：以 `decoder_line_features` 为唯一输入（不再使用 `z` + encoder line），输出 `unary_logits`、`pair_logits`，pair 输出对称化。
- 实现清单：
- 新增 `generation/decoder_output_recon_head.py`（或等价）
- 更新 `domain/services.py` 中约束重建服务：输入改为 decoder line features
- `LossComposer` / BCE 接口与现有 `unary_gt`、`pair_gt`、`line_mask`、`pos_weight` 兼容
- 建议验证：
- `pair_logits.shape[-1] == 2`，对称性检查
- 反向传播有限、无 NaN
- 验证证据：
- 测试文件/命令：`____`
- 结果摘要：`____`

### P2-03 `train_use_case` 全链路接线与 `LossComposer` 对齐

- 完成状态：`[ ] 未完成`  `[ ] 已完成`
- 验证状态：`[ ] 未验证`
- 可测试：是
- 前置任务：P2-02
- 目标：`execute()` 内串联：encoder → pooling → bottleneck → `decoder(z)` → line gather → recon head → pred head →（可选）soft geometry；总损失符合技术方案 §3.10。
- 实现清单：
- `application/train_use_case.py`：全程无 `decoder(..., constraint_memory=...)`
- `line_only_constraint_pred_loss` 默认开启，有效 mask 与 Low Risk 语义一致
- `geom_loss` 仅来自 decoder 输出（如 `args_logits`）经 `SoftGeometryInterpreter` 的路径
- 建议验证：
- 单 batch 前向/反向：`cmd_loss`、`args_loss`、`pred_loss`、`unary/pair_recon_loss`、`geom_loss`（若开）均为有限值
- **真实数据**：`DATA_ROOT` 下 ≥1 batch
- 验证证据：
- 测试文件/命令：`____`
- 结果摘要：`____`

### P2-04 配置默认值、开关、回退策略与 Stage 0–2 训练策略参数

- 完成状态：`[ ] 未完成`  `[ ] 已完成`
- 验证状态：`[ ] 未验证`
- 可测试：是
- 前置任务：P2-03
- 目标：`config_constraint_fused_high_modify.py` 默认 `dim_z=512`、`pooling_strategy=segment_separated`、`recon_input=decoder_hidden`、`line_only_pred_loss=true`；提供辅助权重与 pooling 回退开关便于消融与排障。
- 实现清单：
- 禁止默认开启 decoder cross-attn（或完全移除主路径）
- 记录配置快照到实验目录（与 Low Risk 习惯对齐）
- 文档化 Stage 0–2：先小步验证结构 → 低辅助权重稳主任务 → 再升 `alpha`/`beta`/`gamma`
- 建议验证：
- 多组开关组合下模型可构建；关键路径与默认一致
- 验证证据：
- 测试文件/命令：`____`
- 结果摘要：`____`

### G2 High Modify 主训练链路放行

- 完成状态：`[ ] 未完成`  `[ ] 已完成`
- 验证状态：`[ ] 未验证`
- 前置任务：P2-04
- 放行标准：
- 训练一步内所有损失均从 decoder 后或可解释输出计算，无 `z` 直连 recon
- `decoder(z)` 为唯一解码条件
- 配置与代码一致，具备可回退消融开关
- 结论：`____`

### P3-01 短程训练验证与平台损失复测（主任务优先 + warmup）

- 完成状态：`[ ] 未完成`  `[ ] 已完成`
- 验证状态：`[ ] 未验证`
- 可测试：是
- 前置任务：G2
- 目标：按技术方案 §4.2 完成 Stage 0–1 短跑，观察主损失与 `unary/pair_recon_loss`、`geom_loss` 是否异常主导。
- 实现清单：
- smoke training：`train.py` 小 `max_steps` / 少 epoch
- 对比 P1-01 冻结的基线窗口（若可）记录 `pred_loss`、`pair_recon_loss` 等
- 建议验证：
- `train_metrics.csv` / checkpoint / manifest 正常落盘
- 无持续 NaN；主任务 loss 可下降或至少稳定合理区间
- 验证证据：
- 测试文件/命令：`____`
- 结果摘要：`____`

### P3-02 测试集重建与统一口径评估

- 完成状态：`[ ] 未完成`  `[ ] 已完成`
- 验证状态：`[ ] 未验证`
- 可测试：是
- 前置任务：P3-01
- 目标：基于新 checkpoint 跑 `reconstruct.py` / `evaluate.py`（或包内等价脚本），产出 `per_sample_counts.csv` 与 `summary.json`。
- 实现清单：
- 重建与评估与仓库统一口径兼容或可说明差异
- 记录 `ratio_h`、`ratio_v`、`parallel/perpendicular recall` 等
- 建议验证：
- 输出文件字段完整；test/val split 与数据配置一致
- 验证证据：
- 测试文件/命令：`____`
- 结果摘要：`____`

### P3-03 与 Low Risk / Modify2 / simplify / 原始 DeepCAD 对比与消融摘要

- 完成状态：`[ ] 未完成`  `[ ] 已完成`
- 验证状态：`[ ] 未验证`
- 可测试：是
- 前置任务：P3-02
- 目标：用统一口径回答：闭包收紧与监督后移后，约束指标与主任务是否达到预期权衡；对照 §4.3 A0–A4。
- 实现清单：
- 至少一份对比表（含主任务与约束指标、解析失败计数）
- 对照技术方案 §4.5 预期现象做文字解读
- 建议验证：
- 每个结论可追溯到 checkpoint 与评估文件路径
- 验证证据：
- 实验记录路径：`____`
- 结果摘要：`____`

### G3 High Modify 实验闭环放行

- 完成状态：`[ ] 未完成`  `[ ] 已完成`
- 验证状态：`[ ] 未验证`
- 前置任务：P3-03
- 放行标准：
- 训练、重建、评估主链路已跑通
- 与选定基线（Low Risk 或 Modify2）可比
- 对「闭包 + 监督后移 + 扩容 + 分离池化」有初步数据结论
- 结论：`____`

### P4-01 消融：`dim_z` 256 vs 512（A1 vs A2）

- 完成状态：`[ ] 未完成`  `[ ] 已完成`
- 验证状态：`[ ] 未验证`
- 可测试：是
- 前置任务：G3
- 目标：验证 latent 扩容对主任务与约束指标的独立贡献（其余配置尽量与 A1 对齐）。
- 实现清单：
- 固定 `pooling_strategy`、`recon_input`、`decoder` 输入，仅改 `dim_z`
- 建议验证：
- ≥2 组对照；记录训练稳定性
- 验证证据：
- 实验记录路径：`____`
- 结果摘要：`____`

### P4-02 消融：pooling `masked_mean` vs `segment_separated`（A2 vs A3）

- 完成状态：`[ ] 未完成`  `[ ] 已完成`
- 验证状态：`[ ] 未验证`
- 可测试：是
- 前置任务：G3
- 目标：验证分离池化对 parallel/perpendicular 类指标的敏感性（技术方案 §4.5 现象 3）。
- 实现清单：
- 固定 `dim_z=512`、decoder-side recon、latent-only decoder
- 建议验证：
- 至少 1 组对照；关注 `pair_recon_loss` 与 pair recall 是否同向变化
- 验证证据：
- 实验记录路径：`____`
- 结果摘要：`____`

### P4-03 消融：soft geometry 开关（A3 vs A4）与论文证据链归档

- 完成状态：`[ ] 未完成`  `[ ] 已完成`
- 验证状态：`[ ] 未验证`
- 可测试：是
- 前置任务：P4-02
- 目标：归档最佳 checkpoint、曲线、`summary.json` 与消融表，形成可引用论文的证据链。
- 实现清单：
- A4：`gamma=0` 或关闭 soft geometry 路径
- 汇总 A0–A4 结论与风险对策（§6）是否被数据验证
- 建议验证：
- 所有数字可回溯到具体实验目录与 commit
- 验证证据：
- 实验记录路径：`____`
- 结果摘要：`____`

### G4 最终验收放行

- 完成状态：`[ ] 未完成`  `[ ] 已完成`
- 验证状态：`[ ] 未验证`
- 前置任务：P4-03
- 最终验收标准：
- decoder 主路径满足 `P(S | z)`，推理随机采样无需约束 token
- unary/pair 重建与 pred、geom（若用）均基于 decoder 后输出，无 `z` 直连 recon 损失
- 默认 `dim_z=512` 与 `segment_separated` pooling 已落地并通过消融支撑结论
- 统一评估口径下主任务与约束指标结论明确，且可知悉相对基线的代价与收益
- 结论：`____`

## 6. 建议实验记录模板


| 实验名   | 版本定位（Ai） | `dim_z` | `pooling_strategy`   | `recon_input`   | decoder 输入      | soft geom | `cmd_acc` | `args_acc` | `ratio_h` | `ratio_v` | `parallel_recall_index_aligned` | `perpendicular_recall_index_aligned` | `pair_recon_loss` | 结论   |
| ----- | --------- | ------- | -------------------- | --------------- | --------------- | --------- | --------- | ---------- | --------- | --------- | --------------------------------- | ------------------------------------ | ----------------- | ---- |
| A0    | 对照基线      | 256     | masked_mean          | z+encoder line  | z + C_mem（若存在） | 按基线       | ____      | ____       | ____      | ____      | ____                              | ____                                 | ____              | ____ |
| A1    | 监督后移      | 256     | masked_mean          | decoder_hidden | z only          | 按基线       | ____      | ____       | ____      | ____      | ____                              | ____                                 | ____              | ____ |
| A2    | +latent 扩容 | 512     | masked_mean          | decoder_hidden | z only          | 按基线       | ____      | ____       | ____      | ____      | ____                              | ____                                 | ____              | ____ |
| A3    | High Modify 完整 | 512     | segment_separated    | decoder_hidden | z only          | on        | ____      | ____       | ____      | ____      | ____                              | ____                                 | ____              | ____ |
| A4    | 无 soft geom | 512     | segment_separated    | decoder_hidden | z only          | off       | ____      | ____       | ____      | ____      | ____                              | ____                                 | ____              | ____ |


## 7. 建议验收口径

建议把最终验收压缩为八个问题：

1. 训练与推理是否均以 `decoder(z)` 为主接口，且不存在对 `constraint_memory` 的必需依赖。
2. unary / pair 约束重建的 logits 是否**仅**由 decoder line features（及 head 内部结构）产生，而非 `z` 或 encoder line 直连。
3. `ConstraintPredHead` 的 BCE 是否默认只在真实 line 命令位置计算（`line_only_pred_loss`）。
4. `dim_z=512` 与 `Bottleneck512` 是否已作为默认路径验证稳定。
5. 默认 pooling 是否为命令/约束分离的 `segment_separated`，且消融证明相对 `masked_mean` 的影响。
6. soft geometry（若启用）是否仅来自 decoder 输出的可微解释路径，权重合理且不压垮主任务。
7. 统一口径下 `parallel/perpendicular recall` 与 `ratio_h` / `ratio_v`、解析失败计数是否满足产品/论文可接受的权衡。
8. 技术方案 §4.3 中 A0–A4 的结论是否均有数据支撑并可回溯归档。
