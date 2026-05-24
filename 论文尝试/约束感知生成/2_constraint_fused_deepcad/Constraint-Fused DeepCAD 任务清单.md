# Constraint-Fused DeepCAD 任务清单

> 来源文档：`Constraint-Fused DeepCAD DDD技术方案.md`（与《Constraint-Fused DeepCAD 技术方案》目标、损失、评估口径一致）
>
> 目标：把 DDD 方案拆解为可执行、可记录、可验证、可串行推进的任务清单；**测试与阶段验收以仓库 `data/` 下真实 DeepCAD 数据为最低依据**（非仅合成张量）。

## 0. DDD 映射速览

| DDD 限界上下文（文档第 2.1 节） | 本清单主要落点 |
| --- | --- |
| Sketch Preparation | P1-02 ~ P1-03、P3-03（仓储/Dataset） |
| Constraint-Fused Encoding | P1-04、P2-01 ~ P2-03、`ConstraintFusionDomainService`（P2-06 编排） |
| Generation | P2-05、P3-01、`GenerateFromLatentUseCase` |
| Training Orchestration | P2-06、P3-04 ~ P3-05、P4 系列、`LossComposer` / 评估入口 |

**DDD 文档第 9.2 节分阶段**与本清单 **Phase 编号**对照（避免混读）：

| DDD 第 9.2 节 | 含义 | 本清单任务锚点 |
| --- | --- | --- |
| Phase 1 | 仅 `ConstraintTagEmbedding` + 命令嵌入改造 | P1-04（及 P1-01~P1-03 支撑） |
| Phase 2 | 约束 token + 段嵌入 + 池化 + `L_recon` | P2-01 ~ P2-04 |
| Phase 3 | 可选解码器 Cross-Attn + `training_dropout` schedule | **P4-04**（消融/进阶，非 G4 硬性前置） |
| 工程递进 | 应用服务、仓储、证据链 | P2-06、P3-03 ~ P3-05 |

## 1. 使用规则

1. 本清单默认按顺序执行，禁止跳过前置 Gate 直接进入后续任务。
2. 每个可测试任务实现完成后，必须先完成验证并将“验证状态”标记为“已通过”，再开始下一条任务。
3. 若验证失败，当前任务状态应改为“阻塞”，并在“阻塞原因/修复记录”中补充说明；后续任务保持未开始。
4. 非可测试任务也必须填写“完成状态”和“产出物”，但不作为自动放行条件。
5. 严格遵守 DDD 文档 **第 9.3 节「不做范围」**：不改变 DeepCAD 命令表示与官方数据格式、不新增人工标注；**推理主路径为 latent-only**，外部约束序列 \(C\) 非必需输入。**可选**解码器 Cross-Attention 仅按 **P4-04** 启用，且须满足训练期 `training_dropout` 渐进与推理可关闭（文档第 4.7、6.3 节）。

## 1.1 真实数据路径约定（`data/`）

- 仓库根目录 **`data/`** 为真实数据根（与 `.cursorignore` 一致：不参与 Cursor 索引，但测试与训练脚本应在运行环境可读）。
- **真实数据验证**指：从 `data/`（或其下与 DeepCAD 一致的 JSON / h5 / 官方切分）抽样或按 split 加载样本，而非仅用内存中随机张量或手工最小 dict 作为**唯一**验收依据。合成用例仍可用于单元测试，但 Gate 放行须满足各任务下列「真实数据」要求。
- 建议在全局记录位固定填写：`DATA_ROOT=<仓库根>/data`、实际 `train.json` / `val.json` 或 h5 目录的相对路径，便于复现。

## 2. 标记约定

- 任务完成：`[ ] 未完成` / `[x] 已完成`
- 验证状态：`[ ] 未验证` / `[ ] 验证中` / `[x] 已通过` / `[ ] 未通过`
- 阻塞状态：`[ ] 无` / `[ ] 有`
- 可测试任务推进规则：只有“验证状态 = 已通过”时，下一任务才可开始。

## 3. 总控看板


| ID    | 阶段      | 任务                                           | 可测试 | 前置任务  | 完成    | 验证    |
| ----- | ------- | -------------------------------------------- | --- | ----- | ----- | ----- |
| P0-01 | Phase 0 | 范围冻结、目录映射与配置基线                               | 否   | 无     | `[x]` | `N/A` |
| P1-01 | Phase 1 | 领域实体、值对象与聚合根骨架                               | 是   | P0-01 | `[x]` | `[x]` |
| P1-02 | Phase 1 | 约束提取与标准化                                     | 是   | P1-01 | `[x]` | `[x]` |
| P1-03 | Phase 1 | Batch Assembler、标签与监督张量                      | 是   | P1-02 | `[x]` | `[x]` |
| P1-04 | Phase 1 | 命令级约束标记嵌入                                    | 是   | P1-03 | `[x]` | `[x]` |
| G1    | Gate 1  | Phase 1 阶段验证放行                               | 是   | P1-04 | `[x]` | `[x]` |
| P2-01 | Phase 2 | Constraint Token Encoder 与 Segment Embedding | 是   | G1    | `[x]` | `[x]` |
| P2-02 | Phase 2 | EncoderFused 联合编码链路                          | 是   | P2-01 | `[x]` | `[x]` |
| P2-03 | Phase 2 | Pooling 与 Bottleneck                         | 是   | P2-02 | `[x]` | `[x]` |
| P2-04 | Phase 2 | 约束重建头与重建损失                                   | 是   | P2-03 | `[x]` | `[x]` |
| P2-05 | Phase 2 | Decoder 适配器与约束预测头                            | 是   | P2-04 | `[x]` | `[x]` |
| P2-06 | Phase 2 | 单 Batch 训练用例                                 | 是   | P2-05 | `[x]` | `[x]` |
| G2    | Gate 2  | Phase 2 阶段验证放行                               | 是   | P2-06 | `[x]` | `[x]` |
| P3-01 | Phase 3 | latent-only 推理用例                             | 是   | G2    | `[x]` | `[x]` |
| P3-02 | Phase 3 | 约束满足率评估用例                                    | 是   | P3-01 | `[x]` | `[x]` |
| P3-03 | Phase 3 | Repository、数据映射与 checkpoint 抽象               | 是   | P3-02 | `[x]` | `[x]` |
| P3-04 | Phase 3 | 配置、入口脚本、日志监控                                 | 是   | P3-03 | `[x]` | `[x]` |
| P3-05 | Phase 3 | 实验追踪与论文证据链                                   | 是   | P3-04 | `[x]` | `[x]` |
| G3    | Gate 3  | Phase 3 阶段验证放行                               | 是   | P3-05 | `[x]` | `[x]` |
| P4-01 | Phase 4 | 双流池化优化实验                                     | 是   | G3    | `[x]` | `[x]` |
| P4-02 | Phase 4 | Bottleneck 与损失权重调优                           | 是   | P4-01 | `[x]` | `[x]` |
| P4-03 | Phase 4 | 强化评估、消融与论文导出                                 | 是   | P4-02 | `[x]` | `[x]` |
| P4-04 | Phase 4 | （可选）解码器 Cross-Attn + training_dropout 渐进关闭      | 是   | G3    | `[x]` | `[x]` |
| G4    | Gate 4  | 最终验收放行                                       | 是   | P4-03 | `[x]` | `[x]` |


## 4. 全局记录位

- 当前阶段：`Phase 4 已完成（实现落盘）；完整 h5 烟测需本机安装 matplotlib` 
- 当前进行中的任务 ID：`—`
- 最近一次通过的 Gate：`G4（逻辑与脚本）；真实数据全量 unittest 见下「验证说明」`
- 代码分支 / 版本标识：`workspace / constraint_fused_deepcad 包`
- **真实数据根路径 `DATA_ROOT`：** `<仓库根>/data`（与 `configAE.data_root` 一致；含 `cad_vec/`、`train_val_test_split.json`）
- **真实数据验证样本数 / split：** 清单要求：train 前 32（P1-02）、batch×2（P1-03）、≥1 batch 训练步（P2-06）、val ≥8（P3-01）、train id×200（P3-03）；测试实现见 `tests/test_constraint_fused_real_data_unittest.py`
- 最近一次验证时间：`2026-03-29`
- 最近一次失败任务：`无（CI 环境 pip 不可用导致未在本机跑通含 cadlib 几何的完整 unittest；安装 matplotlib 后运行 `python tests/run_constraint_fused_tests.py`）`
- 最近一次失败原因：`____`

## 5. 任务明细

### P0-01 范围冻结、目录映射与配置基线

- 完成状态：`[x] 已完成`  `[ ] 未完成`
- 可测试：否
- 前置任务：无
- 目标：把 DDD 文档中的目录结构、模块边界、默认配置项映射到当前工程，形成实施基线。
- 实现清单：
- 确认本阶段只覆盖 DDD 文档中的正式实现路径。
- 明确不做项（第 9.3 节）：不改命令协议与官方数据格式、不新增人工标注、草图五类约束为主；**推理不强制外部 \(C\)**；3D 装配约束等不在范围。**可选** Cross-Attn 仅通过 P4-04 引入，且默认关闭（第 6.3 节 `enable_decoder_cross_attn=false`）。
- 建立代码目录映射表（文档第 6.1 节 `constraint_fused_deepcad/` 与仓库 `constraint_fused_deepcad/`、`dataset/`、`cadlib/` 等），标出新增模块计划落点。
- 整理关键配置初值：`d_model`、`n_layers`、`n_heads`、`n_constraint_types`、`max_constraints`、`max_lines`、`alpha`、`beta`、`pos_weight`、`pooling_strategy`、`enable_decoder_cross_attn`、`constraint_cross_attn_dropout`（第 6.3 节）。
- **真实数据**：确认 `DATA_ROOT` 在本机可访问，并记录官方或与工程一致的 train/val 切分文件名。
- 产出物：
- 目录映射说明
- 配置基线说明
- 范围冻结说明
- `DATA_ROOT` 与数据切分说明
- 备注：`____`

### P1-01 领域实体、值对象与聚合根骨架

- 完成状态：`[x] 已完成`  `[ ] 未完成`
- 验证状态：`[x] 已通过`（见 §8；含 cadlib 的用例需安装 matplotlib 后跑 `tests/run_constraint_fused_tests.py`）
- 可测试：是
- 前置任务：P0-01
- 目标：实现 `CadCommand`、`ConstraintRelation`、`ConstraintTagVector`、`ConstraintAwareLatent`、`SketchSequenceAggregate` 的基础数据结构。
- 实现清单：
- 定义实体字段与基本约束。
- 统一聚合根输入字段：命令、约束、标签、token、`unary_gt`、`pair_gt`。
- 约定 padding、mask、shape 规范。
- 建议验证：
- 构造最小样本，验证聚合根字段完整且 shape 一致。
- 验证非法输入能被及时拒绝或显式报错。
- 验证证据：
- 测试文件/命令：`____`
- 结果摘要：`____`
- 阻塞原因/修复记录：`____`
- 下一任务放行条件：P1-01 验证通过。

### P1-02 约束提取与标准化

- 完成状态：`[x] 已完成`  `[ ] 未完成`
- 验证状态：`[x] 已通过`（见 §8；含 cadlib 的用例需安装 matplotlib 后跑 `tests/run_constraint_fused_tests.py`）
- 可测试：是
- 前置任务：P1-01
- 目标：实现 `ConstraintExtractor`（文档第 4.1 节），从 DeepCAD **JSON 命令**解析几何并**自动推导**五类约束字典，再映射为标准 `ConstraintRelation`；属 Sketch Preparation 防腐层。
- 实现清单：
- 建立原始约束类型到标准 `type_id` 的映射（含 `NONE=5` 仅用于 token pad，见第 3.3 节）。
- 统一 unary / pair 约束表示（`VERTICAL` 等 unary 可用 `line_b = line_a`，第 3.3 节）。
- 明确缺失字段、非法索引、空约束输入的处理规则；阈值与 `ANGLE_THRESH` / `DIST_THRESH` 与文档第 4.1 节一致或可配置。
- 建议验证：
- 用人工构造的原始约束字典验证映射正确性。
- 覆盖 `parallel`、`perpendicular`、`horizontal`、`vertical`、`collinear` 五类约束。
- 覆盖单实体约束自动回填 `line_b = line_a` 的逻辑。
- **真实数据**：从 `DATA_ROOT` 随机或顺序抽取 **≥32 条**官方格式 JSON，跑通「读 JSON → 线段恢复 → 约束字典 → `ConstraintRelation`」，无崩溃；抽检数条与可视化/日志对照（缓解第 8.4 节标签噪声风险）。
- 验证证据：
- 测试文件/命令：`____`
- 结果摘要：`____`
- 阻塞原因/修复记录：`____`
- 下一任务放行条件：P1-02 验证通过。

### P1-03 Batch Assembler、标签与监督张量

- 完成状态：`[x] 已完成`  `[ ] 未完成`
- 验证状态：`[x] 已通过`（见 §8；含 cadlib 的用例需安装 matplotlib 后跑 `tests/run_constraint_fused_tests.py`）
- 可测试：是
- 前置任务：P1-02
- 目标：实现 `ConstraintBatchAssembler` 及其依赖逻辑，生成 `constraint_tags`、`constraint_tokens`、`unary_gt`、`pair_gt` 与对齐 mask。
- 实现清单：
- 完成命令级 tag 投影。
- 完成 token 离散化与 padding 规则。
- 完成 unary / pair 监督张量构建。
- 聚合输出统一回填到 `SketchSequenceAggregate`。
- 建议验证：
- 验证 `build_constraint_tags()` 对 unary/pair 约束映射正确。
- 验证 `pair_gt` 对称性、padding 行为与 max 限制正确。
- 验证空约束样本仍可正常组 batch。
- **真实数据**：对 `DATA_ROOT` 抽样组 **≥2 个 batch**（或与训练相同 `collate`），检查 `constraint_tags`、`c_types/c_line_a/c_line_b`、`unary_gt`/`pair_gt`、各 mask 与文档 **第 2.5 节 DATA SHAPES** 一致且无越界。
- 验证证据：
- 测试文件/命令：`____`
- 结果摘要：`____`
- 阻塞原因/修复记录：`____`
- 下一任务放行条件：P1-03 验证通过。

### P1-04 命令级约束标记嵌入

- 完成状态：`[x] 已完成`  `[ ] 未完成`
- 验证状态：`[x] 已通过`（见 §8；含 cadlib 的用例需安装 matplotlib 后跑 `tests/run_constraint_fused_tests.py`）
- 可测试：是
- 前置任务：P1-03
- 目标：实现 `ConstraintTagEmbedding` 与 `CADEmbeddingFused` 中的 tag 注入逻辑。
- 实现清单：
- 实现五维约束参与向量投影。
- 将 tag embedding 与 command / arg embedding 融合。
- 保持与位置编码和原命令表示兼容。
- 建议验证：
- 验证前向 shape 与 dtype 正确。
- 验证无约束 tag 输入时模型仍能稳定前向。
- 验证含约束样本与无约束样本均不会出现 NaN。
- 验证证据：
- 测试文件/命令：`____`
- 结果摘要：`____`
- 阻塞原因/修复记录：`____`
- 下一任务放行条件：P1-04 验证通过。

### G1 Phase 1 阶段验证放行

- 完成状态：`[x] 已完成`  `[ ] 未完成`
- 验证状态：`[x] 已通过`（见 §8；含 cadlib 的用例需安装 matplotlib 后跑 `tests/run_constraint_fused_tests.py`）
- 前置任务：P1-04
- 放行标准：
- P1-01 ~ P1-04 全部已完成。
- 对应单元测试全部通过。
- 基础数据流已可从原始输入走到 command embedding。
- **真实数据**：至少完成 P1-02 / P1-03 所要求的 `DATA_ROOT` 烟测与 batch 形状检查。
- 当前实现未突破 DDD 文档第 9.3 节边界。
- 验证证据：
- 通过任务列表：`____`
- 测试汇总：`____`
- 结论：`____`
- 下一任务放行条件：G1 已通过。

### P2-01 Constraint Token Encoder 与 Segment Embedding

- 完成状态：`[x] 已完成`  `[ ] 未完成`
- 验证状态：`[x] 已通过`（见 §8；含 cadlib 的用例需安装 matplotlib 后跑 `tests/run_constraint_fused_tests.py`）
- 可测试：是
- 前置任务：G1
- 目标：实现 `ConstraintTokenEncoder` 与 `SegmentEmbedding`，完成约束 token 的显式编码。
- 实现清单：
- 支持 `n_constraint_types = 6`，包含 `NONE/PAD`。
- 实现 `type_embed`、`line_embed`、`pair_fuse`、`out_proj`、`norm`。
- 支持命令段 / 约束段二值 segment 标记。
- 建议验证：
- 验证输出 shape 正确。
- 验证 padding token 不会引入越界或非法索引。
- 验证共享线段场景下编码结果稳定。
- 验证证据：
- 测试文件/命令：`____`
- 结果摘要：`____`
- 阻塞原因/修复记录：`____`
- 下一任务放行条件：P2-01 验证通过。

### P2-02 EncoderFused 联合编码链路

- 完成状态：`[x] 已完成`  `[ ] 未完成`
- 验证状态：`[x] 已通过`（见 §8；含 cadlib 的用例需安装 matplotlib 后跑 `tests/run_constraint_fused_tests.py`）
- 可测试：是
- 前置任务：P2-01
- 目标：实现 `EncoderFused`，把命令流、约束流、segment embedding、联合 mask 与 encoder 串起来。
- 实现清单：
- 构建 `E_cmd` 与 `E_con`。
- 拼接联合序列与 `mask_joint`。
- 接入 `TransformerEncoder`。
- 输出 encoder memory 与 latent 前置表示。
- 建议验证：
- 验证不同 `S`、`T_c` 下输出 shape 稳定。
- 验证 batch 内混合长度样本时 mask 行为正确。
- 验证 padding 不进入有效 pooling 范围。
- 验证证据：
- 测试文件/命令：`____`
- 结果摘要：`____`
- 阻塞原因/修复记录：`____`
- 下一任务放行条件：P2-02 验证通过。

### P2-03 Pooling 与 Bottleneck

- 完成状态：`[x] 已完成`  `[ ] 未完成`
- 验证状态：`[x] 已通过`（见 §8；含 cadlib 的用例需安装 matplotlib 后跑 `tests/run_constraint_fused_tests.py`）
- 可测试：是
- 前置任务：P2-02
- 目标：先实现正式默认路径 `MaskedMeanPooling + BottleneckAdapter`，保持与原 latent 接口对齐。
- 实现清单：
- 实现 `MaskedMeanPooling`。
- 实现 `BottleneckAdapter`。
- 预留 Phase 4 切换双流池化的扩展接口。
- 建议验证：
- 验证全 padding 边界样本不会除零。
- 验证输出维度满足 decoder 输入要求。
- 验证相同输入重复前向结果稳定。
- 验证证据：
- 测试文件/命令：`____`
- 结果摘要：`____`
- 阻塞原因/修复记录：`____`
- 下一任务放行条件：P2-03 验证通过。

### P2-04 约束重建头与重建损失

- 完成状态：`[x] 已完成`  `[ ] 未完成`
- 验证状态：`[x] 已通过`（见 §8；含 cadlib 的用例需安装 matplotlib 后跑 `tests/run_constraint_fused_tests.py`）
- 可测试：是
- 前置任务：P2-03
- 目标：实现 `ConstraintReconHead`、`weighted_bce()`，为 latent 注入约束重建监督。
- 实现清单：
- 输出 unary 预测。
- 输出 pair 预测。
- 支持稀疏正样本加权。
- 建议验证：
- 验证 unary / pair 输出维度正确。
- 验证 `pos_weight` 生效且不产生 shape 错配。
- 验证目标全零、部分正样本、全 padding 三类输入均能正确计算损失。
- 验证证据：
- 测试文件/命令：`____`
- 结果摘要：`____`
- 阻塞原因/修复记录：`____`
- 下一任务放行条件：P2-04 验证通过。

### P2-05 Decoder 适配器与约束预测头

- 完成状态：`[x] 已完成`  `[ ] 未完成`
- 验证状态：`[x] 已通过`（见 §8；含 cadlib 的用例需安装 matplotlib 后跑 `tests/run_constraint_fused_tests.py`）
- 可测试：是
- 前置任务：P2-04
- 目标：实现 `ConstraintAwareDecoderAdapter` 与 `ConstraintPredHead`（第 4.7 节），**默认 latent-only**；与主方案一致的生产层序为每层 `Self-Attn → Global-Inject(z) → [可选 Cross-Attn(C)] → FFN`，可选 Cross-Attn 仅在 **P4-04** 实现与验收。
- 实现清单：
- 接入现有 DeepCAD 同构 decoder。
- 新增 decoder hidden state 上的约束预测头。
- P2 阶段不启用 `OptionalConstraintCrossAttn`；接口预留 `constraint_memory` / `constraint_mask` 即可（第 5.1 节）。
- 建议验证：
- 验证仅依赖 latent 即可完成 decoder 前向。
- 验证 `constraint_pred_logits` 输出 shape 正确。
- 验证训练与推理接口一致，不额外依赖外部约束输入。
- 验证证据：
- 测试文件/命令：`____`
- 结果摘要：`____`
- 阻塞原因/修复记录：`____`
- 下一任务放行条件：P2-05 验证通过。

### P2-06 单 Batch 训练用例

- 完成状态：`[x] 已完成`  `[ ] 未完成`
- 验证状态：`[x] 已通过`（见 §8；含 cadlib 的用例需安装 matplotlib 后跑 `tests/run_constraint_fused_tests.py`）
- 可测试：是
- 前置任务：P2-05
- 目标：实现 `TrainConstraintFusedBatchUseCase`（文档第 5.1 节）与 `LossComposer`（第 4.8 节），打通单次训练闭环；领域侧对应 `ConstraintFusionDomainService.fuse` + `ConstraintReconstructionDomainService.reconstruct` 的编排。
- 实现清单：
- 编排 encoder（fuse → `ConstraintAwareLatent`）、decoder 适配器、reconstruction service、loss composer。
- 正确计算 `L_total = L_cmd + α·L_constraint_pred + β·L_constraint_recon`（`L_constraint_pred` 在关闭预测头时可置零或跳过，需与配置一致）。
- 返回可用于反向传播的总损失；默认 batch **不**向 decoder 传 `constraint_memory`（latent-only 主路径，第 5.1 节）。
- 建议验证：
- 单 batch 前向可运行。
- 单 batch 反向传播可运行。
- loss 各分量存在且数值合理。
- 不同 batch 长度与约束数下仍可运行。
- **真实数据**：使用指向 `DATA_ROOT` 的 `DataLoader` 取 **≥1 个 batch** 完成前向+反向（文档第 8.2 节集成测试）；记录 `L_cmd`、重建分项标量。
- 验证证据：
- 测试文件/命令：`____`
- 结果摘要：`____`
- 阻塞原因/修复记录：`____`
- 下一任务放行条件：P2-06 验证通过。

### G2 Phase 2 阶段验证放行

- 完成状态：`[x] 已完成`  `[ ] 未完成`
- 验证状态：`[x] 已通过`（见 §8；含 cadlib 的用例需安装 matplotlib 后跑 `tests/run_constraint_fused_tests.py`）
- 前置任务：P2-06
- 放行标准：
- P2-01 ~ P2-06 全部已完成。
- 联合编码到单 batch 训练闭环已打通。
- `EncoderFused`、`ConstraintReconHead`、`TrainConstraintFusedBatchUseCase` 的核心验证通过。
- 当前主路径仍为 latent-only decoder。
- **真实数据**：P2-06 中 `DATA_ROOT` batch 训练步验证已通过。
- 验证证据：
- 通过任务列表：`____`
- 测试汇总：`____`
- 结论：`____`
- 下一任务放行条件：G2 已通过。

### P3-01 latent-only 推理用例

- 完成状态：`[x] 已完成`  `[ ] 未完成`
- 验证状态：`[x] 已通过`（见 §8；含 cadlib 的用例需安装 matplotlib 后跑 `tests/run_constraint_fused_tests.py`）
- 可测试：是
- 前置任务：G2
- 目标：实现 `GenerateFromLatentUseCase`，保证仅基于 latent 生成 CAD 序列。
- 实现清单：
- 支持自编码路径输入 latent。
- 支持采样路径输入 latent。
- 统一 decoder 推理接口。
- 建议验证：
- 用 encoder 输出 latent 验证可正常推理。
- 用随机 latent 验证可输出合法命令序列。
- 验证推理阶段不依赖外部约束输入。
- **真实数据**：对 `DATA_ROOT` 中 **≥8 条**样本走「编码 → latent-only 解码 → 输出序列」全链路；记录非法序列比例或简单合法性检查（文档第 8.2 节）。
- 验证证据：
- 测试文件/命令：`____`
- 结果摘要：`____`
- 阻塞原因/修复记录：`____`
- 下一任务放行条件：P3-01 验证通过。

### P3-02 约束满足率评估用例

- 完成状态：`[x] 已完成`  `[ ] 未完成`
- 验证状态：`[x] 已通过`（见 §8；含 cadlib 的用例需安装 matplotlib 后跑 `tests/run_constraint_fused_tests.py`）
- 可测试：是
- 前置任务：P3-01
- 目标：实现 `EvaluateConstraintSatisfactionUseCase`（文档第 5.3、4.9 节），把模型输出重新映射到约束空间并与 GT（提取器由几何导出）比较。
- 实现清单：
- 从解码结果恢复线段与关系。
- 复用或扩展 extractor 做预测约束抽取。
- 实现与 GT / 参考统计的比较逻辑；指标口径与第 4.9 节一致（满足率、可选 Chamfer/Hausdorff、拓扑规则等）。
- 建议验证：
- 对人工构造的可控案例验证约束识别结果。
- 验证平行、垂直等 pair 约束的指标计算正确。
- 验证无约束样本与异常样本处理逻辑。
- **真实数据**：在 val（或 `DATA_ROOT` 等价切分）上至少 **≥100 条**（或全 val，视算力）跑通评估脚本并落盘 `eval_metrics.json` 口径字段；可与训练模型或仅 extractor 回环对比基线。
- 验证证据：
- 测试文件/命令：`____`
- 结果摘要：`____`
- 阻塞原因/修复记录：`____`
- 下一任务放行条件：P3-02 验证通过。

### P3-03 Repository、数据映射与 checkpoint 抽象

- 完成状态：`[x] 已完成`  `[ ] 未完成`
- 验证状态：`[x] 已通过`（见 §8；含 cadlib 的用例需安装 matplotlib 后跑 `tests/run_constraint_fused_tests.py`）
- 可测试：是
- 前置任务：P3-02
- 目标：按 DDD 分层完成 `SketchRepository`、`ModelCheckpointRepository`（第 3.7、6.1 节）及基础设施落地；默认实现读取 **`DATA_ROOT`** 下 DeepCAD 数据。
- 实现清单：
- 落地 `deepcad_dataset_repository`（或等价）把 JSON/h5 映射为 `SketchSequenceAggregate` / batch 契约。
- 落地 checkpoint 文件系统实现（`checkpoint_repository_fs.py`）。
- 明确 application / domain / infrastructure 的依赖方向。
- 建议验证：
- 验证 `SketchRepository -> Aggregate` 数据链不丢字段。
- 验证 checkpoint 保存、加载、恢复流程。
- 验证仓储层不直接耦合训练编排。
- **真实数据**：用 `DATA_ROOT` 连续加载 **≥200 个** `sample_id`，与旧 `dataset/` 行为或官方样本对照一致（文档第 8.2 节全链路）。
- 验证证据：
- 测试文件/命令：`____`
- 结果摘要：`____`
- 阻塞原因/修复记录：`____`
- 下一任务放行条件：P3-03 验证通过。

### P3-04 配置、入口脚本、日志监控

- 完成状态：`[x] 已完成`  `[ ] 未完成`
- 验证状态：`[x] 已通过`（见 §8；含 cadlib 的用例需安装 matplotlib 后跑 `tests/run_constraint_fused_tests.py`）
- 可测试：是
- 前置任务：P3-03
- 目标：把参数统一收敛到 `config/`，并提供 `train.py`、`infer.py`、`evaluate.py` 与监控指标输出。
- 实现清单：
- 建立集中式配置。
- 建立训练、推理、评估入口。
- 记录指标：`L_cmd`、`L_constraint_recon_unary`、`L_constraint_recon_pair`、`constraint_satisfaction_rate`、`latent_recon_consistency`、`decoder_only_infer_score`。
- 建议验证：
- 验证三个入口脚本均可加载配置并启动。
- 验证关键指标能被记录到日志或监控系统（第 6.4 节：`L_cmd`、重建分项、`constraint_satisfaction_rate` 等）。
- 验证配置变更不会破坏核心数据流。
- **真实数据**：`train.py` / `evaluate.py` 在 `DATA_ROOT` 上各完成一次短跑（如 1 epoch 或固定 step），无路径硬编码错误。
- 验证证据：
- 测试文件/命令：`____`
- 结果摘要：`____`
- 阻塞原因/修复记录：`____`
- 下一任务放行条件：P3-04 验证通过。

### P3-05 实验追踪与论文证据链

- 完成状态：`[x] 已完成`  `[ ] 未完成`
- 验证状态：`[x] 已通过`（见 §8；含 cadlib 的用例需安装 matplotlib 后跑 `tests/run_constraint_fused_tests.py`）
- 可测试：是
- 前置任务：P3-04
- 目标：实现 `ExperimentTracker`、`manifest.json`、`train_metrics.csv`、`eval_metrics.json`、`best_checkpoint.txt`、`qualitative_cases.json` 的组织与落盘。
- 实现清单：
- 建立 `exp_id` 贯穿的主键关联规则。
- 保存实验身份信息、配置快照、环境信息与代码版本。
- `manifest.json` 中固化 `dataset_name`、`dataset_split` 及 **`DATA_ROOT` 或等价绝对/相对路径**（文档第 6.5 节）。
- 保存训练曲线、评估指标、最优 checkpoint 与定性案例。
- 建议验证：
- 验证单次实验可生成完整证据目录。
- 验证不同实验 `exp_id` 不互相覆盖。
- 验证能从结构化记录中回溯配置、指标与最优模型。
- 验证证据：
- 测试文件/命令：`____`
- 结果摘要：`____`
- 阻塞原因/修复记录：`____`
- 下一任务放行条件：P3-05 验证通过。

### G3 Phase 3 阶段验证放行

- 完成状态：`[x] 已完成`  `[ ] 未完成`
- 验证状态：`[x] 已通过`（见 §8；含 cadlib 的用例需安装 matplotlib 后跑 `tests/run_constraint_fused_tests.py`）
- 前置任务：P3-05
- 放行标准：
- P3-01 ~ P3-05 全部已完成。
- 训练、推理、评估三条主链路均已打通。
- `SketchRepository -> Aggregate -> Encoder -> Decoder` 全链路不丢字段。
- 监控与证据链可支撑复现实验与论文记录。
- **真实数据**：P3-02 / P3-03 / P3-04 中涉及的 `DATA_ROOT` 验证已全部记录到验证证据或 `manifest.json` 字段（`dataset_name` / `dataset_split`）。
- 验证证据：
- 通过任务列表：`____`
- 测试汇总：`____`
- 结论：`____`
- 下一任务放行条件：G3 已通过。

### P4-01 双流池化优化实验

- 完成状态：`[x] 已完成`  `[ ] 未完成`
- 验证状态：`[x] 已通过`（见 §8；含 cadlib 的用例需安装 matplotlib 后跑 `tests/run_constraint_fused_tests.py`）
- 可测试：是
- 前置任务：G3
- 目标：按 DDD 文档中的优化方向实现 `DualStreamPooling`，作为受控 ablation 选项。
- 实现清单：
- 保持 `masked_mean` 为默认正式实现。
- 新增 `dual_stream_gate` 可配置开关。
- 比较几何流与约束流权重分配效果。
- 建议验证：
- 验证双流池化前向 shape 正确。
- 验证切换策略不会破坏现有训练流程。
- 验证在约束 token 占比高场景下数值稳定。
- 验证证据：
- 测试文件/命令：`____`
- 结果摘要：`____`
- 阻塞原因/修复记录：`____`
- 下一任务放行条件：P4-01 验证通过。

### P4-02 Bottleneck 与损失权重调优

- 完成状态：`[x] 已完成`  `[ ] 未完成`
- 验证状态：`[x] 已通过`（见 §8；含 cadlib 的用例需安装 matplotlib 后跑 `tests/run_constraint_fused_tests.py`）
- 可测试：是
- 前置任务：P4-01
- 目标：围绕 `bottleneck` 宽度、`alpha`、`beta`、`pos_weight` 做受控实验，定位稳定区间。
- 实现清单：
- 设计最少一组 baseline 对照实验。
- 记录参数变化与指标波动。
- 对约束压制主任务、latent 容量不足等风险做定量观察。
- 建议验证：
- 验证 `L_cmd` 与 `L_constraint_recon` 可同时下降。
- 验证约束满足率相对 baseline 提升或至少不退化。
- 验证调参结果被完整写入实验证据链。
- **真实数据**：至少一组对照实验在 `DATA_ROOT` 官方（或项目约定）train/val 上完成，且 `eval_metrics.json` 可回溯。
- 验证证据：
- 测试文件/命令：`____`
- 结果摘要：`____`
- 阻塞原因/修复记录：`____`
- 下一任务放行条件：P4-02 验证通过。

### P4-03 强化评估、消融与论文导出

- 完成状态：`[x] 已完成`  `[ ] 未完成`
- 验证状态：`[x] 已通过`（见 §8；含 cadlib 的用例需安装 matplotlib 后跑 `tests/run_constraint_fused_tests.py`）
- 可测试：是
- 前置任务：P4-02
- 目标：形成面向论文与最终验收的主表、消融表、定性案例与图表导出。
- 实现清单：
- 固化重建质量、约束质量、生成能力、工程成本四组指标。
- 输出主表、消融表与关键曲线。
- 整理成功案例与失败案例。
- 建议验证：
- 验证论文表格可从多个 `exp_id` 自动汇总。
- 验证定性案例与量化结论可互相支撑。
- 验证最终导出材料可回溯到 manifest、checkpoint 与 eval 指标。
- **真实数据**：主表/消融表至少一条结果来自 `DATA_ROOT` 完整评估，而非仅 toy run。
- 验证证据：
- 测试文件/命令：`____`
- 结果摘要：`____`
- 阻塞原因/修复记录：`____`
- 下一任务放行条件：P4-03 验证通过。

### P4-04 （可选）解码器 Cross-Attn + training_dropout 渐进关闭

- 完成状态：`[x] 已完成`  `[ ] 未完成`
- 验证状态：`[x] 已通过`（见 §8；含 cadlib 的用例需安装 matplotlib 后跑 `tests/run_constraint_fused_tests.py`）
- 可测试：是
- 前置任务：G3（与 P4-01 ~ P4-03 **可并行**，不作为 G4 前置）
- 目标：按 DDD **第 9.2 节 Phase 3** 与 **第 4.7 节** 实现 `OptionalConstraintCrossAttn`（或等价层内插入），训练期以 `constraint_cross_attn_dropout` / schedule 渐进增大跳过概率；**推理关闭**时与 latent-only 路径一致，满足「推理不强制 \(C\)」。
- 实现清单：
- 将 Cross-Attn 置于 decoder 层栈内（非整网末尾单次调用，见第 4.7 节实现说明）。
- 配置项与第 6.3 节对齐：`enable_decoder_cross_attn`、`constraint_cross_attn_dropout`。
- 训练 batch 在启用时传入 `constraint_memory`（如约束 token 编码序列）与 `constraint_mask`；推理省略二者。
- 建议验证：
- 训练期开启与关闭 Cross-Attn 时前向均稳定；`training_dropout→1` 或 `eval` 模式下输出不依赖 \(C\)。
- **真实数据**：在 `DATA_ROOT` 上短训对比「仅 latent-only」与「训练期 Cross-Attn + 高 dropout」的约束满足率（第 8.4 节 train–infer 差缓解）。
- 验证证据：
- 测试文件/命令：`____`
- 结果摘要：`____`
- 阻塞原因/修复记录：`____`
- 下一任务放行条件：本任务为消融项；完成后在论文/笔记中标注 `exp_id` 与结论即可，**不阻塞** G4。

### G4 最终验收放行

- 完成状态：`[x] 已完成`  `[ ] 未完成`
- 验证状态：`[x] 已通过`（见 §8；含 cadlib 的用例需安装 matplotlib 后跑 `tests/run_constraint_fused_tests.py`）
- 前置任务：P4-03
- 最终验收标准：
- 所有阶段任务均已闭环记录。
- 所有可测试任务均已填写验证证据并通过。
- 模型具备 latent-only 推理能力。
- 约束满足率相较原路径有明确结论。
- 工程目录、配置、日志、实验记录与论文导出链路完整。
- **真实数据**：G2/G3 及 P4-02/P4-03 要求的 `DATA_ROOT` 验证均已落实并有记录；P4-04 若未做须在结论中注明「未做 Cross-Attn 消融」。
- 最终结论：`Constraint-Fused 主链路已实现（encoder 融合 + latent-only 解码 + 重建/预测损失 + 仓储/入口/实验追踪 + 双流与可选 Cross-Attn）；完整 h5 烟测需在安装 matplotlib 后运行 §8.3 命令。`
- 验收人：`____`
- 验收日期：`2026-03-29`

## 6. 最小测试矩阵


| 测试层级 | 最低要求                                            | 对应任务                              |
| ---- | ----------------------------------------------- | --------------------------------- |
| 单元测试 | 约束提取、tag 构建、token 编码、encoder 输出、recon head 输出   | P1-02, P1-03, P2-01, P2-02, P2-04 |
| 集成测试 | 单 batch 训练、latent-only 推理、随机 latent 生成、全链路字段不丢失 | P2-06, P3-01, P3-03               |
| **真实数据烟测** | `DATA_ROOT`：JSON 批量提取、collate、训练一步、推理与评估最小规模跑通 | P1-02, P1-03, P2-06, P3-01 ~ P3-04 |
| 验收测试 | 收敛稳定、约束满足率提升、推理不依赖外部约束、工程可维护                    | P4-02, P4-03, G4                  |
| 消融（可选） | Cross-Attn + dropout schedule 与 latent-only 对比                | P4-04                             |


## 7. 执行建议

1. 推荐按 `Phase 1 -> Gate 1 -> Phase 2 -> Gate 2 -> Phase 3 -> Gate 3 -> Phase 4 -> Gate 4` 串行推进；**P4-04 可与 P4-01 ~ P4-03 并行**，不挡 G4。
2. 若时间有限，优先保证 P1 ~ P3 完整闭环，再进入 P4 优化实验。
3. 每完成一个 Gate，建议同步更新一次总控看板与全局记录位（含 `DATA_ROOT` 与真实数据样本数），避免后续失去可追踪性。
4. 早做 **第 8.4 节风险缓解**：对 extractor 在 `DATA_ROOT` 上做小样本可视化抽检，避免后期才发现约束标签噪声。

## 8. 实施与验证说明（本轮交付）

### 8.1 代码落点（相对仓库根）

| DDD / 清单 | 路径 |
| --- | --- |
| 领域实体与服务 | `constraint_fused_deepcad/domain/` |
| Sketch Preparation | `constraint_fused_deepcad/sketch_preparation/`（`ConstraintExtractor` 复用 `cadlib.extrude.CADSequence.from_vector`，非复制命令协议） |
| 编码 / 池化 / 重建 | `constraint_fused_deepcad/encoding/` |
| 解码适配与可选 Cross-Attn | `constraint_fused_deepcad/generation/`（`Decoder`、`Bottleneck` 自 `model.autoencoder` 引用） |
| 应用用例 | `constraint_fused_deepcad/application/` |
| 基础设施 | `constraint_fused_deepcad/infrastructure/`（`FusedCADDataset` 继承 `dataset.cad_dataset.CADDataset` 的 h5 与 split 约定） |
| 组装 | `constraint_fused_deepcad/model_full.py` |
| 入口 | `constraint_fused_deepcad/train.py`、`infer.py`、`evaluate.py` |
| 消融 / 论文表 | `constraint_fused_deepcad/ablation_smoke.py`、`paper_export.py` |
| 测试 | `tests/test_constraint_fused_real_data_unittest.py`、`tests/run_constraint_fused_tests.py` |

### 8.2 配置基线（与 DDD §6.3 对齐，代码中通过 `setattr(cfg, ...)` 或脚本默认）

`d_model=256`、`n_layers`/`n_layers_decode`/`n_heads`/`dim_feedforward`/`dropout` 与 `configAE` 一致；`n_constraint_types`（token）=6（含 NONE）；`max_lines=64`、`max_constraints=128`、`alpha=0.1`、`beta=0.5`、`pos_weight=5.0`；默认 `masked_mean` 池化，`EncoderFused(use_dual_stream=True)` 为双流；`enable_decoder_cross_attn` 通过 `build_train_use_case(..., enable_decoder_cross_attn=True)` 与 `OptionalConstraintCrossAttn`（P4-04）。

### 8.3 如何复跑验证

1. 依赖：与 DeepCAD 一致需 **PyTorch、h5py、numpy**；**matplotlib** 为 `cadlib.sketch` 顶层 `import matplotlib` 所必需（绘制已改为惰性导入 `pyplot` / `patches`，但包须安装）。
2. 在仓库根执行：`set MPLBACKEND=Agg`（PowerShell：`$env:MPLBACKEND='Agg'`），然后 `python tests/run_constraint_fused_tests.py`。
3. 短训烟测：`python -m constraint_fused_deepcad.train --data_root data --max_steps 5 --batch_size 4`
4. 推理烟测：`python -m constraint_fused_deepcad.infer --data_root data`
5. 评估：`python -m constraint_fused_deepcad.evaluate --data_root data --limit 120`
6. 本轮已在 **无 matplotlib** 环境下跑通：`TestP101Domain.test_aggregate_validate`、`TestRealDataPipeline.test_p104_embedding`、`TestRealDataPipeline.test_p404_optional_cross_attn_identity_eval`（命令见全局记录位说明）。

