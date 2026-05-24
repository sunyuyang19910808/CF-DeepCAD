# Constraint-Fused DeepCAD Simplify 任务清单

> 来源文档：`Constraint-Fused DeepCAD Simplify DDD技术方案.md`
>
> 目标：围绕 `constraint_fused_deepcad_simplify/` 建立一个只考虑**水平 / 竖直**约束的最小可验证版本，优先验证“约束进入 encoder 是否有效”。

---

## 1. 任务使用说明

1. 本清单默认按 Phase 顺序执行。
2. 只有当前任务验证通过，才建议进入下一任务。
3. 若某任务失败，应记录阻塞原因，不直接跳到后续模型优化项。
4. 本轮不做 pair 约束、token 联合序列、Cross-Attn 与 Latent GAN。
5. 所有新代码默认放在 `constraint_fused_deepcad_simplify/`。

---

## 2. 标记约定

- 完成状态：`[ ] 未完成` / `[x] 已完成`
- 验证状态：`[ ] 未验证` / `[ ] 验证中` / `[x] 已通过` / `[ ] 未通过`
- 阻塞状态：`[ ] 无` / `[ ] 有`

---

## 3. 总控看板

| ID | 阶段 | 任务 | 可测试 | 前置 | 完成 | 验证 |
| --- | --- | --- | --- | --- | --- | --- |
| P0-01 | Phase 0 | 范围冻结与目录建立 | 否 | 无 | `[x]` | `N/A` |
| P1-01 | Phase 1 | 领域实体与聚合根骨架 | 是 | P0-01 | `[x]` | `[x]` |
| P1-02 | Phase 1 | H/V 约束提取器 | 是 | P1-01 | `[x]` | `[x]` |
| P1-03 | Phase 1 | Batch Assembler 与监督张量 | 是 | P1-02 | `[x]` | `[x]` |
| G1 | Gate 1 | 数据闭环放行 | 是 | P1-03 | `[x]` | `[x]` |
| P2-01 | Phase 2 | AxisTagEmbedding 与命令嵌入融合 | 是 | G1 | `[x]` | `[x]` |
| P2-02 | Phase 2 | EncoderSimplify 与 Bottleneck | 是 | P2-01 | `[x]` | `[x]` |
| P2-03 | Phase 2 | AxisReconHead 与 LossComposer | 是 | P2-02 | `[x]` | `[x]` |
| P2-04 | Phase 2 | 单 Batch 训练用例 | 是 | P2-03 | `[x]` | `[x]` |
| G2 | Gate 2 | 模型主链路放行 | 是 | P2-04 | `[x]` | `[x]` |
| P3-01 | Phase 3 | 评估脚本 `R_h/R_v` | 是 | G2 | `[x]` | `[x]` |
| P3-02 | Phase 3 | 与原版基线对比实验 | 是 | P3-01 | `[ ]` | `[ ]` |
| P3-03 | Phase 3 | 日志、checkpoint、配置固化 | 是 | P3-02 | `[ ]` | `[ ]` |
| G3 | Gate 3 | 简化版验证结论输出 | 是 | P3-03 | `[ ]` | `[ ]` |

---

## 4. 全局记录位

- 当前阶段：`Phase 3`
- 当前进行中的任务：`P3-02`
- 最近一次通过的 Gate：`G2`
- 数据根路径 `DATA_ROOT`：`data`
- 训练配置文件：`constraint_fused_deepcad_simplify/config/config_constraint_fused_simplify.py`
- 最近一次验证时间：`2026-04-09 23:45:37`
- 最近一次失败任务：`____`
- 最近一次失败原因：`____`

---

## 5. 任务明细

### P0-01 范围冻结与目录建立

- 完成状态：`[x] 已完成`  `[ ] 未完成`
- 可测试：否
- 前置任务：无
- 目标：确认本轮只做 H/V 最小闭环，并建立独立目录。
- 实现清单：
- 建立 `constraint_fused_deepcad_simplify/` 目录骨架。
- 冻结范围：不做 pair 约束、不做 token encoder、不做 Cross-Attn。
- 确认配置文件命名、训练入口、评估入口。
- 产出物：
- 目录结构
- `config_constraint_fused_simplify.py`
- 范围说明
- 备注：`已建立独立目录骨架、独立 train/evaluate 入口与范围说明，未引用 constraint_fused_deepcad 包。`

### P1-01 领域实体与聚合根骨架

- 完成状态：`[x] 已完成`  `[ ] 未完成`
- 验证状态：`[x] 已通过`  `[ ] 未验证`
- 可测试：是
- 前置任务：P0-01
- 目标：定义最小领域对象，保证后续数据流有统一语义载体。
- 实现清单：
- 实现 `ConstraintTypeSimplify`
- 实现 `ConstraintRelationSimplify`
- 实现 `SketchSequenceAggregateSimplify`
- 定义 `line_count`、`cmd_padding_mask`、`constraint_tags`、`unary_gt` 的 shape 约定
- 建议验证：
- 构造最小样本，检查字段完整性
- 检查非法 `line_idx` 是否能被拒绝
- 验证证据：
- 测试文件/命令：`python -m unittest tests.test_constraint_fused_deepcad_simplify`
- 结果摘要：`已验证最小样本字段完整性；非法 line_idx 会抛出 ValueError。`
- 阻塞原因/修复记录：`无`

### P1-02 H/V 约束提取器

- 完成状态：`[x] 已完成`  `[ ] 未完成`
- 验证状态：`[x] 已通过`  `[ ] 未验证`
- 可测试：是
- 前置任务：P1-01
- 目标：从 `cad_vec` / `CADSequence` 中提取水平、竖直线索引。
- 实现清单：
- 建立 `ConstraintExtractorSimplify`
- 复用 line 收集逻辑
- 实现与 x/y 轴夹角判定
- 支持从 `cad_vec` 直接提取
- 建议验证：
- 人工构造水平线与竖直线样本
- 在真实数据上抽样至少 32 条样本跑通
- 对照 `论文尝试/DeepCAD原始约束指标` 口径确认一致
- 验证证据：
- 测试文件/命令：`python -m unittest tests.test_constraint_fused_deepcad_simplify`
- 结果摘要：`已通过人工构造的水平/竖直线样本验证；提取实现基于 CADSequence.from_vector，与现有 axis 评估口径一致。`
- 阻塞原因/修复记录：`未单独执行 32 条真实样本抽样脚本，待 P3 实验阶段补齐。`

### P1-03 Batch Assembler 与监督张量

- 完成状态：`[x] 已完成`  `[ ] 未完成`
- 验证状态：`[x] 已通过`  `[ ] 未验证`
- 可测试：是
- 前置任务：P1-02
- 目标：把 H/V 约束转成 encoder 可消费的标签与监督。
- 实现清单：
- 实现 `build_constraint_tags(seq_len, commands_np, relations)`
- 实现 `build_unary_gt(max_lines, relations, line_count)`
- 构建 `cmd_padding_mask`
- 输出 `SketchSequenceAggregateSimplify`
- 建议验证：
- 检查 `constraint_tags` 是否只打在 `LINE` 命令位置
- 检查 `unary_gt` 中 H/V 维度是否正确
- 在至少 2 个真实 batch 上检查 shape 无越界
- 验证证据：
- 测试文件/命令：`python -m unittest tests.test_constraint_fused_deepcad_simplify`
- 结果摘要：`constraint_tags 仅在 LINE 命令位置打标；unary_gt 维度与 line_count 对齐；dataset_simplify 已输出聚合后的监督张量。`
- 阻塞原因/修复记录：`真实数据 shape 烟雾验证已在单 batch 训练中覆盖 1 个 batch，待扩大到至少 2 个真实 batch。`

### G1 数据闭环放行

- 完成状态：`[x] 已完成`  `[ ] 未完成`
- 验证状态：`[x] 已通过`  `[ ] 未验证`
- 前置任务：P1-03
- 放行标准：
- P1-01 至 P1-03 全部已完成
- 能从真实数据生成 `constraint_tags` 与 `unary_gt`
- `dataset_simplify` 能返回训练样本
- 结论：`P1-01 至 P1-03 已完成；真实数据单 batch 已能生成 constraint_tags / unary_gt，dataset_simplify 可返回训练样本。`

### P2-01 AxisTagEmbedding 与命令嵌入融合

- 完成状态：`[x] 已完成`  `[ ] 未完成`
- 验证状态：`[x] 已通过`  `[ ] 未验证`
- 可测试：是
- 前置任务：G1
- 目标：让 H/V 标签进入 encoder embedding。
- 实现清单：
- 实现 `AxisTagEmbedding`
- 修改命令 embedding 以接收 `(S, N, 2)` 的 `constraint_tags`
- 保持与原位置编码兼容
- 建议验证：
- 有无标签时都能稳定前向
- 输出 shape 与 dtype 正确
- 不出现 NaN
- 验证证据：
- 测试文件/命令：`python -m unittest tests.test_constraint_fused_deepcad_simplify`
- 结果摘要：`AxisTagEmbedding 已接入命令嵌入，单 batch 前向输出 shape 正确且未出现 NaN。`

### P2-02 EncoderSimplify 与 Bottleneck

- 完成状态：`[x] 已完成`  `[ ] 未完成`
- 验证状态：`[x] 已通过`  `[ ] 未验证`
- 可测试：是
- 前置任务：P2-01
- 目标：完成最小 encoder 侧融合链路。
- 实现清单：
- 实现 `EncoderSimplify`
- 接入 `TransformerEncoder`
- 接入 `MaskedMeanPooling`
- 接入 `Bottleneck`
- 建议验证：
- `z` 形状为 `(1, N, d_model)`
- 不同长度样本 batch 下 mask 正常
- encoder 输出稳定可反向传播
- 验证证据：
- 测试文件/命令：`python -m unittest tests.test_constraint_fused_deepcad_simplify`
- 结果摘要：`EncoderSimplify 输出 z 形状满足 (1, N, d_model)，mask 与 bottleneck 已接通。`

### P2-03 AxisReconHead 与 LossComposer

- 完成状态：`[x] 已完成`  `[ ] 未完成`
- 验证状态：`[x] 已通过`  `[ ] 未验证`
- 可测试：是
- 前置任务：P2-02
- 目标：建立最小约束监督闭环。
- 实现清单：
- 实现 `AxisReconHead`
- 实现 `L_axis_recon`
- 定义 `L_total = L_cmd + beta * L_axis_recon`
- 建立 `line_mask`
- 建议验证：
- `unary_pred` 形状与 `unary_gt` 一致
- loss 可正常下降
- 空约束样本不报错
- 验证证据：
- 测试文件/命令：`python -m unittest tests.test_constraint_fused_deepcad_simplify`
- 结果摘要：`unary_pred 形状与 unary_gt 一致；AxisReconHead、line_mask 与总损失组合已跑通。`

### P2-04 单 Batch 训练用例

- 完成状态：`[x] 已完成`  `[ ] 未完成`
- 验证状态：`[x] 已通过`  `[ ] 未验证`
- 可测试：是
- 前置任务：P2-03
- 目标：把 dataset、encoder、decoder、recon head、loss 串成可训练的单 batch。
- 实现清单：
- 实现 `TrainConstraintFusedSimplifyBatchUseCase`
- 接入现有 decoder 或轻量 adapter
- 输出 `loss`、`loss_cmd`、`axis_loss`
- 建议验证：
- 至少跑通 1 个 batch 反向传播
- loss 数值非 NaN / Inf
- 显存占用与完整方案相比明显更低
- 验证证据：
- 测试文件/命令：`python -m constraint_fused_deepcad_simplify.train --device cpu --gpu_ids cpu --batch_size 1 --num_workers 0 --nr_epochs 1 --max_steps 1 --proj_dir proj_log --exp_name cf_simplify_smoke --force_overwrite`
- 结果摘要：`真实数据 1 个 batch 已完成前向、反向和 latest checkpoint 保存；loss=15.1，axis_loss=3.28，数值非 NaN/Inf。`

### G2 模型主链路放行

- 完成状态：`[x] 已完成`  `[ ] 未完成`
- 验证状态：`[x] 已通过`  `[ ] 未验证`
- 前置任务：P2-04
- 放行标准：
- P2-01 至 P2-04 全部完成
- 单 batch 训练可稳定运行
- `axis_loss` 能被记录
- 结论：`P2-01 至 P2-04 全部完成；单 batch 训练可稳定运行，axis_loss 已写入训练日志链路。`

### P3-01 评估脚本 `R_h/R_v`

- 完成状态：`[x] 已完成`  `[ ] 未完成`
- 验证状态：`[x] 已通过`  `[ ] 未验证`
- 可测试：是
- 前置任务：G2
- 目标：构建简化版效果判断标准。
- 实现清单：
- 实现 `evaluate_axis_constraints.py`
- 对预测 `cad_vec` 重新提取 H/V
- 聚合得到 `R_h`、`R_v`
- 可选补充 `axis_precision_mean`、`axis_recall_mean`
- 建议验证：
- 与 `论文尝试/DeepCAD原始约束指标/eval_original_deepcad_axis_constraints.py` 结果口径一致
- 在小测试集上输出稳定 summary
- 验证证据：
- 测试文件/命令：`python -m unittest tests.test_constraint_fused_deepcad_simplify`
- 结果摘要：`已实现 evaluate_axis_constraints.py，支持重建结果再提取 H/V 并输出 R_h、R_v、axis_precision_mean、axis_recall_mean；聚合逻辑测试通过。`

### P3-02 与原版基线对比实验

- 完成状态：`[ ] 已完成`  `[ ] 未完成`
- 验证状态：`[ ] 已通过`  `[ ] 未验证`
- 可测试：是
- 前置任务：P3-01
- 目标：判断简化版是否值得继续扩展。
- 实现清单：
- 选择基线模型或原始 checkpoint
- 在同一测试集上对比 `R_h/R_v`
- 记录 `loss_cmd` 与 `axis_loss`
- 输出实验结论表
- 建议验证：
- 至少比较 1 组 baseline vs simplify
- 结论包含“是否继续做 token / pair 扩展”
- 验证证据：
- 实验记录路径：`____`
- 结果摘要：`____`

### P3-03 日志、checkpoint、配置固化

- 完成状态：`[ ] 已完成`  `[ ] 未完成`
- 验证状态：`[ ] 已通过`  `[ ] 未验证`
- 可测试：是
- 前置任务：P3-02
- 目标：让简化版成为一个可复现实验分支，而不是一次性脚本。
- 实现清单：
- 固化配置文件
- 补充训练入口与评估入口
- 增加 checkpoint 保存
- 增加 tensorboard / csv 日志
- 建议验证：
- 能复现实验
- 能从 checkpoint 恢复继续训练
- 验证证据：
- 测试文件/命令：`____`
- 结果摘要：`____`

### G3 简化版验证结论输出

- 完成状态：`[ ] 已完成`  `[ ] 未完成`
- 验证状态：`[ ] 已通过`  `[ ] 未验证`
- 前置任务：P3-03
- 最终放行标准：
- 训练、验证、评估全部打通
- 至少产出 1 份对比实验记录
- 明确回答以下问题：
- H/V tag 注入是否有效
- 是否值得继续扩展到 token encoder
- 是否值得继续扩展到 pair 约束
- 最终结论：`____`

---

## 6. 推荐优先级

如果时间非常紧，建议优先完成下面 6 项：

1. `P0-01`
2. `P1-02`
3. `P1-03`
4. `P2-01`
5. `P2-04`
6. `P3-01`

这样就能最快得到一个能回答“这个方向值不值得继续”的实验闭环。

---

## 7. 建议验收口径

建议把最终验收压缩为三个问题：

1. `constraint_fused_deepcad_simplify` 是否能独立跑通训练与评估。
2. 引入 H/V tag 后，`R_h`、`R_v` 是否比基线更接近 1。
3. `axis_loss` 是否在训练中持续下降，且与生成质量没有明显冲突。

若这三点成立，则说明该简化版值得继续扩展到完整 Fused 结构。
