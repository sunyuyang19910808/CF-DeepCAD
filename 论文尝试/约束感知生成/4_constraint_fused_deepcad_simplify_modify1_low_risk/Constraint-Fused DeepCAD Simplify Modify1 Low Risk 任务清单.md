# Constraint-Fused DeepCAD Simplify Modify1 Low Risk 任务清单

> 来源文档：`Constraint-Fused DeepCAD Simplify Modify1 Low Risk DDD技术方案.md`
>
> 目标：在保持 `constraint_fused_deepcad_simplify_modify1` 主体架构不变的前提下，通过 `LINE-only constraint_pred_loss`、`LINE` 相关 CE 加权、损失调度与固定评估闭环，低风险提升 `R_h`、`R_v` 与 aligned recall。

---

## 1. 任务使用说明

1. 本清单默认按 Phase 顺序执行。
2. 只有当前任务验证通过，才建议进入下一任务。
3. 若某任务失败，应优先记录阻塞原因与修复动作，不建议直接跳到后续任务。
4. Low Risk 方案不新增 pair-level 训练损失，不修改整体 `modify1` 主干。
5. 所有实现默认仍放在 `constraint_fused_deepcad_simplify_modify1/`，本目录仅存放设计、任务与架构文档。

---

## 2. 标记约定

- 完成状态：`[ ] 未完成` / `[x] 已完成`
- 验证状态：`[ ] 未验证` / `[ ] 验证中` / `[x] 已通过` / `[ ] 未通过`
- 阻塞状态：`[ ] 无` / `[ ] 有`

---

## 3. 总控看板

| ID | 阶段 | 任务 | 可测试 | 前置 | 完成 | 验证 |
| --- | --- | --- | --- | --- | --- | --- |
| P0-01 | Phase 0 | Low Risk 范围冻结与基线确认 | 否 | 无 | `[ ]` | `N/A` |
| P1-01 | Phase 1 | `constraint_pred_loss` 改为 LINE-only mask | 是 | P0-01 | `[ ]` | `[ ]` |
| P1-02 | Phase 1 | `train_use_case` 传递 `line_cmd_mask` 到 pred loss | 是 | P1-01 | `[ ]` | `[ ]` |
| G1 | Gate 1 | 监督对齐放行 | 是 | P1-02 | `[ ]` | `[ ]` |
| P2-01 | Phase 2 | `CommandCadLoss` 的 LINE 命令加权 | 是 | G1 | `[ ]` | `[ ]` |
| P2-02 | Phase 2 | `CommandCadLoss` 的 line 参数加权 | 是 | P2-01 | `[ ]` | `[ ]` |
| P2-03 | Phase 2 | 新增 `line_cmd_weight` / `line_args_weight` 配置项 | 是 | P2-02 | `[ ]` | `[ ]` |
| G2 | Gate 2 | 主任务加权放行 | 是 | P2-03 | `[ ]` | `[ ]` |
| P3-01 | Phase 3 | `alpha/gamma` warmup 或 staged ramp | 是 | G2 | `[ ]` | `[ ]` |
| P3-02 | Phase 3 | 训练日志补充有效损失权重记录 | 是 | P3-01 | `[ ]` | `[ ]` |
| G3 | Gate 3 | 调度策略放行 | 是 | P3-02 | `[ ]` | `[ ]` |
| P4-01 | Phase 4 | 固定 `angle_thresh=0.1` 的评估脚本运行模板 | 是 | G3 | `[ ]` | `[ ]` |
| P4-02 | Phase 4 | Low Risk 实验结果记录模板固化 | 是 | P4-01 | `[ ]` | `[ ]` |
| P4-03 | Phase 4 | 与当前 Modify1 基线对比实验 | 是 | P4-02 | `[ ]` | `[ ]` |
| G4 | Gate 4 | Low Risk 结论输出 | 是 | P4-03 | `[ ]` | `[ ]` |

---

## 4. 全局记录位

- 当前阶段：`Phase 0`
- 当前进行中的任务：`P0-01`
- 最近一次通过的 Gate：`____`
- 数据根路径 `DATA_ROOT`：`data`
- 训练配置文件：`constraint_fused_deepcad_simplify_modify1/config/config_constraint_fused_simplify_modify1.py`
- Low Risk 设计基线：`论文尝试/约束感知生成/4_constraint_fused_deepcad_simplify_modify1_low_risk/Constraint-Fused DeepCAD Simplify Modify1 Low Risk DDD技术方案.md`
- 当前 Modify1 基线指标文件：`constraint_fused_deepcad_simplify_modify1/summary.json`
- 最近一次验证时间：`____`
- 最近一次失败任务：`____`
- 最近一次失败原因：`____`

> 执行口径说明：本清单采用严格串行验收制。所有“可测试=是”的任务，必须先完成实现并验证通过，才能进入下一任务。

---

## 5. 任务明细

### P0-01 Low Risk 范围冻结与基线确认

- 完成状态：`[ ] 未完成`  `[ ] 已完成`
- 可测试：否
- 前置任务：无
- 目标：确认本轮仅做低风险增量，不改 `modify1` 主体架构。
- 实现清单：
- 确认不新增 pair-level 训练损失
- 确认不增加推理期 snapping / 硬投影
- 确认不修改 encoder / decoder 主干结构
- 固定当前基线指标文件与比较口径
- 产出物：
- 范围说明
- 当前基线 `summary.json` 记录
- 实验对比模板起始版本

### P1-01 `constraint_pred_loss` 改为 LINE-only mask

- 完成状态：`[ ] 未完成`  `[ ] 已完成`
- 验证状态：`[ ] 未验证`
- 可测试：是
- 前置任务：P0-01
- 目标：让 decoder 侧约束预测只在 `LINE` token 上计算 BCE。
- 实现清单：
- 扩展 `constraint_pred_loss()` 入参，支持 `line_cmd_mask`
- 有效 mask 从 `~cmd_padding_mask` 改为 `(~cmd_padding_mask) & line_cmd_mask`
- 保留 backward 兼容或同步修改调用方
- 建议验证：
- 最小样本验证只有 `LINE` 位置参与 loss
- 非 `LINE` 位置目标不会再产生有效监督
- loss 数值有限且无 NaN
- 验证证据：
- 测试文件/命令：`python -m unittest tests.test_constraint_fused_deepcad_simplify_modify1`
- 结果摘要：`____`

### P1-02 `train_use_case` 传递 `line_cmd_mask` 到 pred loss

- 完成状态：`[ ] 未完成`  `[ ] 已完成`
- 验证状态：`[ ] 未验证`
- 可测试：是
- 前置任务：P1-01
- 目标：确保训练编排中 `constraint_pred_loss` 使用的 mask 与 batch 契约一致。
- 实现清单：
- 在 `TrainConstraintFusedSimplifyModify1BatchUseCase.execute()` 中传入 `line_cmd_mask`
- 确认 batch 字段不回归
- 检查 pred loss 与其他 loss 可以共同组合
- 建议验证：
- 单 batch 前向可跑通
- `pred_loss` 数值正常
- 不影响 `axis_loss` / `geom_loss` / `cmd_loss`
- 验证证据：
- 测试文件/命令：`python -m unittest tests.test_constraint_fused_deepcad_simplify_modify1`
- 结果摘要：`____`

### G1 监督对齐放行

- 完成状态：`[ ] 未完成`  `[ ] 已完成`
- 验证状态：`[ ] 未验证`
- 前置任务：P1-02
- 放行标准：
- `constraint_pred_loss` 只在 `LINE` token 上生效
- `train_use_case` 中各损失仍能稳定共存
- 现有 batch 契约没有被破坏
- 结论：`____`

### P2-01 `CommandCadLoss` 的 LINE 命令加权

- 完成状态：`[ ] 未完成`  `[ ] 已完成`
- 验证状态：`[ ] 未验证`
- 可测试：是
- 前置任务：G1
- 目标：提高 `LINE` 命令位置的 `loss_cmd` 权重。
- 实现清单：
- 在 `CommandCadLoss` 中识别 `tgt_commands == LINE_IDX`
- 对 `LINE` 命令位置构造更高权重
- 保留默认行为可回退
- 建议验证：
- `LINE` 命令位置 loss 更高
- 总 loss 数值稳定
- 不影响非 `LINE` 位置基本训练逻辑
- 验证证据：
- 测试文件/命令：`python -m unittest tests.test_constraint_fused_deepcad_simplify_modify1`
- 结果摘要：`____`

### P2-02 `CommandCadLoss` 的 line 参数加权

- 完成状态：`[ ] 未完成`  `[ ] 已完成`
- 验证状态：`[ ] 未验证`
- 可测试：是
- 前置任务：P2-01
- 目标：提高 line 参数对应的 `loss_args` 权重。
- 实现清单：
- 为 line 位置的参数预测添加更高权重
- 与命令 mask、参数 mask 保持兼容
- 允许通过配置关闭
- 建议验证：
- line 参数项 loss 随权重变化
- loss 无 NaN/Inf
- 不影响其他参数的可训练性
- 验证证据：
- 测试文件/命令：`python -m unittest tests.test_constraint_fused_deepcad_simplify_modify1`
- 结果摘要：`____`

### P2-03 新增 `line_cmd_weight` / `line_args_weight` 配置项

- 完成状态：`[ ] 未完成`  `[ ] 已完成`
- 验证状态：`[ ] 未验证`
- 可测试：是
- 前置任务：P2-02
- 目标：把 line-aware weighting 变成明确可配置能力。
- 实现清单：
- 在配置文件中新增权重项
- 提供默认值与说明
- 保持旧配置可兼容
- 建议验证：
- 默认配置可正常训练
- 改动配置值后 loss 曲线确有变化
- 配置文件序列化与日志正常
- 验证证据：
- 测试文件/命令：`python -m unittest tests.test_constraint_fused_deepcad_simplify_modify1`
- 结果摘要：`____`

### G2 主任务加权放行

- 完成状态：`[ ] 未完成`  `[ ] 已完成`
- 验证状态：`[ ] 未验证`
- 前置任务：P2-03
- 放行标准：
- `LINE` 命令与 line 参数加权逻辑可控
- 默认配置下训练闭环不回归
- 相关配置项可被日志与保存流程读取
- 结论：`____`

### P3-01 `alpha/gamma` warmup 或 staged ramp

- 完成状态：`[ ] 未完成`  `[ ] 已完成`
- 验证状态：`[ ] 未验证`
- 可测试：是
- 前置任务：G2
- 目标：降低辅助损失在训练早期对主任务的扰动。
- 实现清单：
- 为 `alpha` / `gamma` 增加 warmup 或分阶段开启
- 明确生效步数或 epoch 口径
- 保持 `beta` 行为不变或单独说明
- 建议验证：
- 前期有效 `alpha/gamma` 符合预期
- 训练日志可读出当前生效权重
- 总损失与各子损失数值稳定
- 验证证据：
- 测试文件/命令：`python -m unittest tests.test_constraint_fused_deepcad_simplify_modify1`
- 结果摘要：`____`

### P3-02 训练日志补充有效损失权重记录

- 完成状态：`[ ] 未完成`  `[ ] 已完成`
- 验证状态：`[ ] 未验证`
- 可测试：是
- 前置任务：P3-01
- 目标：让实验日志显式记录当前 `alpha/gamma` 生效值，便于分析。
- 实现清单：
- 在训练日志或 TensorBoard 中记录 `alpha_eff` / `gamma_eff`
- 若存在 staged 参数，也同步记录阶段切换点
- 建议验证：
- 日志可直接读到有效权重
- 权重变化时间点与训练设计一致
- 不影响现有 loss 记录
- 验证证据：
- 测试文件/命令：`python -m unittest tests.test_constraint_fused_deepcad_simplify_modify1`
- 结果摘要：`____`

### G3 调度策略放行

- 完成状态：`[ ] 未完成`  `[ ] 已完成`
- 验证状态：`[ ] 未验证`
- 前置任务：P3-02
- 放行标准：
- 有效权重调度逻辑稳定
- 日志能清楚复盘每轮训练的真实权重
- 训练未因调度逻辑引入明显不稳定
- 结论：`____`

### P4-01 固定 `angle_thresh=0.1` 的评估脚本运行模板

- 完成状态：`[ ] 未完成`  `[ ] 已完成`
- 验证状态：`[ ] 未验证`
- 可测试：是
- 前置任务：G3
- 目标：统一 Low Risk 实验的对比口径。
- 实现清单：
- 固定 `angle_thresh=0.1`
- 固定 reconstruction 与 metrics 路径约定
- 固定输出 `summary.json` 与 `per_sample_counts.csv`
- 建议验证：
- 能重复跑出同口径结果
- 输出字段齐全
- 与当前 `modify1` 基线可一对一对比
- 验证证据：
- 测试文件/命令：`python -m constraint_fused_deepcad_simplify_modify1.reconstruct ...`；`python -m constraint_fused_deepcad_simplify_modify1.evaluate --skip_reconstruct ... --angle_thresh 0.1`
- 结果摘要：`____`

### P4-02 Low Risk 实验结果记录模板固化

- 完成状态：`[ ] 未完成`  `[ ] 已完成`
- 验证状态：`[ ] 未验证`
- 可测试：是
- 前置任务：P4-01
- 目标：保证每轮实验都记录同一组关键指标。
- 实现清单：
- 固定记录 `R_h`、`R_v`
- 固定记录 `parallel_recall_index_aligned`
- 固定记录 `perpendicular_recall_index_aligned`
- 固定记录 `n_parse_fail_pred`
- 固定记录训练配置与 checkpoint
- 建议验证：
- 至少完成 1 份模板化记录
- 模板能支持多实验横向比较
- 验证证据：
- 结果摘要：`____`

### P4-03 与当前 Modify1 基线对比实验

- 完成状态：`[ ] 未完成`  `[ ] 已完成`
- 验证状态：`[ ] 未验证`
- 可测试：是
- 前置任务：P4-02
- 目标：用统一口径判断 Low Risk 方案是否有效。
- 实现清单：
- 训练 Low Risk 版本模型
- 生成 reconstruction
- 运行统一评估
- 与当前 `modify1` 基线结果横向对比
- 建议验证：
- `R_h`、`R_v` 至少有一项稳定提升
- aligned recall 无明显退化
- 如无提升，形成明确结论并停止继续调参
- 验证证据：
- 结果摘要：`____`

### G4 Low Risk 结论输出

- 完成状态：`[ ] 未完成`  `[ ] 已完成`
- 验证状态：`[ ] 未验证`
- 前置任务：P4-03
- 放行标准：
- 已完成至少 1 组统一口径实验
- 已明确判断 Low Risk 是否值得继续
- 若收益有限，已给出进入中风险方案的依据
- 结论：`____`

---

## 6. 实验记录附录

### 6.1 建议记录模板

| 实验名 | 改动点 | `line_cmd_weight` | `line_args_weight` | `alpha/gamma` 调度 | `R_h` | `R_v` | `parallel_recall` | `perpendicular_recall` | 结论 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline_modify1 | 当前基线 | 1.0 | 1.0 | 固定 | ____ | ____ | ____ | ____ | ____ |
| low_risk_v1 | LINE-only pred loss | ____ | ____ | ____ | ____ | ____ | ____ | ____ | ____ |
| low_risk_v2 | + line-aware CE | ____ | ____ | ____ | ____ | ____ | ____ | ____ | ____ |

### 6.2 结论口径

若出现以下任一情况，则建议停止 Low Risk 继续扩展并转向中风险方案：

1. `R_h`、`R_v` 连续多轮实验无明显提升
2. aligned recall 与 parse fail 指标没有改善
3. 辅助 loss 下降但最终几何指标不动
