# DeepCAD 逐步添加几何约束任务清单

> 来源文档：`DeepCAD逐步添加几何约束技术方案.md`
>
> 目标：将 DeepCAD 逐步几何约束方案拆解为可执行、可验证、可记录、可逐阶段推进的任务清单。实现口径以原始 DeepCAD 为主架构基线：**保持生成闭包 `P(S | z)`；主 decoder 仍只依赖 `z`；总损失仅包含 `L_cmd`、`L_args`、`L_geom`；`L_args` 权重固定为原始 DeepCAD 的 `2.0`；`L_geom` 只奖励/约束 GT 中明确存在的水平、竖直、平行、垂直正关系被恢复，不默认启用双向 hard BCE 负样本惩罚。**

## 0. 模块映射速览

| 方案模块 | 本清单主要落点 |
| --- | --- |
| 原始 DeepCAD 主路径复现 | P0-01、P1-01、G1 |
| GT 关系提取模块 | P2-01、P2-02、G2 |
| 预测线几何解析模块 | P2-03、G2 |
| 正关系 `L_geom` 模块 | P3-01、P3-02、G3 |
| 训练编排与配置快照 | P3-03、P4-01 |
| 统一评估与消融 | P4-02、P4-03、G4 |

## 1. 使用规则

1. 本清单默认按阶段顺序执行，不建议跳过 Gate 直接进入后续任务。
2. 每个可测试任务完成后，必须先完成验证并记录结果，再进入下一任务。
3. 所有实现必须保留原始 DeepCAD 的 `P(S | z)` 主生成闭包，训练和推理阶段的主 decoder 不得依赖外部 `constraint_memory`。
4. `L_geom` 只作用于 decoder/FCN 之后的可解释输出或由 `args_logits` 解析得到的预测线方向，不得直接基于裸 `z` 计算几何损失。
5. 第一版 `L_geom` 只监督 GT 正关系恢复：`GT=1` 参与损失，`GT=0` 不作为强负样本惩罚。
6. 主任务权重固定为 `L_cmd + 2.0 * L_args`；若主任务明显退化，优先降低或关闭 `gamma_geom`，不得通过调低 `L_args` 权重掩盖问题。
7. 实验结果必须最终落到统一口径的 `ACC_cmd`、`ACC_param`、`summary.json`、`per_sample_counts.csv`，不能只看训练 loss 或训练期 `geom_*`。
8. 对比实验必须记录 `eval_split`、`grid_size`、`angle_thresh`、评估脚本、`ratio_h/ratio_v` 口径，避免跨口径比较。

## 2. 标记约定

- 完成状态：`[ ] 未完成` / `[x] 已完成`
- 验证状态：`[ ] 未验证` / `[ ] 验证中` / `[x] 已通过` / `[ ] 未通过`
- 阻塞状态：`[ ] 无` / `[ ] 有`

## 3. 总控看板

| ID | 阶段 | 任务 | 可测试 | 前置 | 完成 | 验证 |
| --- | --- | --- | --- | --- | --- | --- |
| P0-01 | Phase 0 | 范围冻结、架构红线确认与目录映射 | 否 | 无 | `[ ]` | `N/A` |
| P1-01 | Phase 1 | 原始 DeepCAD 基线复现与配置冻结 | 是 | P0-01 | `[ ]` | `[ ]` |
| P1-02 | Phase 1 | 原始 CADLoss 权重与训练日志字段锁定 | 是 | P1-01 | `[ ]` | `[ ]` |
| G1 | Gate 1 | 原始主路径闭环放行 | 是 | P1-02 | `[ ]` | `[ ]` |
| P2-01 | Phase 2 | GT Line 解析与 line index 契约实现 | 是 | G1 | `[ ]` | `[ ]` |
| P2-02 | Phase 2 | GT 正关系标签与 mask 生成 | 是 | P2-01 | `[ ]` | `[ ]` |
| P2-03 | Phase 2 | 预测线方向可微解析器接入 | 是 | P2-02 | `[ ]` | `[ ]` |
| G2 | Gate 2 | GT 关系与预测几何解析放行 | 是 | P2-03 | `[ ]` | `[ ]` |
| P3-01 | Phase 3 | 正关系 `L_geom` 计算模块 | 是 | G2 | `[ ]` | `[ ]` |
| P3-02 | Phase 3 | 总损失接线：`L_cmd + 2.0 * L_args + gamma_geom * L_geom` | 是 | P3-01 | `[ ]` | `[ ]` |
| P3-03 | Phase 3 | 配置开关、warmup、日志与回退策略 | 是 | P3-02 | `[ ]` | `[ ]` |
| G3 | Gate 3 | 训练链路放行 | 是 | P3-03 | `[ ]` | `[ ]` |
| P4-01 | Phase 4 | S0/S1/S2 短程冒烟与真实数据前反向 | 是 | G3 | `[ ]` | `[ ]` |
| P4-02 | Phase 4 | 测试集重建与统一口径评估 | 是 | P4-01 | `[ ]` | `[ ]` |
| P4-03 | Phase 4 | 消融：`gamma_geom`、`bce_scale`、正关系项权重 | 是 | P4-02 | `[ ]` | `[ ]` |
| G4 | Gate 4 | 阶段性实验闭环放行 | 是 | P4-03 | `[ ]` | `[ ]` |

## 4. 全局记录位

- 当前阶段：`____`
- 当前进行中的任务 ID：`____`
- 最近一次通过的 Gate：`____`
- 设计基线：`论文尝试/约束感知生成/7_constraint_fused_deepcad_step/DeepCAD逐步添加几何约束技术方案.md`
- 原始 DeepCAD 参考文档：`论文尝试/约束感知生成/0_original_deepcad/DeepCAD原始技术方案_详细版.md`
- A2d 参考文档：`论文尝试/约束感知生成/6_constraint_fused_deepcad_high_modify/主路径消融定位实验方案.md`
- 建议实现包目录：`constraint_fused_deepcad_step/` 或等价独立实验包
- 建议日志目录：`proj_log/constraint_fused_deepcad_step/<exp_name>/`
- 真实数据根路径 `DATA_ROOT`：`data`
- 原始 DeepCAD 对照 checkpoint：`____`
- Step 方案主实验 checkpoint：`____`
- 最近一次验证时间：`____`
- 最近一次失败任务：`____`
- 最近一次失败原因：`____`

## 5. 任务明细

### P0-01 范围冻结、架构红线确认与目录映射

- 完成状态：`[ ] 未完成`  `[ ] 已完成`
- 可测试：否
- 前置任务：无
- 目标：明确本方案是“原始 DeepCAD + 训练期正关系 `L_geom`”的低风险增量方案，冻结交付边界。
- 实现清单：
- 确认主路径沿用原始 DeepCAD：`Encoder -> Bottleneck -> Decoder(z) -> FCN`
- 明确总损失只允许 `L_cmd`、`L_args`、`L_geom`
- 明确 `L_args` 权重固定为 `2.0`
- 明确不引入 `L_pred`、`L_recon`、constraint token、constraint memory、decoder cross-attention
- 明确第一版不启用双向 hard BCE 负样本惩罚
- 确认新方案代码目录、配置文件、训练入口、评估入口与日志目录
- 产出物：
- 范围与红线说明
- 目录映射说明
- 与 A2c/A2d 差异说明

### P1-01 原始 DeepCAD 基线复现与配置冻结

- 完成状态：`[ ] 未完成`  `[ ] 已完成`
- 验证状态：`[ ] 未验证`
- 可测试：是
- 前置任务：P0-01
- 目标：冻结原始 DeepCAD 的模型结构、训练超参、评估口径，为新增几何损失提供可比基线。
- 实现清单：
- 固定 `dim_z=256`、DeepCAD bottleneck、原始 decoder 全局 `z` 注入与 `FCN` 双头输出
- 固定 `loss_cmd_weight=1.0`、`loss_args_weight=2.0`
- 固定 `data_root`、`train_val_test_split.json`、`eval_split`
- 记录 batch size、epoch、optimizer、lr schedule、gradient clip 等训练超参
- 建立 S0 实验名，例如 `deepcad_step_s0_origin`
- 建议验证：
- 单 batch 前向输出 `command_logits`、`args_logits` shape 正确
- 单 batch 反向无 NaN
- `loss = L_cmd + 2.0 * L_args` 与原始 DeepCAD 口径一致
- 验证证据：
- 测试文件/命令：`____`
- 结果摘要：`____`

### P1-02 原始 CADLoss 权重与训练日志字段锁定

- 完成状态：`[ ] 未完成`  `[ ] 已完成`
- 验证状态：`[ ] 未验证`
- 可测试：是
- 前置任务：P1-01
- 目标：确保新增 `L_geom` 前，主任务 loss 与日志字段已经稳定、可追踪、可对比。
- 实现清单：
- 日志记录 `loss_cmd_raw`、`loss_args_raw`、`loss_cmd`、`loss_args`、`loss_total`
- 确认 `loss_args = 2.0 * loss_args_raw`
- 保留 `CMD_ARGS_MASK`、EOS padding mask、visibility mask 等原始规则
- 配置快照写入 `config.txt` 或等价 JSON
- 建议验证：
- `gamma_geom=0` 时总损失只包含 `L_cmd + 2.0 * L_args`
- 训练日志可导出为 `train_metrics.csv`
- 验证证据：
- 测试文件/命令：`____`
- 结果摘要：`____`

### G1 原始主路径闭环放行

- 完成状态：`[ ] 未完成`  `[ ] 已完成`
- 验证状态：`[ ] 未验证`
- 前置任务：P1-02
- 放行标准：
- encoder、bottleneck、decoder、FCN 前向在真实 batch 上稳定
- 主 decoder 调用签名不需要 `constraint_memory`
- `L_cmd + 2.0 * L_args` 反向通过
- 可完成最小训练、保存 checkpoint、加载 checkpoint
- 结论：`____`

### P2-01 GT Line 解析与 line index 契约实现

- 完成状态：`[ ] 未完成`  `[ ] 已完成`
- 验证状态：`[ ] 未验证`
- 可测试：是
- 前置任务：G1
- 目标：从 GT CAD command/args 中解析有效 Line，并建立与序列 token 对齐的 line index 契约。
- 实现清单：
- 实现或复用 `parse_gt_lines(gt_commands, gt_args)`
- 输出 `line_index_map [B,S]`、`line_mask [B,L]`、`line_count [B]`
- 明确跨 `SOL` / profile / sketch 边界时 line 起点规则
- 首条 line 起点按方案设置为 `(0,0)`，后续 line 起点取同 sketch 内上一条 curve 的 end
- padding、EOS、非 Line 命令不产生有效 line
- 建议验证：
- 合成样本：单 line、多 line、跨 SOL、无 line、含 Arc/Ext
- 真实数据：随机抽样若干 batch，统计 line_count 分布
- 验证证据：
- 测试文件/命令：`____`
- 结果摘要：`____`

### P2-02 GT 正关系标签与 mask 生成

- 完成状态：`[ ] 未完成`  `[ ] 已完成`
- 验证状态：`[ ] 未验证`
- 可测试：是
- 前置任务：P2-01
- 目标：根据 GT line 方向生成正关系标签：水平、竖直、平行、垂直，以及对应有效 mask。
- 实现清单：
- 生成 `gt_horizontal [B,L]`
- 生成 `gt_vertical [B,L]`
- 生成 `gt_parallel [B,L,L]`
- 生成 `gt_perpendicular [B,L,L]`
- 生成 `pair_mask [B,L,L]`，排除 padding line 与 self-pair
- 固定并记录 `angle_thresh`
- GT=0 仅表示“不作为该关系正监督”，不得进入默认负样本损失
- 建议验证：
- 合成水平线、竖直线、平行 pair、垂直 pair 标签正确
- 正样本数量日志合理：`positive_count_h/v/parallel/perpendicular`
- pair 矩阵对称性符合预期
- 验证证据：
- 测试文件/命令：`____`
- 结果摘要：`____`

### P2-03 预测线方向可微解析器接入

- 完成状态：`[ ] 未完成`  `[ ] 已完成`
- 验证状态：`[ ] 未验证`
- 可测试：是
- 前置任务：P2-02
- 目标：从 decoder/FCN 输出的 `args_logits` 中解析预测 line 方向 `pred_unit`，作为 `L_geom` 的预测侧输入。
- 实现清单：
- 实现 `soft_argmax_args(args_logits)` 或等价可微离散参数期望
- 按 `line_index_map` 聚合 Line 终点参数
- 使用与 GT 解析一致的起点规则计算 `direction = end - start`
- 输出 `pred_unit [B,L,2]` 与 `pred_valid [B,L]`
- 对零长度方向使用 `clamp_min(eps)` 防止 NaN
- 本模块只读取 decoder 输出，不改变 decoder 输入
- 建议验证：
- `pred_unit.norm(dim=-1)` 在有效 line 上接近 1
- 早期随机模型也不产生 NaN/Inf
- 与 A2d corrected interpreter 的关键语义一致或偏差已记录
- 验证证据：
- 测试文件/命令：`____`
- 结果摘要：`____`

### G2 GT 关系与预测几何解析放行

- 完成状态：`[ ] 未完成`  `[ ] 已完成`
- 验证状态：`[ ] 未验证`
- 前置任务：P2-03
- 放行标准：
- GT 正关系标签可在真实 batch 上稳定生成
- `pred_unit` 可由 `args_logits` 可微解析得到
- `line_mask`、`pair_mask`、`line_index_map` 在 GT 与预测侧一致
- 所有统计项无 NaN/Inf
- 结论：`____`

### P3-01 正关系 `L_geom` 计算模块

- 完成状态：`[ ] 未完成`  `[ ] 已完成`
- 验证状态：`[ ] 未验证`
- 可测试：是
- 前置任务：G2
- 目标：实现只约束 GT 正关系恢复的 `L_geom`，不启用双向 hard BCE 负样本惩罚。
- 实现清单：
- 计算 `score_h = 1 - uy^2`
- 计算 `score_v = 1 - ux^2`
- 计算 `score_parallel = |u_i dot u_j|`
- 计算 `score_perpendicular = 1 - |u_i dot u_j|`
- 实现 `positive_bce(score, positive_mask, scale)`
- `positive_mask = gt_label * valid_mask`
- 当某类正样本数量为 0 时，该项 loss 返回 0
- 输出 `geom_h`、`geom_v`、`geom_parallel`、`geom_perpendicular`、`loss_geom`
- 建议验证：
- GT=1 且 score 高时 loss 小
- GT=1 且 score 低时 loss 大
- GT=0 不进入默认 loss
- 四类 loss 均可反向传播到 `args_logits`
- 验证证据：
- 测试文件/命令：`____`
- 结果摘要：`____`

### P3-02 总损失接线

- 完成状态：`[ ] 未完成`  `[ ] 已完成`
- 验证状态：`[ ] 未验证`
- 可测试：是
- 前置任务：P3-01
- 目标：在训练 use case 中接入三项总损失，严格保持原始 DeepCAD 主任务权重。
- 实现清单：
- 总损失为 `loss_total = loss_cmd + 2.0 * loss_args + gamma_geom * loss_geom`
- `gamma_geom=0` 时行为等价于原始 DeepCAD
- `loss_args_weight` 不暴露为本方案调参重点，默认固定 `2.0`
- 不接入 `L_pred`、`L_recon`、constraint head loss
- 训练日志记录三项 loss 及四类几何子项
- 建议验证：
- `gamma_geom=0` 与 S0 loss 数值路径一致
- `gamma_geom>0` 时 `args_logits.grad` 有来自 `L_geom` 的梯度
- 不存在任何对 `z` 直接计算几何 loss 的路径
- 验证证据：
- 测试文件/命令：`____`
- 结果摘要：`____`

### P3-03 配置开关、warmup、日志与回退策略

- 完成状态：`[ ] 未完成`  `[ ] 已完成`
- 验证状态：`[ ] 未验证`
- 可测试：是
- 前置任务：P3-02
- 目标：提供可控开关与保守默认值，便于从 S0/S1/S2 逐步推进。
- 实现清单：
- 新增或记录 `--enable_geom_loss`
- 新增或记录 `--geom_positive_only true`
- 新增或记录 `--gamma_geom 0.1`
- 新增或记录 `--geom_bce_scale 4.0`
- 新增或记录 `--geom_negative_weight 0.0`
- 新增或记录 `--geom_warmup_start_epoch`、`--geom_warmup_end_epoch` 或等价调度
- 日志记录 `positive_count_*`、`geom_*`、`parse_fail` 相关后续评估字段
- 回退策略：若 `loss_args` 或 `ACC_param` 明显恶化，先降 `gamma_geom` 或关闭 `L_geom`
- 建议验证：
- 配置快照完整写盘
- 关闭 `enable_geom_loss` 后不影响原始训练
- warmup 权重曲线符合预期
- 验证证据：
- 测试文件/命令：`____`
- 结果摘要：`____`

### G3 训练链路放行

- 完成状态：`[ ] 未完成`  `[ ] 已完成`
- 验证状态：`[ ] 未验证`
- 前置任务：P3-03
- 放行标准：
- S0、S1、S2 三种配置均可构建并跑通至少一个真实 batch
- `L_cmd`、`L_args`、`L_geom` 均为有限值
- `L_geom` 默认只使用正关系 mask
- `gamma_geom=0` 可完全回退到原始 DeepCAD 主路径
- 结论：`____`

### P4-01 S0/S1/S2 短程冒烟与真实数据前反向

- 完成状态：`[ ] 未完成`  `[ ] 已完成`
- 验证状态：`[ ] 未验证`
- 可测试：是
- 前置任务：G3
- 目标：在真实数据上验证三阶段实验路线，不用合成张量替代最终判断。
- 实现清单：
- S0：`gamma_geom=0`，确认原始主路径训练正常
- S1：启用 GT 关系提取与预测几何日志，但 `gamma_geom=0`
- S2：启用弱正关系 `L_geom`，如 `gamma_geom=0.1`
- 每组使用独立 `exp_name`
- 保存 checkpoint、config、train_metrics、manifest
- 建议验证：
- 三组均完成短程训练或 `max_steps` 冒烟
- S2 无明显 NaN、loss 爆炸、显存异常
- `positive_count_*` 非异常全 0
- 验证证据：
- 测试文件/命令：`____`
- 结果摘要：`____`

### P4-02 测试集重建与统一口径评估

- 完成状态：`[ ] 未完成`  `[ ] 已完成`
- 验证状态：`[ ] 未验证`
- 可测试：是
- 前置任务：P4-01
- 目标：用统一评估口径判断 `L_geom` 是否改善最终硬解码 CAD 结果。
- 实现清单：
- 对选定 checkpoint 执行测试集重建，生成 `*_vec.h5`
- 运行 `evaluation/evaluate_ae_acc.py` 或等价脚本得到 `ACC_cmd`、`ACC_param`
- 运行统一几何评估脚本得到 `summary.json`、`per_sample_counts.csv`
- 记录 `ratio_h`、`ratio_v`、`parallel_recall_index_aligned`、`perpendicular_recall_index_aligned`
- 记录 `n_parse_fail_pred`、`n_samples_extrude_count_mismatch`
- 明确评估脚本、`eval_split`、`grid_size`、`angle_thresh`
- 建议验证：
- 评估产物可复跑
- S0 与 S2 使用相同评估脚本和阈值
- 不用训练期 `geom_*` 替代测试集结论
- 验证证据：
- 测试文件/命令：`____`
- 结果摘要：`____`

### P4-03 消融：`gamma_geom`、`bce_scale`、正关系项权重

- 完成状态：`[ ] 未完成`  `[ ] 已完成`
- 验证状态：`[ ] 未验证`
- 可测试：是
- 前置任务：P4-02
- 目标：在 S2 初步有效或需要排障时，做小范围消融，定位正关系损失强度与几何收益。
- 实现清单：
- `gamma_geom`: `0.05 / 0.1 / 0.2 / 0.5`
- `geom_bce_scale`: `2.0 / 4.0 / 6.0`
- `relation_weights`: 默认全 1，仅在明确单项欠拟合时微调
- 默认保持 `geom_negative_weight=0.0`
- 禁止在首轮消融中直接切到 A2d 双向 hard BCE
- 对每组记录 `ACC_param` 退化幅度与几何指标增益
- 建议验证：
- 至少一组在 `ACC_param` 下降不超过 1 pt 时提升几何指标
- 若所有组均退化，回到 S1 检查解析与标签口径
- 验证证据：
- 测试文件/命令：`____`
- 结果摘要：`____`

### G4 阶段性实验闭环放行

- 完成状态：`[ ] 未完成`  `[ ] 已完成`
- 验证状态：`[ ] 未验证`
- 前置任务：P4-03
- 放行标准：
- 至少完成 S0/S1/S2 的训练与测试集统一评估
- `ACC_param` 相对 S0 下降不超过 1 pt，或已明确记录失败原因
- `ratio_h`、`ratio_v`、`parallel`、`perpendicular` 至少一项有可解释变化
- `summary.json`、`per_sample_counts.csv`、`config`、checkpoint 路径可追溯
- 结论：`____`

## 6. 阶段实验矩阵

| 实验 ID | 目的 | `gamma_geom` | `geom_positive_only` | `geom_negative_weight` | 备注 |
| --- | --- | ---: | --- | ---: | --- |
| S0 | 原始 DeepCAD 复现 | 0.0 | true | 0.0 | 只训 `L_cmd + 2.0 * L_args` |
| S1 | 几何解析日志自检 | 0.0 | true | 0.0 | 生成 GT 关系与预测 score 日志，不反传 |
| S2 | 弱正关系监督 | 0.1 | true | 0.0 | 第一版主实验 |
| S3a | 权重消融 | 0.05 | true | 0.0 | 更保守 |
| S3b | 权重消融 | 0.2 | true | 0.0 | 观察收益与主任务退化 |
| S3c | scale 消融 | 0.1 | true | 0.0 | `bce_scale=2/4/6` |
| S4 | 可选极弱负样本 | 0.1 | true | 0.01 | 仅在假阳性已证明影响结果时启用 |

## 7. 验收指标记录模板

| 实验 | checkpoint | `ACC_cmd` | `ACC_param` | `ratio_h` | `ratio_v` | `parallel` | `perpendicular` | `parse_fail` | `ext_mismatch` | 结论 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| S0 | ____ | ____ | ____ | ____ | ____ | ____ | ____ | ____ | ____ | ____ |
| S1 | ____ | ____ | ____ | ____ | ____ | ____ | ____ | ____ | ____ | ____ |
| S2 | ____ | ____ | ____ | ____ | ____ | ____ | ____ | ____ | ____ | ____ |
| S3a | ____ | ____ | ____ | ____ | ____ | ____ | ____ | ____ | ____ | ____ |
| S3b | ____ | ____ | ____ | ____ | ____ | ____ | ____ | ____ | ____ | ____ |

## 8. 风险与回退清单

| 风险 | 触发信号 | 优先处理 |
| --- | --- | --- |
| 主任务退化 | `loss_args` 明显升高或 `ACC_param` 下降超过 1 pt | 降低 `gamma_geom`，延后 warmup，必要时回到 S1 |
| 几何 loss 有效但测试无提升 | 训练 `geom_*` 下降但 `summary.json` 不改善 | 检查 GT 标签、预测解析、评估脚本口径是否一致 |
| 正样本过少 | `positive_count_*` 长期接近 0 | 检查 line 提取、pair mask、angle_thresh |
| NaN/Inf | `pred_unit`、`score_*` 或 loss 出现 NaN | 检查零长度线、soft argmax 温度、norm clamp |
| 假阳性过多 | 测试样本出现大量额外轴对齐或平行关系且影响 CAD | 先分析样本；必要时启用极弱 confident-negative margin |
| 评估不可比 | 不同实验 `grid_size`、`angle_thresh`、`ratio_h` 口径不同 | 复跑同一评估链，旧结果只做备注 |

## 9. 完成定义

本任务清单的阶段性完成标准为：

1. 已有可复现的 S0 原始 DeepCAD 基线。
2. 已有 GT 正关系提取与预测几何解析日志。
3. 已有只使用正关系监督的 `L_geom` 训练链路。
4. 已完成至少一组 S2 弱正关系训练与测试集评估。
5. 产物包含 checkpoint、config、训练日志、`summary.json`、`per_sample_counts.csv`。
6. 结论明确说明：该方案是否在不明显牺牲 `ACC_param` 的前提下改善了 `ratio_h/ratio_v/parallel/perpendicular`。
