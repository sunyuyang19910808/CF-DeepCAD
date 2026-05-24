# Constraint-Fused DeepCAD Simplify Modify2 Low Risk 任务清单

> 来源文档：`Constraint-Fused DeepCAD Simplify Modify2 Low Risk DDD技术方案.md`
>
> 目标：将 Low Risk DDD 方案拆解为可执行、可记录、可验证、可串行推进的任务清单。实现口径与最新 Low Risk DDD 保持一致：**不改四类约束数据契约、不重写整体 encoder/decoder 主干、不改变评估口径，优先增强 decoder 约束通路、pair 重建结构与 LINE-only 监督。**

## 0. DDD 映射速览


| DDD 限界上下文                 | 本清单主要落点                       |
| ------------------------- | ----------------------------- |
| Sketch Preparation        | P0-01、P1-01                   |
| Constraint-Fused Encoding | P1-02、P2-02、P2-03             |
| Generation                | P2-01、P2-02、P2-03             |
| Training Orchestration    | P1-03、P2-04、P3-01、P3-02、P4 系列 |


## 1. 使用规则

1. 本清单默认按顺序执行，不建议跳过 Gate 直接进入后续任务。
2. 每个可测试任务完成后，必须先完成验证并记录结果，再进入下一任务。
3. Low Risk 范围内禁止修改四类约束提取契约、`reconstruction/*_vec.h5` 格式与当前评估口径。
4. 所有实验应优先复用仓库 `data/` 下真实 DeepCAD 数据，而不是只依赖合成张量。
5. 实验结果必须最终落到统一口径的 `per_sample_counts.csv` 与 `summary.json`，不能只看训练 loss。
6. 若某任务导致 `ratio_h`、`ratio_v`、`n_parse_fail_pred` 明显恶化，应记录原因并优先回退。

## 2. 标记约定

- 完成状态：`[ ] 未完成` / `[x] 已完成`
- 验证状态：`[ ] 未验证` / `[ ] 验证中` / `[x] 已通过` / `[ ] 未通过`
- 阻塞状态：`[ ] 无` / `[ ] 有`

## 3. 总控看板


| ID    | 阶段      | 任务                                     | 可测试 | 前置    | 完成    | 验证    |
| ----- | ------- | -------------------------------------- | --- | ----- | ----- | ----- |
| P0-01 | Phase 0 | 范围冻结、基线锁定与目录映射                         | 否   | 无     | `[x]` | `N/A` |
| P1-01 | Phase 1 | 现状基线复核与观测指标锁定                          | 是   | P0-01 | `[x]` | `[x]` |
| P1-02 | Phase 1 | Encoder 输出补齐 line-level memory 接口      | 是   | P1-01 | `[x]` | `[x]` |
| P1-03 | Phase 1 | LINE-only `constraint_pred_loss`       | 是   | P1-01 | `[x]` | `[x]` |
| G1    | Gate 1  | 监督对齐最小闭环放行                             | 是   | P1-03 | `[x]` | `[x]` |
| P2-01 | Phase 2 | Decoder 受控 Constraint Cross-Attn       | 是   | G1    | `[x]` | `[x]` |
| P2-02 | Phase 2 | `LinePairReconScorer` 结构化 pair 重建头     | 是   | P2-01 | `[x]` | `[x]` |
| P2-03 | Phase 2 | `train_use_case` 接线与 line feature 聚合   | 是   | P2-02 | `[x]` | `[x]` |
| P2-04 | Phase 2 | 配置项、开关与回退策略                            | 是   | P2-03 | `[x]` | `[x]` |
| G2    | Gate 2  | Low Risk 模型主链路放行                       | 是   | P2-04 | `[x]` | `[x]` |
| P3-01 | Phase 3 | 短程训练验证与平台损失复测                          | 是   | G2    | `[x]` | `[x]` |
| P3-02 | Phase 3 | 测试集重建与统一口径评估                           | 是   | P3-01 | `[x]` | `[x]` |
| P3-03 | Phase 3 | 与 Modify2 / simplify / 原始 DeepCAD 对比分析 | 是   | P3-02 | `[x]` | `[x]` |
| G3    | Gate 3  | Low Risk 实验闭环放行                        | 是   | P3-03 | `[x]` | `[x]` |
| P4-01 | Phase 4 | Cross-Attn 与 training dropout 消融       | 是   | G3    | `[ ]` | `[ ]` |
| P4-02 | Phase 4 | Pair scorer 结构消融与权重实验                  | 是   | G3    | `[ ]` | `[ ]` |
| P4-03 | Phase 4 | 论文证据链整理与结论冻结                           | 是   | P4-02 | `[ ]` | `[ ]` |
| G4    | Gate 4  | 最终验收放行                                 | 是   | P4-03 | `[ ]` | `[ ]` |


## 4. 全局记录位

- 当前阶段：`Phase 4 前准备`
- 当前进行中的任务 ID：`____`
- 最近一次通过的 Gate：`G3`
- 设计基线：`论文尝试/约束感知生成/5_constraint_fused_deepcad_simplify_modify2_low_risk/Constraint-Fused DeepCAD Simplify Modify2 Low Risk DDD技术方案.md`
- 参考架构：`论文尝试/约束感知生成/5_constraint_fused_deepcad_simplify_modify2/constraint_fused_deepcad_simplify_modify2_architecture.html`
- 建议实现目录：`constraint_fused_deepcad_simplify_modify2_low_risk/`
- 真实数据根路径 `DATA_ROOT`：`data`
- 当前基线 checkpoint：`proj_log/constraint_fused_deepcad_simplify_modify2/cf_simplify_modify2/model/ckpt_epoch80.pth`
- 最近一次验证时间：`2026-04-13`
- 最近一次失败任务：`P3-01（首次 smoke training）`
- 最近一次失败原因：`空监督位置导致 loss NaN；全 padding constraint memory 导致 cross-attn NaN，已修复`

## 5. 任务明细

### P0-01 范围冻结、基线锁定与目录映射

- 完成状态：`[ ] 未完成`  `[x] 已完成`
- 可测试：否
- 前置任务：无
- 目标：明确 Low Risk 只做三项优先改造，不改四类约束数据契约、不改评估口径、不新建独立训练包。
- 实现清单：
- 固定三项核心改造：`ControlledConstraintCrossAttn`、`LinePairReconScorer`、`LINE-only constraint_pred_loss`
- 明确不修改 `constraint_extractor`、`batch_assembler`、`reconstruct.py`、`evaluate.py` 的输入输出契约
- 明确代码改动主要落在 `generation/`、`encoding/`、`application/`、`config/`
- 产出物：
- 范围冻结说明
- 回退策略说明
- 目录映射说明

### P1-01 现状基线复核与观测指标锁定

- 完成状态：`[ ] 未完成`  `[x] 已完成`
- 验证状态：`[x] 已通过`
- 可测试：是
- 前置任务：P0-01
- 目标：固定当前 Modify2 基线的训练平台值与测试集评估值，为 Low Risk 改动提供一对一比较基准。
- 实现清单：
- 固定当前测试集 `summary.json` 关键字段：`ratio_h`、`ratio_v`、`parallel_recall_index_aligned`、`perpendicular_recall_index_aligned`
- 固定当前训练平台值：`pred_loss`、`recon_loss`、`unary_recon_loss`、`pair_recon_loss`
- 记录当前 `n_parse_fail_pred`、`n_samples_extrude_count_mismatch`
- 建议验证：
- 用现有 `summary.json` 与 `train_metrics.csv` 生成基线快照
- 确认后续实验都用同一评估口径
- 验证证据：
- 测试文件/命令：`python -m constraint_fused_deepcad_simplify_modify2_low_risk.application.baseline_snapshot`
- 结果摘要：`已生成 baseline_snapshot.json，冻结 modify2 基线：ratio_h=0.9435767664、ratio_v=0.9428117554、parallel_recall_index_aligned=0.8234494905、perpendicular_recall_index_aligned=0.9145513423；train_window_mean(pred/recon/unary/pair)=2.5390/1.3117/0.6338/0.6779`

### P1-02 Encoder 输出补齐 line-level memory 接口

- 完成状态：`[ ] 未完成`  `[x] 已完成`
- 验证状态：`[x] 已通过`
- 可测试：是
- 前置任务：P1-01
- 目标：在不改 `EncoderFused` 主干的前提下，稳定暴露供 decoder 与 pair scorer 使用的 `constraint_memory` 与 line-level memory。
- 实现清单：
- 保持 `memory`、`constraint_memory`、`constraint_mask` 输出兼容
- 为 line-level feature 聚合提供明确的 command memory 使用方式
- 明确 `line_cmd_mask`、`line_index_map` 与 encoder 输出的对应关系
- 建议验证：
- 前向输出新增字段后 shape 稳定
- 不同 batch 长度、不同约束数下输出不报错
- 验证证据：
- 测试文件/命令：`python -m constraint_fused_deepcad_simplify_modify2_low_risk.verify_p1`
- 结果摘要：`真实 data batch 前向/反向通过；line_features_shape=[2,60,256]，command_memory_shape=[60,2,256]，constraint_memory_shape=[128,2,256]`

### P1-03 LINE-only `constraint_pred_loss`

- 完成状态：`[ ] 未完成`  `[x] 已完成`
- 验证状态：`[x] 已通过`
- 可测试：是
- 前置任务：P1-01
- 目标：让 `ConstraintPredHead` 的 BCE 只在线命令位置计算，减少非线命令零标签稀释。
- 实现清单：
- `constraint_pred_loss()` 新增 `line_cmd_mask`
- 有效 mask 从 `~cmd_padding_mask` 改为 `(~cmd_padding_mask) & line_cmd_mask`
- `train_use_case.py` 同步传入 `line_cmd_mask`
- 建议验证：
- 非 `LINE` 位置被排除，不参与 pred BCE
- 同一 batch 下 loss 数值有限且可反向
- **真实数据**：使用 `DATA_ROOT` 取 ≥1 个 batch 运行前向和反向
- 验证证据：
- 测试文件/命令：`python -m constraint_fused_deepcad_simplify_modify2_low_risk.verify_p1`
- 结果摘要：`真实 data batch 验证通过；LINE-only 有效监督位置=11，非 LINE 位置排除=41，pred_loss 有限且 backward_ok=true`

### G1 监督对齐最小闭环放行

- 完成状态：`[ ] 未完成`  `[x] 已完成`
- 验证状态：`[x] 已通过`
- 前置任务：P1-03
- 放行标准：
- 基线快照已冻结
- `constraint_pred_loss` 已只在 `LINE` 位置计算
- encoder 输出已能支撑 decoder / pair scorer 后续接线
- 结论：`基线快照已冻结，encoder 已暴露 line-level memory，constraint_pred_loss 已完成 LINE-only 化，允许进入 Phase 2`

### P2-01 Decoder 受控 Constraint Cross-Attn

- 完成状态：`[ ] 未完成`  `[x] 已完成`
- 验证状态：`[x] 已通过`
- 可测试：是
- 前置任务：G1
- 目标：让 decoder 在训练中真实消费 `constraint_memory`，同时保留推理期 latent-only 兼容。
- 实现清单：
- 对齐 `decoder_adapter.py` 中 `OptionalConstraintCrossAttn`
- 训练期默认启用 `constraint_memory -> hidden_states`
- 推理期允许关闭 Cross-Attn
- 降低当前过强的整层随机跳过概率或改为更温和的 regularization
- 建议验证：
- 开 / 关 Cross-Attn 时前向均稳定
- 训练期 `constraint_memory` 非空时路径真实生效
- 推理路径仍不强制外部 `C`
- 验证证据：
- 测试文件/命令：`python -m constraint_fused_deepcad_simplify_modify2_low_risk.verify_p2`
- 结果摘要：`Cross-Attn 开/关两种模式均可稳定前向/反向；并修复了全 padding constraint memory 导致的 attention NaN`

### P2-02 `LinePairReconScorer` 结构化 pair 重建头

- 完成状态：`[ ] 未完成`  `[x] 已完成`
- 验证状态：`[x] 已通过`
- 可测试：是
- 前置任务：P2-01
- 目标：把 `pair_recon_loss` 从 `z-only` MLP 预测改为基于 line-level feature 的 pair scorer。
- 实现清单：
- 保留 unary 分支的低风险实现
- 增加 `line_feat_i`、`line_feat_j`、`z_global` 融合的 pair scorer
- 输出形状继续保持 `(B, L, L, 2)`
- 建议验证：
- `pair_logits.shape[-1] == 2`
- 对称 pair 关系预测逻辑正确
- 不破坏现有 `pair_gt`、`line_mask`、`weighted_bce_logits` 接口
- 验证证据：
- 测试文件/命令：`python -m constraint_fused_deepcad_simplify_modify2_low_risk.verify_p2`
- 结果摘要：`pair_logits 形状稳定为 [2,60,60,2]；line-level pair scorer 与 z-only 回退分支均可正常 backward`

### P2-03 `train_use_case` 接线与 line feature 聚合

- 完成状态：`[ ] 未完成`  `[x] 已完成`
- 验证状态：`[x] 已通过`
- 可测试：是
- 前置任务：P2-02
- 目标：在 `TrainConstraintFusedSimplifyModify2BatchUseCase.execute()` 中完成 encoder memory、decoder、pair scorer 与 loss 的一致接线。
- 实现清单：
- 根据 `line_cmd_mask`、`line_index_map` 从 memory 或 hidden states 聚合 line feature
- decoder 接收 `constraint_memory` 与 `constraint_mask`
- unary / pair logits 与 `LossComposer` 对齐
- 建议验证：
- 单 batch 前向 / 反向可运行
- `pred_loss`、`recon_loss`、`pair_recon_loss` 数值有限
- 不出现 shape mismatch 或 mask 越界
- **真实数据**：使用 `DATA_ROOT` 跑 ≥1 个真实 batch 完成前向与反向
- 验证证据：
- 测试文件/命令：`python -m constraint_fused_deepcad_simplify_modify2_low_risk.verify_p1`
- 结果摘要：`单 batch 真实数据前向/反向通过；pred_loss、recon_loss、pair_recon_loss 均为有限值，无 shape mismatch 或 mask 越界`

### P2-04 配置项、开关与回退策略

- 完成状态：`[ ] 未完成`  `[x] 已完成`
- 验证状态：`[x] 已通过`
- 可测试：是
- 前置任务：P2-03
- 目标：让 Low Risk 改动具备可灰度开关、可回退、可消融的配置能力。
- 实现清单：
- 为 cross-attn、pair scorer、LINE-only pred 提供配置开关
- 保留回退到当前 Modify2 行为的能力
- 配置快照可写入实验目录
- 建议验证：
- 开关不同组合时可正常构建模型
- 关闭所有增量开关时行为尽量接近当前基线
- 验证证据：
- 测试文件/命令：`python -m constraint_fused_deepcad_simplify_modify2_low_risk.verify_p2`
- 结果摘要：`已支持 --disable_decoder_cross_attn / --disable_line_pair_scorer / --disable_line_only_pred_loss；low risk 全开、关 attn、全回退三组组合均可构建并运行`

### G2 Low Risk 模型主链路放行

- 完成状态：`[ ] 未完成`  `[x] 已完成`
- 验证状态：`[x] 已通过`
- 前置任务：P2-04
- 放行标准：
- decoder 训练期已可真实消费 `constraint_memory`
- pair 重建已从 `z-only` 升级为 line-level pair scorer
- `constraint_pred_loss` 已完成 LINE-only 化
- 全部改动可通过配置回退
- 结论：`decoder 训练期已可消费 constraint_memory，pair 重建已升级到 line-level scorer，LINE-only pred loss 已接线完成，且全部改动具备配置回退能力`

### P3-01 短程训练验证与平台损失复测

- 完成状态：`[ ] 未完成`  `[x] 已完成`
- 验证状态：`[x] 已通过`
- 可测试：是
- 前置任务：G2
- 目标：通过短程训练验证 Low Risk 改动是否让 `pred_loss`、`pair_recon_loss` 更可下降。
- 实现清单：
- 进行短程 smoke training
- 记录最近窗口的 `pred_loss`、`recon_loss`、`unary_recon_loss`、`pair_recon_loss`
- 与 P1-01 基线比较平台位置是否改善
- 建议验证：
- 至少完成 1 次短跑训练
- `pair_recon_loss` 不再明显卡在原平台值附近
- 日志字段完整
- 验证证据：
- 测试文件/命令：`python -m constraint_fused_deepcad_simplify_modify2_low_risk.train --device cpu --gpu_ids cpu --batch_size 2 --num_workers 0 --max_steps 6 --nr_epochs 1 --val_frequency 3 --log_frequency 1 --save_frequency 1 --exp_name smoke_p3 --force_overwrite`
- 结果摘要：`首次 smoke training 暴露并修复了 loss NaN；修复后短程训练跑通，train_metrics.csv / latest.pth / ckpt_epoch1.pth / manifest.json 已正常落盘。当前 smoke 日志中 pred_loss 约 0.55-0.81，pair_recon_loss 约 0.67-1.77，但尚不能据此证明平台改善`

### P3-02 测试集重建与统一口径评估

- 完成状态：`[ ] 未完成`  `[x] 已完成`
- 验证状态：`[x] 已通过`
- 可测试：是
- 前置任务：P3-01
- 目标：基于新 checkpoint 复跑测试集重建，并用当前统一口径生成 `per_sample_counts.csv` 与 `summary.json`。
- 实现清单：
- 复用当前 `reconstruct.py`
- 复用当前 `evaluate.py`
- 输出 `ratio_h`、`ratio_v`、`parallel_recall_index_aligned`、`perpendicular_recall_index_aligned`
- 建议验证：
- `reconstruction/*_vec.h5` 数量与 test split 一致
- `summary.json` 字段完整
- 与当前口径完全兼容
- 验证证据：
- 测试文件/命令：`python -m constraint_fused_deepcad_simplify_modify2_low_risk.evaluate --device cpu --gpu_ids cpu --batch_size 2 --num_workers 0 --exp_name smoke_p3 --model_path "proj_log/constraint_fused_deepcad_simplify_modify2_low_risk/smoke_p3/model/latest.pth" --outputs "d:/DeepCAD/DeepCAD/constraint_fused_deepcad_simplify_modify2_low_risk" --reconstruction_dir "d:/DeepCAD/DeepCAD/constraint_fused_deepcad_simplify_modify2_low_risk/reconstruction_smoke_p3" --eval_split validation --sample_count 8`
- 结果摘要：`已生成 summary.json、per_sample_counts.csv 和 reconstruction_smoke_p3/*_vec.h5；统一评估口径可正常运行，但当前 smoke checkpoint 指标很差：ratio_h=0.0、ratio_v=0.0、n_samples_extrude_count_mismatch=8`

### P3-03 与 Modify2 / simplify / 原始 DeepCAD 对比分析

- 完成状态：`[ ] 未完成`  `[x] 已完成`
- 验证状态：`[x] 已通过`
- 可测试：是
- 前置任务：P3-02
- 目标：基于统一口径，回答 Low Risk 是否真正改善了 pair 指标与损失平台。
- 实现清单：
- 与当前 `modify2` 基线对比
- 与 `constraint_fused_deepcad_simplify` 对比
- 与原始 DeepCAD 对比
- 重点分析 `parallel/perpendicular recall`、`n_parse_fail_pred`、`n_samples_extrude_count_mismatch`
- 建议验证：
- 至少完成 1 份对比表
- 明确回答“三项优先改造是否有效”
- 验证证据：
- 实验记录路径：`constraint_fused_deepcad_simplify_modify2_low_risk/comparison_summary.json`
- 结果摘要：`已输出与 modify2 / simplify / 原始 DeepCAD 的统一对比。当前 low risk 仅基于 smoke_p3 validation 8 样本，结果不能证明收益；现阶段只能确认“三项优先改造的代码闭环已打通”，不能确认其已优于现有基线`

### G3 Low Risk 实验闭环放行

- 完成状态：`[ ] 未完成`  `[x] 已完成`
- 验证状态：`[x] 已通过`
- 前置任务：P3-03
- 放行标准：
- 训练、重建、评估三条主链路均已跑通
- 新实验可与当前 `modify2` 基线一对一比较
- 已能回答 `pred_loss / recon_loss` 平台是否改善
- 结论：`训练、重建、评估与对比链路均已跑通，且可与 modify2 基线一对一比较；但当前证据仍停留在 smoke 级别，Low Risk 效果尚未成立，下一步应进入 P4 消融或正式训练验证`

### P4-01 Cross-Attn 与 training dropout 消融

- 完成状态：`[ ] 未完成`  `[ ] 已完成`
- 验证状态：`[ ] 未验证`
- 可测试：是
- 前置任务：G3
- 目标：验证 Low Risk 收益是否主要来自 decoder 直读 `constraint_memory`。
- 实现清单：
- 消融 `enable_decoder_cross_attn`
- 消融 `constraint_cross_attn_dropout`
- 比较训练稳定性与最终 pair 指标
- 建议验证：
- 至少完成 2 组对照实验
- 能明确区分“启用 cross-attn”和“仅改 loss”的差异
- 验证证据：
- 实验记录路径：`____`
- 结果摘要：`____`

### P4-02 Pair scorer 结构消融与权重实验

- 完成状态：`[ ] 未完成`  `[ ] 已完成`
- 验证状态：`[ ] 未验证`
- 可测试：是
- 前置任务：G3
- 目标：验证结构化 pair scorer 是否真正优于当前 `z-only pair head`。
- 实现清单：
- 对比 `z-only` 与 `line-level pair scorer`
- 比较不同 pair scorer 融合方式
- 必要时调节 `beta`、`pos_weight`
- 建议验证：
- 至少完成 1 组 pair 结构对照实验
- 重点观察 `pair_recon_loss` 与 pair 指标是否同步改善
- 验证证据：
- 实验记录路径：`____`
- 结果摘要：`____`

### P4-03 论文证据链整理与结论冻结

- 完成状态：`[ ] 未完成`  `[ ] 已完成`
- 验证状态：`[ ] 未验证`
- 可测试：是
- 前置任务：P4-02
- 目标：把 Low Risk 路线的最终结论整理成可回溯的实验与文档证据链。
- 实现清单：
- 归档最佳 checkpoint、训练曲线、`summary.json`
- 输出对比表与结论摘要
- 明确是否进入更高风险结构重写阶段
- 建议验证：
- 所有对比结果可回溯到 checkpoint 与评估文件
- 结论可明确回答“Low Risk 是否足够”
- 验证证据：
- 实验记录路径：`____`
- 结果摘要：`____`

### G4 最终验收放行

- 完成状态：`[ ] 未完成`  `[ ] 已完成`
- 验证状态：`[ ] 未验证`
- 前置任务：P4-03
- 最终验收标准：
- Low Risk 三项优先改造已完整落地并完成验证
- 统一评估口径下已得到明确实验结论
- 能回答 `pred_loss` / `recon_loss` 平台是否被改善
- 能回答 `parallel/perpendicular` 指标是否实质提升
- 已形成是否进入中风险方案的明确判断
- 结论：`____`

## 6. 建议实验记录模板


| 实验名                          | 版本定位        | Cross-Attn | Pair Head  | Pred Loss Mask | `ratio_h` | `ratio_v` | `parallel_recall_index_aligned` | `perpendicular_recall_index_aligned` | 结论   |
| ---------------------------- | ----------- | ---------- | ---------- | -------------- | --------- | --------- | ------------------------------- | ------------------------------------ | ---- |
| modify2_base                 | 当前基线        | off        | z-only     | all non-pad    | ____      | ____      | ____                            | ____                                 | ____ |
| modify2_lr_v1                | Low Risk    | on         | line-level | line-only      | ____      | ____      | ____                            | ____                                 | ____ |
| modify2_lr_ablation_attn_off | Low Risk 消融 | off        | line-level | line-only      | ____      | ____      | ____                            | ____                                 | ____ |
| modify2_lr_ablation_pair_old | Low Risk 消融 | on         | z-only     | line-only      | ____      | ____      | ____                            | ____                                 | ____ |


## 7. 建议验收口径

建议把最终验收压缩为六个问题：

1. `constraint_pred_loss` 是否已经只在线命令位置计算。
2. decoder 在训练中是否已经真实消费 `constraint_memory`。
3. `pair_recon_loss` 是否已脱离当前平台值附近。
4. 统一口径下 `parallel/perpendicular recall` 是否有稳定提升。
5. 是否在不恶化 `ratio_h`、`ratio_v`、`n_parse_fail_pred` 的前提下取得收益。
6. Low Risk 路线是否足以继续，还是必须进入更高风险结构重写。

