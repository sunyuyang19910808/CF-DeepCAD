# CAD-only 消融与辅助损失 Warmup 训练 100 Epoch 任务清单

> 来源文档：`CAD-only消融与辅助损失Warmup训练100epoch方案.md`
>
> 目标：将 CAD-only 消融、辅助损失 warmup、100 epoch 训练和统一评估拆解为可执行、可记录、可验证的任务。任务默认作用于 `constraint_fused_deepcad_simplify_modify2_low_riskbased_high_modify`，实验输出落在 `proj_log/constraint_fused_deepcad_simplify_modify2_low_riskbased_high_modify/`。

## 0. 使用规则

1. 本清单按 Phase 0 到 Phase 5 顺序推进，Gate 未通过前不建议直接进入下一阶段。
2. 每个训练实验必须使用独立 `exp_name`，禁止覆盖当前 `cf_high_modify` 结果。
3. 每个可测试任务必须记录命令、checkpoint、`summary.json`、`reconstruction_*_acc_stat.txt` 和关键指标。
4. 评估不能只看 `latest`，至少评估 `ckpt_epoch25`、`ckpt_epoch50`、`ckpt_epoch100`；若资源允许，增加 `ckpt_epoch75`。
5. 架构红线保持不变：主解码路径只依赖 `z`，辅助损失不得直接作用于 `z`，约束监督服务于最终 CAD 序列重建。
6. 若 warmup 后 `ACC_param` 比 CAD-only 下降超过 2%，必须降低辅助权重或延长 CAD-only 预热。

## 1. 标记约定

- 完成状态：`[ ] 未完成` / `[x] 已完成`
- 验证状态：`[ ] 未验证` / `[ ] 验证中` / `[x] 已通过` / `[ ] 未通过`
- 阻塞状态：`[ ] 无` / `[ ] 有`

## 2. 总控看板

| ID | 阶段 | 任务 | 可测试 | 前置 | 完成 | 验证 |
| --- | --- | --- | --- | --- | --- | --- |
| P0-01 | Phase 0 | 冻结当前 `H0_current` 基线结果 | 是 | 无 | `[x]` | `[x]` |
| P0-02 | Phase 0 | 修复或规避 `train_metrics.csv` 重复写入 | 是 | P0-01 | `[x]` | `[x]` |
| P0-03 | Phase 0 | 建立实验目录命名与结果归档模板 | 否 | P0-01 | `[x]` | `N/A` |
| G0 | Gate 0 | 基线与记录口径放行 | 是 | P0-01~P0-03 | `[x]` | `[x]` |
| P1-01 | Phase 1 | 增加 warmup 调度配置参数 | 是 | G0 | `[x]` | `[x]` |
| P1-02 | Phase 1 | 实现辅助损失权重调度函数 | 是 | P1-01 | `[x]` | `[x]` |
| P1-03 | Phase 1 | 将动态权重接入训练链路 | 是 | P1-02 | `[x]` | `[x]` |
| P1-04 | Phase 1 | 训练日志记录实际 `aux_alpha/beta/gamma` | 是 | P1-03 | `[x]` | `[x]` |
| G1 | Gate 1 | Warmup 训练机制放行 | 是 | P1-01~P1-04 | `[x]` | `[x]` |
| P2-01 | Phase 2 | 启动 `H1_cad_only_100` 训练 | 是 | G1 | `[x]` | `[x]` |
| P2-02 | Phase 2 | 评估 `H1` 的 25/50/100 epoch checkpoint | 是 | P2-01 | `[x]` | `[x]` |
| P2-03 | Phase 2 | 分析 CAD-only 主路径上限 | 是 | P2-02 | `[x]` | `[x]` |
| G2 | Gate 2 | CAD-only 消融结论放行 | 是 | P2-01~P2-03 | `[x]` | `[x]` |
| P3-01 | Phase 3 | 启动 `H2_warmup_mild_100` 训练 | 是 | G2 | `[ ]` | `[ ]` |
| P3-02 | Phase 3 | 评估 `H2` 的 25/50/75/100 epoch checkpoint | 是 | P3-01 | `[ ]` | `[ ]` |
| P3-03 | Phase 3 | 与 `H1` 对比验证 warmup 是否有效 | 是 | P3-02 | `[ ]` | `[ ]` |
| G3 | Gate 3 | 温和 Warmup 结论放行 | 是 | P3-01~P3-03 | `[ ]` | `[ ]` |
| P4-01 | Phase 4 | 启动 `H4_no_geom_100` 几何损失消融 | 是 | G3 | `[ ]` | `[ ]` |
| P4-02 | Phase 4 | 必要时启动 `H3_warmup_strong_100` | 是 | G3 | `[ ]` | `[ ]` |
| P4-03 | Phase 4 | 汇总 H1/H2/H4/H3 横向对比 | 是 | P4-01 | `[ ]` | `[ ]` |
| G4 | Gate 4 | 消融矩阵结论放行 | 是 | P4-01~P4-03 | `[ ]` | `[ ]` |
| P5-01 | Phase 5 | 形成最终结果表与结论摘要 | 是 | G4 | `[ ]` | `[ ]` |
| P5-02 | Phase 5 | 归档配置、命令、日志和评估文件 | 是 | P5-01 | `[ ]` | `[ ]` |
| G5 | Gate 5 | 论文证据链放行 | 是 | P5-01~P5-02 | `[ ]` | `[ ]` |

## 3. 全局记录位

- 当前阶段：`主路径消融定位准备`
- 当前进行中的任务 ID：`主路径消融 A1：仅去掉 constraint_tags`
- 最近一次通过的 Gate：`G2`
- 方案文档：`论文尝试/约束感知生成/6_constraint_fused_deepcad_high_modify/CAD-only消融与辅助损失Warmup训练100epoch方案.md`
- 实现目录：`constraint_fused_deepcad_simplify_modify2_low_riskbased_high_modify/`
- 实验根目录：`proj_log/constraint_fused_deepcad_simplify_modify2_low_riskbased_high_modify/`
- 数据根路径：`data`
- 当前问题基线：`cf_high_modify`
- CAD-only 实验：`cf_high_modify_cad_only_100`
- 温和 warmup 实验：`cf_high_modify_warmup_mild_100`
- 无几何损失实验：`cf_high_modify_no_geom_100`
- 强 warmup 实验：`cf_high_modify_warmup_strong_100`
- 最近一次验证时间：`2026-05-18 21:52 UTC+8`
- 最近一次失败任务：`____`
- 最近一次失败原因：`____`

## 4. 任务明细

### P0-01 冻结当前 `H0_current` 基线结果

- 完成状态：`[ ] 未完成`  `[x] 已完成`
- 验证状态：`[x] 已通过`
- 可测试：是
- 前置任务：无
- 目标：固定当前 `cf_high_modify` 作为问题基线，避免后续训练覆盖或污染对照结果。
- 执行清单：
- 记录当前 `proj_log/constraint_fused_deepcad_simplify_modify2_low_riskbased_high_modify/cf_high_modify/config.txt`
- 记录当前 `artifacts/test_eval_latest_*/summary.json`
- 记录当前 `artifacts/reconstruction_*_acc_stat.txt`
- 记录当前训练停止位置：约 `epoch=25`、`step=64010`
- 记录当前关键指标：`ACC_cmd=0.9671`、`ACC_param=0.8690`、`ratio_h=0.8965`、`ratio_v=0.9148`、`parallel=0.7377`、`perpendicular=0.8595`
- 产出物：
- `H0_current_config.txt` 或配置路径记录
- `H0_current_summary.json` 或评估路径记录
- `H0_current_acc_stat.txt` 或重建精度路径记录
- 验证证据：
- 文件/命令：`proj_log/constraint_fused_deepcad_simplify_modify2_low_riskbased_high_modify/cf_high_modify/config.txt`；`proj_log/constraint_fused_deepcad_simplify_modify2_low_riskbased_high_modify/cf_high_modify/artifacts/test_eval_latest_20260517_0715/summary.json`；`proj_log/constraint_fused_deepcad_simplify_modify2_low_riskbased_high_modify/cf_high_modify/artifacts/reconstruction_test_latest_20260517_0715_acc_stat.txt`
- 结果摘要：冻结 H0 current：`ACC_cmd=0.96712562355405`，`ACC_param=0.8689659342507456`，`ratio_h=0.8965216050968504`，`ratio_v=0.9148407987048031`，`parallel=0.7376895938244429`，`perpendicular=0.8594641628556396`，`n_parse_fail_pred=315`，`n_samples_extrude_count_mismatch=523`。

### P0-02 修复或规避 `train_metrics.csv` 重复写入

- 完成状态：`[ ] 未完成`  `[x] 已完成`
- 验证状态：`[x] 已通过`
- 可测试：是
- 前置任务：P0-01
- 目标：避免后续训练日志每个 `(epoch, step)` 出现两行，保证损失趋势统计准确。
- 执行清单：
- 检查 `train.py` 中 `append_csv_row(metrics_csv, row)` 与 `tracker.append_train_metrics(row)`
- 选择一种处理方式：
- 方式 A：注释或删除 `tracker.append_train_metrics(row)`
- 方式 B：让 `ExperimentTracker.append_train_metrics` 写入独立文件
- 方式 C：暂不改代码，但所有统计脚本按 `(epoch, step)` 去重
- 建议优先方式 A，减少后续分析复杂度
- 建议验证：
- 启动一个短程 `--max_steps 20` debug 训练
- 检查 `train_metrics.csv` 中同一 `(epoch, step)` 是否只出现一次
- 验证证据：
- 测试命令：`python -m constraint_fused_deepcad_simplify_modify2_low_riskbased_high_modify.train --data_root data --proj_dir proj_log/constraint_fused_deepcad_simplify_modify2_low_riskbased_high_modify --exp_name cf_high_modify_warmup_smoke --batch_size 2 --num_workers 0 --nr_epochs 1 --aux_schedule warmup --aux_warmup_start_epoch 10 --aux_warmup_end_epoch 30 --alpha 1.0 --beta 0.5 --gamma 1.0 --log_frequency 1 --val_frequency 100 --max_steps 2 --device cpu -g cpu`
- 结果摘要：已移除 `train.py` 中对同一 `train_metrics.csv` 的 `tracker.append_train_metrics(row)` 二次写入；smoke 实验 CSV 只有 header 与 step 0、step 1 两条记录，未出现重复 `(epoch, step)`。

### P0-03 建立实验目录命名与结果归档模板

- 完成状态：`[ ] 未完成`  `[x] 已完成`
- 可测试：否
- 前置任务：P0-01
- 目标：固定所有实验名、checkpoint、评估输出和结果表字段，避免实验之间相互覆盖。
- 执行清单：
- 确认实验名：
- `cf_high_modify_cad_only_100`
- `cf_high_modify_warmup_mild_100`
- `cf_high_modify_no_geom_100`
- `cf_high_modify_warmup_strong_100`
- 确认每个实验至少保留：
- `config.txt`
- `model/latest.pth`
- `model/ckpt_epoch25.pth`
- `model/ckpt_epoch50.pth`
- `model/ckpt_epoch100.pth`
- `artifacts/train_metrics.csv`
- `artifacts/test_eval_*/summary.json`
- `artifacts/reconstruction_*_acc_stat.txt`
- 产出物：
- 实验记录表模板：`论文尝试/约束感知生成/6_constraint_fused_deepcad_high_modify/CAD-only消融与辅助损失Warmup训练100epoch实验记录模板.md`
- 结果汇总表模板：同上，第 4 节“统一评估结果”

### G0 基线与记录口径放行

- 完成状态：`[ ] 未完成`  `[x] 已完成`
- 验证状态：`[x] 已通过`
- 可测试：是
- 前置任务：P0-01、P0-02、P0-03
- 放行标准：
- 已记录 `H0_current` 当前配置和测试结果
- 已明确日志重复写入处理方式
- 已冻结实验命名与归档路径
- 放行结论：`通过`
- 备注：H0 基线已记录，重复写入已修复并验证，实验记录模板已建立。

### P1-01 增加 warmup 调度配置参数

- 完成状态：`[ ] 未完成`  `[x] 已完成`
- 验证状态：`[x] 已通过`
- 可测试：是
- 前置任务：G0
- 目标：在配置层支持 `constant` 与 `warmup` 两种辅助损失调度方式。
- 实现清单：
- 在 `config/config_constraint_fused_high_modify.py` 增加：

```python
parser.add_argument("--aux_schedule", type=str, default="constant", choices=["constant", "warmup"])
parser.add_argument("--aux_warmup_start_epoch", type=int, default=10)
parser.add_argument("--aux_warmup_end_epoch", type=int, default=30)
```

- 保持默认 `constant`，避免影响旧实验复现
- 确认 `config.txt` 能记录这三个字段
- 建议验证：
- 运行配置解析或短程训练，检查打印配置包含新增字段
- 验证证据：
- 测试命令：`python -m py_compile constraint_fused_deepcad_simplify_modify2_low_riskbased_high_modify/config/config_constraint_fused_high_modify.py constraint_fused_deepcad_simplify_modify2_low_riskbased_high_modify/application/loss_schedule.py constraint_fused_deepcad_simplify_modify2_low_riskbased_high_modify/application/train_use_case.py constraint_fused_deepcad_simplify_modify2_low_riskbased_high_modify/train.py`
- 结果摘要：配置解析输出已包含 `aux_schedule=warmup`、`aux_warmup_start_epoch=10`、`aux_warmup_end_epoch=30`；默认值为 `constant/10/30`。

### P1-02 实现辅助损失权重调度函数

- 完成状态：`[ ] 未完成`  `[x] 已完成`
- 验证状态：`[x] 已通过`
- 可测试：是
- 前置任务：P1-01
- 目标：根据 epoch 输出当前实际 `alpha/beta/gamma`。
- 实现清单：
- 新增 `application/loss_schedule.py`，或在 `train.py` 中新增轻量函数
- 实现 `resolve_aux_weights(cfg, epoch)`
- `constant` 模式直接返回 `cfg.alpha/cfg.beta/cfg.gamma`
- `warmup` 模式按 `aux_warmup_start_epoch` 到 `aux_warmup_end_epoch` 线性增长
- 建议验证：
- 用手工 epoch 输入检查返回值：
- epoch 1：`0, 0, 0`
- epoch 20：约 `0.5, 0.25, 0.5`，以 H2 配置为例
- epoch 31：`1.0, 0.5, 1.0`
- 验证证据：
- 测试文件/命令：`python -c "from types import SimpleNamespace; from constraint_fused_deepcad_simplify_modify2_low_riskbased_high_modify.application.loss_schedule import resolve_aux_weights; cfg=SimpleNamespace(alpha=1.0,beta=0.5,gamma=1.0,aux_schedule='warmup',aux_warmup_start_epoch=10,aux_warmup_end_epoch=30); print(resolve_aux_weights(cfg,1)); print(resolve_aux_weights(cfg,20)); print(resolve_aux_weights(cfg,31))"`
- 结果摘要：输出 `(0.0, 0.0, 0.0)`、`(0.5, 0.25, 0.5)`、`(1.0, 0.5, 1.0)`，符合 H2 warmup 预期。

### P1-03 将动态权重接入训练链路

- 完成状态：`[ ] 未完成`  `[x] 已完成`
- 验证状态：`[x] 已通过`
- 可测试：是
- 前置任务：P1-02
- 目标：让每个 batch 使用当前 epoch 对应的辅助损失权重。
- 实现清单：
- 推荐方式：`execute(batch, aux_weights=None)` 接收当前权重，避免直接覆盖 `cfg` 最终配置
- 备选方式：每个 epoch 开始临时更新 `cfg.alpha/cfg.beta/cfg.gamma`
- 确保 `LossComposer` 使用实际权重，而不是固定初始配置
- 确保 CAD-only 模式下辅助权重为 0，`loss = loss_cmd + loss_args`
- 建议验证：
- 短程运行 `--aux_schedule warmup --max_steps 20`
- 检查日志中实际权重符合预期
- 验证证据：
- 测试命令：同 P0-02 smoke 训练命令。
- 结果摘要：训练日志显示 epoch 1 实际 `aux_alpha=0`、`aux_beta=0`、`aux_gamma=0`，总损失等于主 CAD 损失，不覆盖配置中的最终权重。

### P1-04 训练日志记录实际 `aux_alpha/beta/gamma`

- 完成状态：`[ ] 未完成`  `[x] 已完成`
- 验证状态：`[x] 已通过`
- 可测试：是
- 前置任务：P1-03
- 目标：让 `train_metrics.csv` 可以区分最终配置权重与当前实际训练权重。
- 实现清单：
- 在 `metrics = OrderedDict(...)` 中加入：
- `aux_alpha`
- `aux_beta`
- `aux_gamma`
- 确认 CSV header 自动包含新增字段
- 确认 TensorBoard 可选记录这三个字段
- 建议验证：
- 短程训练后读取 `train_metrics.csv`
- 检查 `aux_alpha/beta/gamma` 随 epoch 或配置正确变化
- 验证证据：
- 测试命令：读取 `proj_log/constraint_fused_deepcad_simplify_modify2_low_riskbased_high_modify/cf_high_modify_warmup_smoke/artifacts/train_metrics.csv`
- 结果摘要：CSV header 已包含 `aux_alpha`、`aux_beta`、`aux_gamma`，step 0 和 step 1 均记录为 `0.0/0.0/0.0`。

### G1 Warmup 训练机制放行

- 完成状态：`[ ] 未完成`  `[x] 已完成`
- 验证状态：`[x] 已通过`
- 可测试：是
- 前置任务：P1-01、P1-02、P1-03、P1-04
- 放行标准：
- `constant` 模式能复现原固定权重行为
- `warmup` 模式能在 epoch 1-10 输出 0 权重
- 训练 CSV 中能看到实际辅助权重
- CAD-only 命令不会触发 soft geometry 额外权重
- 放行结论：`通过`
- 备注：`constant` 模式保持原权重行为；`warmup` 模式通过函数输出与 smoke 训练验证；CSV 可记录实际辅助权重；CAD-only 可通过 `alpha=0 beta=0 gamma=0 --disable_soft_geometry` 运行。

### P2-01 启动 `H1_cad_only_100` 训练

- 完成状态：`[ ] 未完成`  `[x] 已完成`
- 验证状态：`[x] 已通过`
- 可测试：是
- 前置任务：G1
- 目标：训练只包含主 CAD 重建损失的 High Modify 模型，建立主路径上限。
- 推荐命令：

```bat
python -m constraint_fused_deepcad_simplify_modify2_low_riskbased_high_modify.train ^
  --data_root data ^
  --proj_dir proj_log/constraint_fused_deepcad_simplify_modify2_low_riskbased_high_modify ^
  --exp_name cf_high_modify_cad_only_100 ^
  --batch_size 64 ^
  --nr_epochs 100 ^
  --alpha 0 ^
  --beta 0 ^
  --gamma 0 ^
  --disable_soft_geometry ^
  --save_frequency 5 ^
  -g 0
```

- 执行清单：
- 确认不覆盖 `cf_high_modify`
- 确认 `config.txt` 中 `alpha=0`、`beta=0`、`gamma=0`
- 确认 `train_metrics.csv` 中 `loss` 近似等于 `loss_cmd`
- 产出物：
- `cf_high_modify_cad_only_100/model/ckpt_epoch25.pth`
- `cf_high_modify_cad_only_100/model/ckpt_epoch50.pth`
- `cf_high_modify_cad_only_100/model/ckpt_epoch100.pth`
- 验证证据：
- 训练命令：`python -m constraint_fused_deepcad_simplify_modify2_low_riskbased_high_modify.train --data_root data --proj_dir proj_log/constraint_fused_deepcad_simplify_modify2_low_riskbased_high_modify --exp_name cf_high_modify_cad_only_100 --batch_size 64 --nr_epochs 100 --alpha 0 --beta 0 --gamma 0 --disable_soft_geometry --save_frequency 5 -g 0`
- 训练状态：已于 `2026-05-17 08:24 UTC+8` 启动后台训练；启动日志确认 `alpha=0.0`、`beta=0.0`、`gamma=0.0`、`enable_soft_geometry=False`，训练 batch 中 `aux_alpha/beta/gamma=0/0/0`，`geom_loss=0`。
- 阶段训练结果：训练至 `epoch 52` 中途后于 `2026-05-18 20:13 UTC+8` 为评估手动暂停；`latest.pth` 保存点为 `epoch=51, step=128520`。
- 已生成 checkpoint：`ckpt_epoch5/10/15/20/25/30/35/40/45/50.pth` 与 `latest.pth`。

### P2-02 评估 `H1` 的 25/50/100 epoch checkpoint

- 完成状态：`[ ] 未完成`  `[x] 已完成`
- 验证状态：`[x] 已通过`
- 可测试：是
- 前置任务：P2-01
- 目标：观察 CAD-only 主路径随训练轮数的恢复趋势。
- 执行清单：
- 对 `ckpt_epoch25` 运行测试集重建与约束评估
- 对 `ckpt_epoch50` 运行测试集重建与约束评估
- 对 `ckpt_epoch100` 运行测试集重建与约束评估
- 每次记录：
- `ACC_cmd`
- `ACC_param`
- `ratio_h`
- `ratio_v`
- `parallel_recall_index_aligned`
- `perpendicular_recall_index_aligned`
- `n_parse_fail_pred`
- `n_samples_extrude_count_mismatch`
- 推荐评估命令模板：

```bat
python -m constraint_fused_deepcad_simplify_modify2_low_riskbased_high_modify.evaluate ^
  --data_root data ^
  --proj_dir proj_log/constraint_fused_deepcad_simplify_modify2_low_riskbased_high_modify ^
  --exp_name cf_high_modify_cad_only_100 ^
  --ckpt ckpt_epoch100 ^
  --eval_split test
```

- 验证证据：
- 评估范围：已评估 `ckpt_epoch5/10/15/20/25/30/35/40/45/50.pth` 与 `latest.pth`；`ckpt_epoch100.pth` 尚未产生，故本轮以 `ckpt_epoch50` 和 `latest(epoch51)` 作为阶段结论依据。
- 汇总文件：`proj_log/constraint_fused_deepcad_simplify_modify2_low_riskbased_high_modify/cf_high_modify_cad_only_100/artifacts/h1_all_checkpoint_eval_summary.csv`
- `ckpt_epoch25` 结果：`ACC_cmd=0.9732`、`ACC_param=0.9098`、`ratio_h=0.9032`、`ratio_v=0.9118`、`parallel=0.7562`、`perpendicular=0.8807`、`parse_fail=167`、`ext_mismatch=278`。
- `ckpt_epoch50` 结果：`ACC_cmd=0.9787`、`ACC_param=0.9239`、`ratio_h=0.9129`、`ratio_v=0.9155`、`parallel=0.7689`、`perpendicular=0.8824`、`parse_fail=97`、`ext_mismatch=234`。
- `latest(epoch51)` 结果：`ACC_cmd=0.9787`、`ACC_param=0.9243`、`ratio_h=0.9120`、`ratio_v=0.9185`、`parallel=0.7651`、`perpendicular=0.8830`、`parse_fail=134`、`ext_mismatch=179`。
- `ckpt_epoch100` 结果：`未产生；本轮暂停继续训练，转入主路径消融定位。`

### P2-03 分析 CAD-only 主路径上限

- 完成状态：`[ ] 未完成`  `[x] 已完成`
- 验证状态：`[x] 已通过`
- 可测试：是
- 前置任务：P2-02
- 目标：判断性能下降是否主要来自辅助损失，而不是主架构。
- 判定规则：
- 若 `ACC_param` 明显高于 `H0_current=0.8690`，说明主路径可训练，进入 warmup 阶段
- 若 `ACC_param` 仍低于或接近 `0.87~0.90`，优先检查 `dim_z`、pooling、batch size、训练轮数
- 若 `ACC_cmd` 恢复但 `ACC_param` 仍低，重点检查参数通道和 `loss_args`
- 产出物：
- CAD-only 结论摘要
- 是否进入 Phase 3 的建议
- 验证证据：
- 结果摘要：H1 CAD-only 相比 H0 current 明显恢复，`ACC_param` 从 H0 `0.8690` 提升至 latest `0.9243`，`ACC_cmd` 从 H0 `0.9671` 提升至 latest `0.9787`，说明关闭辅助损失和 soft geometry 后主 CAD 重建显著改善。但与原始 DeepCAD epoch 27 的 `ACC_param=0.9410` 仍有约 `0.0167` 差距；从 epoch 30 到 latest 的 `ACC_param` 仅由 `0.9142` 提升至 `0.9243`，后期提升偏慢。结论：辅助损失不是唯一瓶颈，主路径结构仍需消融定位。
- Phase 3 建议：暂缓 H2 warmup；优先进入“主路径消融 A1：仅去掉 constraint_tags”。若 A1 无明显改善，再评估不拼 constraint token、masked_mean pooling、dim_z=256 等消融。

### G2 CAD-only 消融结论放行

- 完成状态：`[ ] 未完成`  `[x] 已完成`
- 验证状态：`[x] 已通过`
- 可测试：是
- 前置任务：P2-01、P2-02、P2-03
- 放行标准：
- 已完成 H1 至少 `ckpt_epoch50` 评估
- 已判断主路径是否明显恢复
- 已决定是否继续跑 H2 warmup
- 放行结论：`通过`
- 备注：H1 已完成至 `ckpt_epoch50` 与 `latest(epoch51)` 阶段评估；主路径较 H0 明显恢复，但仍显著落后原始 DeepCAD，且后期提升变慢。因此 G2 放行结论为：不直接进入 H2 warmup，优先转入主路径消融定位（A1 仅去掉 constraint_tags）。

### P3-01 启动 `H2_warmup_mild_100` 训练

- 完成状态：`[ ] 未完成`  `[ ] 已完成`
- 验证状态：`[ ] 未验证`
- 可测试：是
- 前置任务：G2
- 目标：验证温和辅助损失 warmup 是否能在保护 `ACC_param` 的同时提升约束指标。
- 推荐命令：

```bat
python -m constraint_fused_deepcad_simplify_modify2_low_riskbased_high_modify.train ^
  --data_root data ^
  --proj_dir proj_log/constraint_fused_deepcad_simplify_modify2_low_riskbased_high_modify ^
  --exp_name cf_high_modify_warmup_mild_100 ^
  --batch_size 64 ^
  --nr_epochs 100 ^
  --aux_schedule warmup ^
  --aux_warmup_start_epoch 10 ^
  --aux_warmup_end_epoch 30 ^
  --alpha 1.0 ^
  --beta 0.5 ^
  --gamma 1.0 ^
  --save_frequency 5 ^
  -g 0
```

- 执行清单：
- 确认 epoch 1-10 实际 `aux_alpha/beta/gamma=0`
- 确认 epoch 11-30 权重线性增长
- 确认 epoch 31-100 使用 `1.0/0.5/1.0`
- 产出物：
- `cf_high_modify_warmup_mild_100/model/ckpt_epoch*.pth`
- `cf_high_modify_warmup_mild_100/artifacts/train_metrics.csv`
- 验证证据：
- 训练命令：`____`
- 训练状态：`____`

### P3-02 评估 `H2` 的 25/50/75/100 epoch checkpoint

- 完成状态：`[ ] 未完成`  `[ ] 已完成`
- 验证状态：`[ ] 未验证`
- 可测试：是
- 前置任务：P3-01
- 目标：确认 warmup 后序列精度和约束保持的变化趋势。
- 执行清单：
- 评估 `ckpt_epoch25`
- 评估 `ckpt_epoch50`
- 评估 `ckpt_epoch75`
- 评估 `ckpt_epoch100`
- 对每个 checkpoint 记录序列重建、约束保持和解析稳定性指标
- 验证证据：
- `ckpt_epoch25` 结果：`____`
- `ckpt_epoch50` 结果：`____`
- `ckpt_epoch75` 结果：`____`
- `ckpt_epoch100` 结果：`____`

### P3-03 与 `H1` 对比验证 warmup 是否有效

- 完成状态：`[ ] 未完成`  `[ ] 已完成`
- 验证状态：`[ ] 未验证`
- 可测试：是
- 前置任务：P3-02
- 目标：判断温和 warmup 是否值得保留。
- 判定规则：
- `ACC_param` 比 H1 下降小于 1%，且至少一个约束指标明显提升：保留 H2
- `ACC_param` 比 H1 下降 1% 到 2%：谨慎保留，考虑降低 `gamma`
- `ACC_param` 比 H1 下降超过 2%：不通过，降低辅助权重或延长 CAD-only 预热
- `geom_loss` 下降但测试约束不升：进入 H4 no-geom
- 产出物：
- H1 vs H2 对比表
- H2 是否进入最终候选的结论
- 验证证据：
- 结果摘要：`____`

### G3 温和 Warmup 结论放行

- 完成状态：`[ ] 未完成`  `[ ] 已完成`
- 验证状态：`[ ] 未验证`
- 可测试：是
- 前置任务：P3-01、P3-02、P3-03
- 放行标准：
- 已完成 H2 至少 `ckpt_epoch50` 评估
- 已与 H1 对比 `ACC_param` 和四类约束指标
- 已决定是否启动 H4 或 H3
- 放行结论：`通过 / 不通过`
- 备注：`____`

### P4-01 启动 `H4_no_geom_100` 几何损失消融

- 完成状态：`[ ] 未完成`  `[ ] 已完成`
- 验证状态：`[ ] 未验证`
- 可测试：是
- 前置任务：G3
- 目标：判断 `geom_loss` 是否拖慢参数重建或造成代理目标失真。
- 推荐命令：

```bat
python -m constraint_fused_deepcad_simplify_modify2_low_riskbased_high_modify.train ^
  --data_root data ^
  --proj_dir proj_log/constraint_fused_deepcad_simplify_modify2_low_riskbased_high_modify ^
  --exp_name cf_high_modify_no_geom_100 ^
  --batch_size 64 ^
  --nr_epochs 100 ^
  --aux_schedule warmup ^
  --aux_warmup_start_epoch 10 ^
  --aux_warmup_end_epoch 30 ^
  --alpha 1.0 ^
  --beta 0.5 ^
  --gamma 0 ^
  --disable_soft_geometry ^
  --save_frequency 5 ^
  -g 0
```

- 执行清单：
- 训练 H4 到 100 epoch
- 评估 `ckpt_epoch50` 与 `ckpt_epoch100`
- 与 H2 对比 `ACC_param` 和四类约束指标
- 判定规则：
- H4 的 `ACC_param` 更高且约束指标不差：说明 `geom_loss` 可能偏强
- H4 的约束指标明显低于 H2：说明 soft geometry 仍有价值
- 验证证据：
- 训练命令：`____`
- 结果摘要：`____`

### P4-02 必要时启动 `H3_warmup_strong_100`

- 完成状态：`[ ] 未完成`  `[ ] 已完成`
- 验证状态：`[ ] 未验证`
- 可测试：是
- 前置任务：G3
- 目标：当 H2 约束指标提升不足且 `ACC_param` 仍稳定时，测试更强辅助监督。
- 启动条件：
- H2 的 `ACC_param` 比 H1 下降小于 1%
- H2 的约束指标提升不足
- GPU 时间允许
- 推荐命令：

```bat
python -m constraint_fused_deepcad_simplify_modify2_low_riskbased_high_modify.train ^
  --data_root data ^
  --proj_dir proj_log/constraint_fused_deepcad_simplify_modify2_low_riskbased_high_modify ^
  --exp_name cf_high_modify_warmup_strong_100 ^
  --batch_size 64 ^
  --nr_epochs 100 ^
  --aux_schedule warmup ^
  --aux_warmup_start_epoch 10 ^
  --aux_warmup_end_epoch 40 ^
  --alpha 2.0 ^
  --beta 0.7 ^
  --gamma 2.0 ^
  --save_frequency 5 ^
  -g 0
```

- 判定规则：
- 若 H3 提升约束指标但 `ACC_param` 下降超过 2%，不采用
- 若 H3 同时保持 `ACC_param` 并提升约束指标，可作为后续长训候选
- 验证证据：
- 训练命令：`____`
- 结果摘要：`____`

### P4-03 汇总 H1/H2/H4/H3 横向对比

- 完成状态：`[ ] 未完成`  `[ ] 已完成`
- 验证状态：`[ ] 未验证`
- 可测试：是
- 前置任务：P4-01，P4-02 可选
- 目标：形成消融矩阵，判断最终推荐训练策略。
- 结果表模板：

| 实验 | epoch | `ACC_cmd` | `ACC_param` | `ratio_h` | `ratio_v` | `parallel` | `perpendicular` | `parse_fail` | `ext_mismatch` | 结论 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 原始 DeepCAD | 1000 | 0.9936 | 0.9759 | 0.9510 | 0.9574 | 0.8617 | 0.9279 | 29 | 64 | 参考基线 |
| H0 current | 25 | 0.9671 | 0.8690 | 0.8965 | 0.9148 | 0.7377 | 0.8595 | 315 | 523 | 问题基线 |
| H1 CAD-only | 100 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 主路径上限 |
| H2 warmup mild | 100 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 首选候选 |
| H4 no geom | 100 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 几何损失消融 |
| H3 warmup strong | 100 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 可选增强 |

- 产出物：
- 横向对比表
- 推荐保留的训练策略
- 是否继续长训到 200/500/1000 epoch 的建议
- 验证证据：
- 结果摘要：`____`

### G4 消融矩阵结论放行

- 完成状态：`[ ] 未完成`  `[ ] 已完成`
- 验证状态：`[ ] 未验证`
- 可测试：是
- 前置任务：P4-01、P4-03，P4-02 可选
- 放行标准：
- 已完成 H1/H2/H4 至少三组 100 epoch 或中止原因明确
- 已形成统一指标表
- 已判断 `geom_loss` 是否保留
- 已判断温和或强 warmup 是否可作为后续主配置
- 放行结论：`通过 / 不通过`
- 备注：`____`

### P5-01 形成最终结果表与结论摘要

- 完成状态：`[ ] 未完成`  `[ ] 已完成`
- 验证状态：`[ ] 未验证`
- 可测试：是
- 前置任务：G4
- 目标：将实验结论转化为论文或技术报告可直接引用的证据。
- 执行清单：
- 总结当前问题是否主要来自训练不足
- 总结 CAD-only 是否恢复主任务
- 总结 warmup 是否改善约束指标
- 总结 `geom_loss` 是否保留
- 写出最终推荐配置
- 写出失败实验和原因，避免后续重复尝试
- 产出物：
- 最终实验结果表
- 结论摘要
- 下一阶段训练建议
- 验证证据：
- 结果摘要：`____`

### P5-02 归档配置、命令、日志和评估文件

- 完成状态：`[ ] 未完成`  `[ ] 已完成`
- 验证状态：`[ ] 未验证`
- 可测试：是
- 前置任务：P5-01
- 目标：保证实验可复现、可审查、可写入论文附录或实验记录。
- 归档清单：
- 每个实验的 `config.txt`
- 每个实验的训练命令
- 每个实验的 `train_metrics.csv`
- 每个实验的 `summary.json`
- 每个实验的 `reconstruction_*_acc_stat.txt`
- 每个实验的最佳 checkpoint 路径
- 最终对比表
- 建议验证：
- 随机抽查一个实验，确认从记录命令可以复现对应 `config.txt`
- 确认所有结果表中的数值都能追溯到原始文件
- 验证证据：
- 归档路径：`____`
- 结果摘要：`____`

### G5 论文证据链放行

- 完成状态：`[ ] 未完成`  `[ ] 已完成`
- 验证状态：`[ ] 未验证`
- 可测试：是
- 前置任务：P5-01、P5-02
- 放行标准：
- 已完成最终对比表
- 已归档所有配置、命令、日志和评估结果
- 已明确最终推荐训练策略
- 已明确下一步是否继续长训
- 放行结论：`通过 / 不通过`
- 备注：`____`

## 5. 快速执行优先级

若时间有限，优先执行以下最小闭环：

1. P0-01：冻结当前 H0 结果。
2. P0-02：处理训练日志重复写入。
3. P1-01 到 P1-04：实现 warmup 调度和日志记录。
4. P2-01 到 P2-03：跑 H1 CAD-only，至少评估 50 epoch。
5. P3-01 到 P3-03：跑 H2 warmup mild，至少评估 50 epoch。
6. P4-01：若 H2 参数精度下降，跑 H4 no-geom。
7. P5-01：形成阶段性结论。

## 6. 当前首轮推荐

首轮不建议同时展开所有实验。推荐顺序为：

1. 先跑 `H1_cad_only_100`，它决定主路径上限。
2. 若 H1 明显恢复，再跑 `H2_warmup_mild_100`。
3. 若 H2 的 `ACC_param` 明显下降或 `geom_loss` 与测试约束指标不一致，再跑 `H4_no_geom_100`。
4. 只有当 H2 稳定但约束指标提升不足时，才跑 `H3_warmup_strong_100`。

最终判断标准：**温和辅助损失 warmup 必须在不显著牺牲 `ACC_param` 的前提下提升约束保持指标，否则应回退到更弱的辅助权重或更长的 CAD-only 预热。**
