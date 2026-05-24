# Constraint-Fused DeepCAD Simplify Modify1 任务清单

> 来源文档：`Constraint-Fused DeepCAD Simplify Modify1 DDD技术方案.md`
>
> 目标：围绕 `constraint_fused_deepcad_simplify/` 建立一个只考虑 **Horizontal / Vertical** 约束的 Modify1 增强版，在保留 simplify 轻量 encoder 融合的前提下，补回 `ConstraintPredHead`、`DifferentiableSketchInterpreter`、`ConstraintEvaluator` 三条新增监督路径。

---

## 1. 任务使用说明

1. 本清单默认按 Phase 顺序执行。
2. 只有当前任务验证通过，才建议进入下一任务。
3. 若某任务失败，应优先记录阻塞原因与修复动作，不建议直接跳到后续调参项。
4. Modify1 仍然只做 H/V，不做 pair 约束、token 联合序列、decoder `Cross-Attn(C)` 与 Latent GAN。
5. 所有实现默认仍放在 `constraint_fused_deepcad_simplify/`，本目录仅存放设计与任务文档。

---

## 2. 标记约定

- 完成状态：`[ ] 未完成` / `[x] 已完成`
- 验证状态：`[ ] 未验证` / `[ ] 验证中` / `[x] 已通过` / `[ ] 未通过`
- 阻塞状态：`[ ] 无` / `[ ] 有`

---

## 3. 总控看板

| ID | 阶段 | 任务 | 可测试 | 前置 | 完成 | 验证 |
| --- | --- | --- | --- | --- | --- | --- |
| P0-01 | Phase 0 | Modify1 范围冻结与入口确认 | 否 | 无 | `[x]` | `N/A` |
| P1-01 | Phase 1 | 监督目标与 batch 契约扩展 | 是 | P0-01 | `[x]` | `[x]` |
| P1-02 | Phase 1 | `ConstraintPredHead` 目标张量定义 | 是 | P1-01 | `[x]` | `[x]` |
| P1-03 | Phase 1 | `line_cmd_mask` 与 geometry 输入准备 | 是 | P1-02 | `[x]` | `[x]` |
| G1 | Gate 1 | Modify1 数据闭环放行 | 是 | P1-03 | `[x]` | `[x]` |
| P2-01 | Phase 2 | `ConstraintPredHead` 与 decoder adapter 改造 | 是 | G1 | `[x]` | `[x]` |
| P2-02 | Phase 2 | `L_constraint_pred` 接入与 loss 扩展 | 是 | P2-01 | `[x]` | `[x]` |
| P2-03 | Phase 2 | `DifferentiableSketchInterpreter` | 是 | P2-02 | `[x]` | `[x]` |
| P2-04 | Phase 2 | `ConstraintEvaluator` 与 `L_geom_constraint` | 是 | P2-03 | `[x]` | `[x]` |
| P2-05 | Phase 2 | Modify1 单 batch 训练用例 | 是 | P2-04 | `[x]` | `[x]` |
| G2 | Gate 2 | 新监督链路放行 | 是 | P2-05 | `[x]` | `[x]` |
| P3-01 | Phase 3 | Modify1 训练入口与日志对接 | 是 | G2 | `[x]` | `[x]` |
| P3-02 | Phase 3 | Modify1 评估口径与 `R_h/R_v` 对齐 | 是 | P3-01 | `[x]` | `[x]` |
| P3-03 | Phase 3 | Simplify vs Modify1 对比实验 | 是 | P3-02 | `[ ]` | `[ ]` |
| P3-04 | Phase 3 | `alpha/beta/gamma` 消融实验 | 是 | P3-03 | `[ ]` | `[ ]` |
| P3-05 | Phase 3 | 配置、checkpoint、复现记录固化 | 是 | P3-04 | `[ ]` | `[ ]` |
| G3 | Gate 3 | Modify1 验证结论输出 | 是 | P3-05 | `[ ]` | `[ ]` |

---

## 4. 全局记录位

- 当前阶段：`Phase 3`
- 当前进行中的任务：`P3-03`
- 最近一次通过的 Gate：`G2`
- 数据根路径 `DATA_ROOT`：`data`
- 训练配置文件：`constraint_fused_deepcad_simplify_modify1/config/config_constraint_fused_simplify_modify1.py`
- Modify1 设计基线：`论文尝试/约束感知生成/4_constraint_fused_deepcad_simplify_modify1/Constraint-Fused DeepCAD Simplify Modify1 DDD技术方案.md`
- 最近一次验证时间：`2026-04-11`
- 最近一次失败任务：`____`
- 最近一次失败原因：`____`

> 执行口径说明：本清单采用严格串行验收制。对“可测试=是”的任务，必须先完成实现并验证通过，才能进入下一个任务。即使后续代码已提前存在，也不在本清单中记为已完成。

---

## 5. 任务明细

### P0-01 Modify1 范围冻结与入口确认

- 完成状态：`[x] 已完成`  `[ ] 未完成`
- 可测试：否
- 前置任务：无
- 目标：确认 Modify1 是对 simplify 的增量增强，而不是回到完整 Fused。
- 实现清单：
- 冻结范围：只做 `Horizontal`、`Vertical`
- 冻结范围：不做 `pair_gt` / `pair_pred`
- 冻结范围：不做 `ConstraintTokenEncoder` / `E_joint`
- 冻结范围：不做 decoder `Cross-Attn(C)`
- 确认新增模块落点：
- `generation/constraint_pred_head.py` 或并入 `generation/decoder_adapter.py`
- `application/differentiable_sketch_interpreter.py`
- `application/geometry_constraint.py`
- 产出物：
- 范围说明
- 目录落点说明
- Modify1 训练链路总览
- 备注：`已建立独立包 constraint_fused_deepcad_simplify_modify1，并冻结为 H/V-only；未引入 pair、token joint、Cross-Attn。`

### P1-01 监督目标与 batch 契约扩展

- 完成状态：`[x] 已完成`  `[ ] 未完成`
- 验证状态：`[x] 已通过`  `[ ] 未验证`
- 可测试：是
- 前置任务：P0-01
- 目标：在不破坏原 simplify 数据闭环的前提下，为 Modify1 增加新增监督所需字段。
- 实现清单：
- 复用现有 `constraint_tags`
- 复用现有 `unary_gt`
- 明确 decoder 侧监督仍使用 H/V 二维目标
- 在 dataset / collate 中补充 geometry 路径所需字段约定
- 建议验证：
- 检查现有 batch 字段不回归
- 检查新增字段在 CPU 下可被 DataLoader 正常拼接
- 至少打印 1 个真实 batch 的字段字典
- 验证证据：
- 测试文件/命令：`python -m unittest tests.test_constraint_fused_deepcad_simplify_modify1`；`python -c "from constraint_fused_deepcad_simplify_modify1.config.config_constraint_fused_simplify_modify1 import ConfigConstraintFusedSimplifyModify1; from constraint_fused_deepcad_simplify_modify1.infrastructure.dataset_simplify_modify1 import get_simplify_modify1_dataloader; cfg=ConfigConstraintFusedSimplifyModify1('train'); cfg.batch_size=1; cfg.num_workers=0; loader=get_simplify_modify1_dataloader('train', cfg, shuffle=False); batch=next(iter(loader)); print(sorted(batch.keys())); print(tuple(batch['command'].shape), tuple(batch['constraint_tags'].shape), tuple(batch['unary_gt'].shape), tuple(batch['line_cmd_mask'].shape), tuple(batch['line_index_map'].shape))"` 
- 结果摘要：`8/8 单元测试通过；真实 DataLoader batch 成功输出 command/constraint_tags/unary_gt/line_cmd_mask/line_index_map，新增字段可在 CPU 下正常拼接，原有 batch 主字段未回归。`
- 阻塞原因/修复记录：`无`

### P1-02 `ConstraintPredHead` 目标张量定义

- 完成状态：`[x] 已完成`  `[ ] 未完成`
- 验证状态：`[x] 已通过`  `[ ] 未验证`
- 可测试：是
- 前置任务：P1-01
- 目标：把 decoder 侧 H/V 监督目标定义清楚，避免与 latent 侧 `unary_gt` 语义混淆。
- 实现清单：
- 明确 `constraint_pred_logits` 目标维度为 `(S, N, 2)`
- 定义命令级监督与 `constraint_tags` 的对齐关系
- 约定非 `LINE` 命令的监督行为
- 明确 padding 位置不参与 `L_constraint_pred`
- 建议验证：
- 构造最小样本，确认 `LINE` 位置得到正确 H/V 标签
- 检查非 `LINE` 命令是否不会产生错误监督
- 检查 `cmd_padding_mask` 是否正确屏蔽 loss
- 验证证据：
- 测试文件/命令：`python -m unittest tests.test_constraint_fused_deepcad_simplify_modify1`
- 结果摘要：`最小样本验证通过：constraint_tags 在 LINE 位置生成二维 H/V 标签，非 LINE 位置保持 0；constraint_pred_loss 能正确使用 cmd_padding_mask 屏蔽 padding 位置，loss 数值有限。`

### P1-03 `line_cmd_mask` 与 geometry 输入准备

- 完成状态：`[x] 已完成`  `[ ] 未完成`
- 验证状态：`[x] 已通过`  `[ ] 未验证`
- 可测试：是
- 前置任务：P1-02
- 目标：为 `DifferentiableSketchInterpreter` 提供 teacher-forced 几何入口。
- 实现清单：
- 生成 `line_cmd_mask`
- 约定 line 位置与 `args_logits` 的对齐方式
- 明确 `LINE` 参数中 `x1/y1/x2/y2` 或等价坐标语义
- 如有必要，补充 `line_ref` 或 line 顺序索引映射
- 建议验证：
- 检查 `line_cmd_mask` 仅在 `LINE` 命令位置为真
- 至少在 1 个真实 batch 上打印 line 命令位置与对应参数
- 检查 geometry 输入 shape 不越界
- 验证证据：
- 测试文件/命令：`python -m unittest tests.test_constraint_fused_deepcad_simplify_modify1`；`python -c "from constraint_fused_deepcad_simplify_modify1.config.config_constraint_fused_simplify_modify1 import ConfigConstraintFusedSimplifyModify1; from constraint_fused_deepcad_simplify_modify1.infrastructure.dataset_simplify_modify1 import get_simplify_modify1_dataloader; cfg=ConfigConstraintFusedSimplifyModify1('train'); cfg.batch_size=1; cfg.num_workers=0; loader=get_simplify_modify1_dataloader('train', cfg, shuffle=False); batch=next(iter(loader)); print('line_positions', batch['line_cmd_mask'][0].nonzero().flatten().tolist()[:10]); print('line_index_map', batch['line_index_map'][0][batch['line_cmd_mask'][0]].tolist()[:10]); print('args_at_lines', batch['args'][0][batch['line_cmd_mask'][0]][:, :4])"` 
- 结果摘要：`line_cmd_mask 仅在真实 LINE 命令位置为真；line_index_map 与 LINE 位置一一对齐；真实 batch 中 geometry 入口参数已可按 LINE 位置提取，shape 未越界。`

### G1 Modify1 数据闭环放行

- 完成状态：`[x] 已完成`  `[ ] 未完成`
- 验证状态：`[x] 已通过`  `[ ] 未验证`
- 前置任务：P1-03
- 放行标准：
- `constraint_tags`、`unary_gt`、`line_cmd_mask` 均可从真实 batch 中稳定得到
- decoder 侧监督目标与 geometry 输入约定清晰
- 不引入任何 pair 结构字段
- 结论：`已放行。constraint_tags、unary_gt、line_cmd_mask 可从真实 batch 稳定得到；decoder 侧 H/V 二维监督与 geometry 输入约定已落地；未引入任何 pair 结构字段。`

### P2-01 `ConstraintPredHead` 与 decoder adapter 改造

- 完成状态：`[x] 已完成`  `[ ] 未完成`
- 验证状态：`[x] 已通过`  `[ ] 未验证`
- 可测试：是
- 前置任务：G1
- 目标：在现有 simplify decoder 基础上，增加 decoder hidden 的 H/V 预测分支。
- 实现清单：
- 实现 `ConstraintPredHead(d_model -> 2)`
- 修改 `decoder_adapter.py` 输出 `hidden_states`
- 输出 `constraint_pred_logits`
- 保持原 `command_logits` / `args_logits` 接口兼容
- 建议验证：
- 前向时 `constraint_pred_logits` shape 为 `(S, N, 2)`
- 不启用 geometry loss 时主 decoder 前向不回归
- 不出现 NaN / Inf
- 验证证据：
- 测试文件/命令：`python -m unittest tests.test_constraint_fused_deepcad_simplify_modify1`
- 结果摘要：`decoder adapter 前向验证通过：constraint_pred_logits 输出 shape 为 batch-first `(N, S, 2)`，同时保留 command_logits/args_logits/hidden_states 接口；单测与后续 batch use case 中均未出现 NaN/Inf。`

### P2-02 `L_constraint_pred` 接入与 loss 扩展

- 完成状态：`[x] 已完成`  `[ ] 未完成`
- 验证状态：`[x] 已通过`  `[ ] 未验证`
- 可测试：是
- 前置任务：P2-01
- 目标：把 decoder 侧监督接入总损失。
- 实现清单：
- 在 `loss_composer.py` 中增加 `constraint_pred_loss`
- 定义 `L_total = L_cmd + α·L_constraint_pred + β·L_axis_recon + ...`
- 保持原有 `line_mask` 与 `axis_loss` 行为兼容
- 支持 `alpha=0` 的关闭式消融
- 建议验证：
- 空 padding 下 loss 不报错
- `alpha=0` 时行为回退到原 simplify 主链路
- `alpha>0` 时 loss 数值正常
- 验证证据：
- 测试文件/命令：`python -m unittest tests.test_constraint_fused_deepcad_simplify_modify1`
- 结果摘要：`constraint_pred_loss 与 compose_loss 验证通过：padding 屏蔽有效，alpha>0 时总损失数值有限；四项损失已能共同组合，不破坏原有 axis_loss/line_mask 行为。`

### P2-03 `DifferentiableSketchInterpreter`

- 完成状态：`[x] 已完成`  `[ ] 未完成`
- 验证状态：`[x] 已通过`  `[ ] 未验证`
- 可测试：是
- 前置任务：P2-02
- 目标：对 `args_logits` 做 soft dequantization，恢复连续线段几何。
- 实现清单：
- 实现 `soft_dequantize`
- 输出 `start`、`end`、`dir`、`unit`
- 使用 `line_cmd_mask` 过滤非 `LINE` 命令
- 明确第一阶段只解释 H/V 所需的 line 几何
- 建议验证：
- shape 与 dtype 正确
- 单元向量无 NaN
- 非 `LINE` 位置不会产生错误几何监督
- 验证证据：
- 测试文件/命令：`python -m unittest tests.test_constraint_fused_deepcad_simplify_modify1`
- 结果摘要：`DifferentiableSketchInterpreter 验证通过：soft_dequantize 后可输出 start/end/unit/valid，shape 与 dtype 正确，非 LINE 位置通过 line_cmd_mask 与 line_index_map 被过滤，单元向量无 NaN。`
- 阻塞原因/修复记录：`若 DeepCAD 参数语义与 x1/y1/x2/y2 不一致，需要记录对齐方案。`

### P2-04 `ConstraintEvaluator` 与 `L_geom_constraint`

- 完成状态：`[x] 已完成`  `[ ] 未完成`
- 验证状态：`[x] 已通过`  `[ ] 未验证`
- 可测试：是
- 前置任务：P2-03
- 目标：对 soft line 几何计算 H/V 连续残差，并接入总损失。
- 实现清单：
- 实现 `horizontal_residual = u_y^2`
- 实现 `vertical_residual = u_x^2`
- 使用 `unary_gt` 和 `valid` 做 mask
- 接入 `γ·L_geom_constraint`
- 支持 `gamma=0` 的关闭式消融
- 建议验证：
- 人工构造接近水平/竖直的 unit vector 检查残差数值
- 检查无约束样本不报错
- 检查 `gamma=0` 时训练可退化为无 geometry loss 版本
- 验证证据：
- 测试文件/命令：`python -m unittest tests.test_constraint_fused_deepcad_simplify_modify1`
- 结果摘要：`ConstraintEvaluator 验证通过：人工构造水平/竖直 unit vector 时残差为 0；geom loss 在 use case 中数值有限，并支持作为总损失的一部分稳定返回。`

### P2-05 Modify1 单 batch 训练用例

- 完成状态：`[x] 已完成`  `[ ] 未完成`
- 验证状态：`[x] 已通过`  `[ ] 未验证`
- 可测试：是
- 前置任务：P2-04
- 目标：把 encoder、decoder、`ConstraintPredHead`、`AxisReconHead`、Interpreter、Evaluator 和总损失串成可训练的单 batch。
- 实现清单：
- 实现 `TrainConstraintFusedSimplifyModify1BatchUseCase`
- 输出 `loss`、`loss_cmd`、`axis_loss`、`pred_loss`、`geom_loss`
- 保持与原 train 入口兼容
- 支持 CPU smoke test
- 建议验证：
- 至少跑通 1 个 batch 的前向和反向传播
- `loss_cmd`、`axis_loss`、`pred_loss`、`geom_loss` 全部可记录
- 不出现 NaN / Inf
- 验证证据：
- 测试文件/命令：`python -m unittest tests.test_constraint_fused_deepcad_simplify_modify1`；`python -m constraint_fused_deepcad_simplify_modify1.train --device cpu --gpu_ids cpu --batch_size 1 --num_workers 0 --nr_epochs 1 --max_steps 1 --proj_dir proj_log/constraint_fused_deepcad_simplify_modify1 --exp_name cf_simplify_modify1_smoke --force_overwrite`
- 结果摘要：`单 batch 训练 smoke test 通过：前向/反向可执行，loss_cmd、axis_loss、pred_loss、geom_loss 全部输出并可记录，CPU 下未出现 NaN/Inf。`

### G2 新监督链路放行

- 完成状态：`[x] 已完成`  `[ ] 未完成`
- 验证状态：`[x] 已通过`  `[ ] 未验证`
- 前置任务：P2-05
- 放行标准：
- `ConstraintPredHead` 路径可用
- `DifferentiableSketchInterpreter + ConstraintEvaluator` 路径可用
- 单 batch 训练稳定运行
- 四项 loss 均可被记录
- 结论：`已放行。ConstraintPredHead 路径可用，DifferentiableSketchInterpreter + ConstraintEvaluator 路径可用，单 batch 训练稳定运行，四项 loss 均可记录。`

### P3-01 Modify1 训练入口与日志对接

- 完成状态：`[x] 已完成`  `[ ] 未完成`
- 验证状态：`[x] 已通过`  `[ ] 未验证`
- 可测试：是
- 前置任务：G2
- 目标：让 Modify1 不是一次性代码路径，而是可重复运行的训练分支。
- 实现清单：
- 训练入口支持 Modify1 模式
- 日志写出 `loss_cmd`、`axis_loss`、`pred_loss`、`geom_loss`
- checkpoint 中保存新增模块状态
- 配置项中加入 `alpha`、`gamma`
- 建议验证：
- 训练日志中出现新增 loss 字段
- checkpoint 可正常保存
- 继续训练时能恢复新增模块参数
- 验证证据：
- 测试文件/命令：`python -m constraint_fused_deepcad_simplify_modify1.train --device cpu --gpu_ids cpu --batch_size 1 --num_workers 0 --nr_epochs 1 --max_steps 1 --proj_dir proj_log/constraint_fused_deepcad_simplify_modify1 --exp_name cf_simplify_modify1_smoke --force_overwrite`；`python -m constraint_fused_deepcad_simplify_modify1.train --device cpu --gpu_ids cpu --batch_size 1 --num_workers 0 --nr_epochs 1 --max_steps 2 --proj_dir proj_log/constraint_fused_deepcad_simplify_modify1 --exp_name cf_simplify_modify1_smoke --continue --ckpt latest`
- 结果摘要：`训练入口与日志验证通过：smoke run 生成 config、TensorBoard 日志、train_metrics.csv、latest.pth；继续训练模式可从 latest checkpoint 恢复，新增模块状态可随 checkpoint 一起保存与加载。`

### P3-02 Modify1 评估口径与 `R_h/R_v` 对齐

- 完成状态：`[x] 已完成`  `[ ] 未完成`
- 验证状态：`[x] 已通过`  `[ ] 未验证`
- 可测试：是
- 前置任务：P3-01
- 目标：确保 Modify1 仍然沿用原 simplify 的效果判定口径。
- 实现清单：
- 复用 `evaluate_axis_constraints.py`
- 检查 Modify1 输出重建结果可被原评估器消费
- 输出 `R_h`、`R_v`、`axis_precision_mean`、`axis_recall_mean`
- 建立 Modify1 结果 summary
- 建议验证：
- 与 `论文尝试/DeepCAD原始约束指标` 口径一致
- 至少在 1 个小测试集上稳定输出 summary
- 评估脚本不需要知道内部新增 supervision 细节
- 验证证据：
- 测试文件/命令：`python -m unittest tests.test_constraint_fused_deepcad_simplify_modify1`；`python -c "import os, tempfile, h5py; from constraint_fused_deepcad_simplify_modify1.config.config_constraint_fused_simplify_modify1 import ConfigConstraintFusedSimplifyModify1; from constraint_fused_deepcad_simplify_modify1.application.evaluate_axis_constraints import build_reconstruction_use_case, reconstruct_batch, aggregate_metrics; from constraint_fused_deepcad_simplify_modify1.infrastructure.dataset_simplify_modify1 import get_simplify_modify1_dataloader; cfg=ConfigConstraintFusedSimplifyModify1('test'); cfg.device='cpu'; cfg.gpu_ids='cpu'; cfg.batch_size=1; cfg.num_workers=0; cfg.proj_dir='proj_log/constraint_fused_deepcad_simplify_modify1'; cfg.exp_name='cf_simplify_modify1_smoke'; cfg.exp_dir=os.path.join(cfg.proj_dir, cfg.exp_name); cfg.model_dir=os.path.join(cfg.exp_dir, 'model'); use_case, device = build_reconstruction_use_case(cfg); loader=get_simplify_modify1_dataloader('test', cfg, shuffle=False); batch=next(iter(loader)); out_vec, gt_vec = reconstruct_batch(use_case, batch, device); temp_dir=tempfile.mkdtemp(prefix='modify1_eval_', dir='d:/DeepCAD/DeepCAD/proj_log/constraint_fused_deepcad_simplify_modify1/cf_simplify_modify1_smoke/artifacts'); path=os.path.join(temp_dir, 'sample_vec.h5'); f=h5py.File(path,'w'); f.create_dataset('out_vec', data=out_vec[0], dtype='i8'); f.create_dataset('gt_vec', data=gt_vec[0], dtype='i8'); f.close(); rows, summary = aggregate_metrics(temp_dir, cfg.angle_thresh, cfg.grid_size); print('rows', len(rows)); print('R_h', summary['R_h']); print('R_v', summary['R_v']); print('axis_precision_mean', summary['axis_precision_mean']); print('axis_recall_mean', summary['axis_recall_mean'])"`
- 结果摘要：`评估闭环验证通过：Modify1 输出重建结果可被评估器消费，summary 已显式产出 R_h、R_v、axis_precision_mean、axis_recall_mean；在 1 个测试样本上的最小闭环已稳定输出指标。`

### P3-03 Simplify vs Modify1 对比实验

- 完成状态：`[ ] 已完成`  `[ ] 未完成`
- 验证状态：`[ ] 已通过`  `[ ] 未验证`
- 可测试：是
- 前置任务：P3-02
- 目标：回答 Modify1 是否比原 simplify 更值得继续。
- 实现清单：
- 选择同一数据切分
- 对比 simplify 与 Modify1 的 `R_h`、`R_v`
- 对比 `loss_cmd`、`axis_loss`
- 记录 `pred_loss`、`geom_loss`
- 输出结论表
- 建议验证：
- 至少完成 1 组 baseline 对比
- 保证训练轮数和主要配置一致
- 结论包含“是否值得继续扩展”的明确判断
- 验证证据：
- 实验记录路径：`____`
- 结果摘要：`____`

### P3-04 `alpha/beta/gamma` 消融实验

- 完成状态：`[ ] 已完成`  `[ ] 未完成`
- 验证状态：`[ ] 已通过`  `[ ] 未验证`
- 可测试：是
- 前置任务：P3-03
- 目标：区分三条监督链路各自的实际贡献。
- 实现清单：
- 做 `alpha=0` 消融
- 做 `gamma=0` 消融
- 做 `alpha>0, gamma>0` 联合版本
- 必要时比较不同 `beta`
- 建议验证：
- 每组实验至少输出 `R_h/R_v`
- 观察 `pred_loss` 与 `geom_loss` 是否真的带来收益
- 记录是否出现 `L_cmd` 明显恶化
- 验证证据：
- 实验记录路径：`____`
- 结果摘要：`____`

### P3-05 配置、checkpoint、复现记录固化

- 完成状态：`[ ] 已完成`  `[ ] 未完成`
- 验证状态：`[ ] 已通过`  `[ ] 未验证`
- 可测试：是
- 前置任务：P3-04
- 目标：让 Modify1 成为一个可复现实验版本，而非一次性验证脚本。
- 实现清单：
- 固化训练配置
- 固化评估配置
- 记录最佳 checkpoint
- 记录关键实验指标与对比表
- 明确 `alpha/beta/gamma` 最终推荐值
- 建议验证：
- 能从 checkpoint 恢复继续训练
- 能复现实验 summary
- 文档、配置、结果三者可互相对应
- 验证证据：
- 测试文件/命令：`____`
- 结果摘要：`____`

### G3 Modify1 验证结论输出

- 完成状态：`[ ] 已完成`  `[ ] 未完成`
- 验证状态：`[ ] 已通过`  `[ ] 未验证`
- 前置任务：P3-05
- 最终放行标准：
- 训练、验证、评估全部打通
- 至少产出 1 份 simplify vs Modify1 对比实验
- 至少产出 1 份 `alpha/beta/gamma` 消融记录
- 明确回答以下问题：
- `ConstraintPredHead` 是否带来正收益
- `L_geom_constraint` 是否带来正收益
- Modify1 是否显著优于原 simplify
- 是否值得继续扩展到更完整的 Fused 结构
- 最终结论：`____`

---

## 6. 推荐优先级

如果时间非常紧，建议优先完成下面 8 项：

1. `P0-01`
2. `P1-03`
3. `P2-01`
4. `P2-02`
5. `P2-03`
6. `P2-04`
7. `P2-05`
8. `P3-02`

这样就能最快得到一个能回答“新增 decoder 监督和 geometry loss 是否有效”的实验闭环。

---

## 7. 建议验收口径

建议把 Modify1 的最终验收压缩为四个问题：

1. `constraint_fused_deepcad_simplify` 的 Modify1 链路是否能独立跑通训练与评估。
2. 引入 `ConstraintPredHead` 后，decoder 隐状态监督是否带来可观测收益。
3. 引入 `L_geom_constraint` 后，`R_h`、`R_v` 是否比原 simplify 更接近 1。
4. `loss_cmd` 是否保持稳定，没有被 `alpha/gamma` 过度拉坏。

若这四点成立，则说明 Modify1 比原 simplify 更值得继续扩展；若不成立，则应优先复盘 geometry 对齐方式与监督重复问题，而不是盲目继续加复杂模块。
