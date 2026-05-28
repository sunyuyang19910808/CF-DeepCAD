# Constraint-Fused DeepCAD Step

基于原始 DeepCAD 主路径，逐步添加 **GT 正关系恢复** 几何约束 `L_geom` 的独立实验包。

## 架构红线

- 主生成闭包：`P(S | z)`，decoder 只依赖潜变量 `z`
- 总损失：`L_cmd + 2.0 * L_args + gamma_geom * L_geom`
- `L_geom` 只监督 GT 中明确存在的水平/竖直/平行/垂直正关系
- 不引入 `L_pred`、`L_recon`、constraint memory、decoder cross-attention
- 第一版不启用 A2d 双向 hard BCE 负样本惩罚

## 目录映射

| 模块 | 路径 |
| --- | --- |
| 配置 | `config/config_step.py` |
| 训练入口 | `train.py` |
| 重建入口 | `reconstruct.py` |
| 评估入口 | `evaluate.py` |
| 训练用例 | `application/train_use_case.py` |
| 重建/评估逻辑 | `application/evaluate_reconstruction.py` |
| GT 关系数据 | `infrastructure/dataset_step.py` |
| 预测线解析 | `application/differentiable_sketch_interpreter.py` |
| 正关系 L_geom | `application/geometry_constraint.py` |
| Gate 验证 | `verify_gates.py` |

## 训练示例

```bash
# S0: 原始 DeepCAD（无 L_geom 反传）
python -m constraint_fused_deepcad_step.train \
  --proj_dir proj_log/constraint_fused_deepcad_step \
  --exp_name deepcad_step_s0_origin \
  --data_root data \
  --batch_size 64 --nr_epochs 100 \
  -g 0

# S1: 几何日志（计算 geom，不反传）
python -m constraint_fused_deepcad_step.train \
  --proj_dir proj_log/constraint_fused_deepcad_step \
  --exp_name deepcad_step_s1_geom_log \
  --enable_geom_loss --geom_log_only \
  --batch_size 64 --nr_epochs 100 \
  -g 0

# S2: 弱正关系 L_geom（推荐 disk cache + num_workers=0）
python -m constraint_fused_deepcad_step.train \
  --proj_dir proj_log/constraint_fused_deepcad_step \
  --exp_name deepcad_step_s2_geom_pos \
  --data_root D:/DeepCAD/DeepCAD/data \
  --enable_geom_loss --gamma_geom 0.1 \
  --dataset_cache disk --num_workers 0 \
  --batch_size 64 --nr_epochs 100 \
  -g 0
```

磁盘缓存默认写入：

```text
{data_root}/.cache/deepcad_step/<cache_key>/<phase>/<sample_id>.pt
```

每个 `.pt` 含 `command/args/groups/unary_gt/pair_gt/line_*` 等字段；源 h5 变更时会自动失效并重建。

**注意**：`--dataset_cache disk` 时请配合 `--num_workers 0`，避免多 worker 重复建 cache。

## 评估（TrainRules §4）

评估分两步：硬解码重建 `*_vec.h5` → 离散 ACC + index-aligned 约束指标。

默认输出：

| 产物 | 路径 |
| --- | --- |
| 重建向量 | `{exp_dir}/artifacts/reconstruction_{eval_split}_latest/*_vec.h5` |
| 离散 ACC | `{reconstruction_dir}_acc_stat.txt` |
| 约束汇总 | `{exp_dir}/artifacts/{eval_split}_eval_latest/summary.json` |
| 逐样本约束 | `{exp_dir}/artifacts/{eval_split}_eval_latest/per_sample_counts.csv` |

`ratio_h` / `ratio_v` / `parallel` / `perpendicular` 采用 **index-aligned** 口径（与 High Modify `evaluate_constraints` 一致），可与主路径消融表对齐。

### 一步完成（重建 + ACC + 约束聚合）

```bash
python -m constraint_fused_deepcad_step.evaluate \
  --proj_dir proj_log/constraint_fused_deepcad_step \
  --exp_name deepcad_step_s2_geom_pos \
  --data_root D:/DeepCAD/DeepCAD/data \
  --ckpt latest \
  --eval_split test \
  -g 0
```

### 分步执行

```bash
# 1) 仅重建
python -m constraint_fused_deepcad_step.reconstruct \
  --proj_dir proj_log/constraint_fused_deepcad_step \
  --exp_name deepcad_step_s2_geom_pos \
  --data_root D:/DeepCAD/DeepCAD/data \
  --ckpt 20 \
  --reconstruction_dir proj_log/constraint_fused_deepcad_step/deepcad_step_s2_geom_pos/artifacts/reconstruction_test_ep20

# 2) 跳过重建，只重算指标
python -m constraint_fused_deepcad_step.evaluate \
  --proj_dir proj_log/constraint_fused_deepcad_step \
  --exp_name deepcad_step_s2_geom_pos \
  --skip_reconstruct \
  --reconstruction_dir proj_log/constraint_fused_deepcad_step/deepcad_step_s2_geom_pos/artifacts/reconstruction_test_ep20
```

子集调试（不写进论文主表）：

```bash
python -m constraint_fused_deepcad_step.evaluate \
  ... \
  --sample_count 8
```

## Gate 验证（P0-01 ~ G3）

```bash
python -m constraint_fused_deepcad_step.verify_gates --data_root data --device cpu
```

## 与 A2c/A2d 差异

| 项 | Step 方案 | A2c/A2d |
| --- | --- | --- |
| 主路径 | 原始 DeepCAD | High Modify encoder/decoder 栈 |
| 辅助损失 | 仅 `L_geom` | `L_pred` + `L_recon` + `L_geom` |
| 几何监督 | GT 正关系 BCE | soft residual 或双向 hard BCE |
| `L_args` 权重 | 固定 2.0 | 可变/受辅助冲击 |

详细方案见：`论文尝试/约束感知生成/7_constraint_fused_deepcad_step/DeepCAD逐步添加几何约束技术方案.md`
