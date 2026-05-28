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
| 训练用例 | `application/train_use_case.py` |
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

# S2: 弱正关系 L_geom（推荐开启 disk cache，避免重复解析 GT 关系）
python -m constraint_fused_deepcad_step.train \
  --proj_dir proj_log/constraint_fused_deepcad_step \
  --exp_name deepcad_step_s2_geom_pos \
  --data_root D:/DeepCAD/DeepCAD/data \
  --enable_geom_loss --gamma_geom 0.1 \
  --dataset_cache disk \
  --batch_size 64 --nr_epochs 100 \
  -g 0
```

磁盘缓存默认写入：

```text
{data_root}/.cache/deepcad_step/<cache_key>/<phase>/<sample_id>.pt
```

每个 `.pt` 含 `command/args/groups/unary_gt/pair_gt/line_*` 等字段；源 h5 变更时会自动失效并重建。

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
