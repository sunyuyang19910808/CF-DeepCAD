# P0-01 范围冻结与目录映射

## 方案定位

在原始 DeepCAD 的 `Encoder -> Bottleneck -> Decoder(z) -> FCN` 主路径上，只增加训练期 GT 正关系几何约束 `L_geom`。

## 冻结边界

### 包含

- 原始 DeepCAD 模型结构与 `P(S | z)` 生成闭包
- GT CAD 向量解析出的 line / pair 正关系标签
- 从 `args_logits` 解析预测线方向并计算正关系恢复损失
- 总损失三项：`L_cmd`、`L_args`（权重 2.0）、`L_geom`

### 不包含

- `L_pred`、`L_recon`
- constraint token / constraint memory / decoder cross-attention
- A2d 双向 hard BCE 强负样本惩罚
- 推理期必需的外部约束输入

## 代码落点

```
constraint_fused_deepcad_step/
  config/config_step.py
  application/train_use_case.py
  application/geometry_constraint.py
  application/differentiable_sketch_interpreter.py
  application/geom_schedule.py
  infrastructure/dataset_step.py
  train.py
  verify_gates.py
```

## 日志落点

```
proj_log/constraint_fused_deepcad_step/<exp_name>/
  config.txt
  model/
  log/
  artifacts/train_metrics.csv
  artifacts/manifest.json
```

## 与 A2c/A2d 差异摘要

- **A2c**：软几何残差，训练指标与测试评估口径不一致
- **A2d**：GT 关系 + 双向 hard BCE + 强辅助权重，易冲击主路径
- **Step**：原始 DeepCAD + GT 正关系 BCE，仅监督 GT=1 的关系恢复
