# Constraint-Fused DeepCAD Simplify DDD技术方案

本文档面向一个**快速验证版** `constraint_fused_deepcad_simplify`，目标是在保持 DeepCAD 主干兼容的前提下，单独实现一个只考虑**水平（Horizontal）**、**竖直（Vertical）**约束的最小闭环。文档遵循项目 `.cursor/rules/TechnicalProposal.mdc`：包含**方案目标**、**整体架构**、**各模块架构**（每模块含作用、原理、代码、举例）与**总结**。

本方案不是对 `constraint_fused_deepcad` 的删减式侵入修改，而是一个**独立代码包**的技术设计，未来代码落点为：

```text
constraint_fused_deepcad_simplify/
```

---

## 1. 方案目标

### 1.1 业务目标

在不引入新数据、不修改 DeepCAD 命令表示的前提下，快速验证一个问题：

> 当约束范围只保留水平/竖直，并且约束信息在编码器侧进入模型时，是否能以较低工程复杂度提升生成结果中的轴对齐约束保持能力。

### 1.2 技术目标

| 目标 | 说明 | 验收方式 |
| --- | --- | --- |
| 最小闭环可运行 | 从 JSON / vec 提取 H/V 约束，到训练，再到评估完整打通 | 能跑通 1 个 batch 训练和 1 次验证 |
| 编码器侧约束注入 | 约束不只在 loss 或评估出现，而是进入 encoder embedding | `constraint_tags` 参与 encoder 前向 |
| 独立代码包 | 与 `constraint_fused_deepcad` 解耦，便于后续平行演进 | 新代码全部落在 `constraint_fused_deepcad_simplify/` |
| 快速验证优先 | 暂不实现平行/垂直/共线、token 联合序列、Cross-Attn | 范围冻结清晰，文档与任务清单一致 |
| 指标简单可信 | 先用 `R_h`、`R_v` 和重建误差验证方向正确 | 对齐 `论文尝试/DeepCAD原始约束指标` 口径 |

### 1.3 非目标

以下内容明确**不在本轮范围**：

1. 不处理 `Parallel`、`Perpendicular`、`Collinear`。
2. 不实现约束 token 联合序列编码。
3. 不实现 decoder 侧 `Constraint Cross-Attn`。
4. 不实现 Latent GAN 适配。
5. 不新增人工标注，不改 DeepCAD 官方数据格式。

### 1.4 核心设计原则

1. **快速验证优先于完备性**。
2. **只保留对 H/V 约束有效的最短路径**。
3. **编码器融合必须保留**，这是区别于“只做评估脚本”的关键。
4. **领域边界先清晰，再考虑神经网络细节**。
5. **实现独立于 `constraint_fused_deepcad`，但尽量复用其思路与命名习惯**。

---

## 2. 整体架构

### 2.1 方案概览

本简化版把完整 Constraint-Fused 思路收缩为 4 个限界上下文：

| 限界上下文 | 职责 | 是否保留 |
| --- | --- | --- |
| `Sketch Preparation Context` | 提取 H/V 约束，构造标签与监督 | 保留 |
| `Encoding Context` | 将命令 embedding 与 H/V tags 融合进 encoder | 保留 |
| `Generation Context` | 基于 latent 重建 CAD 命令序列 | 保留 |
| `Training & Evaluation Context` | 组合损失、训练单步、评估 `R_h/R_v` | 保留 |

完整 Fused 方案中的以下上下文能力，本次降级：

| 完整版能力 | 本版处理方式 |
| --- | --- |
| `ConstraintTokenEncoder` | 删除 |
| 联合序列 `E_joint=[E_cmd; E_con]` | 删除 |
| `pair_gt` / pair recon | 删除 |
| decoder Cross-Attn(C) | 删除 |
| H/V 之外的五类约束统一建模 | 删除 |

### 2.2 核心信息流

```text
DeepCAD JSON / cad_vec
        │
        ▼
[Sketch Preparation]
提取 Line → 判定 Horizontal / Vertical
        │
        ├──► constraint_tags  (S, 2)
        └──► unary_gt         (L, 2)
        │
        ▼
[Encoding]
CADEmbedding + AxisTagEmbedding
        │
        ▼
Transformer Encoder
        │
        ▼
Pooling + Bottleneck
        │
        ├──► z → Decoder → CAD logits
        └──► z → AxisReconHead → unary_pred
        │
        ▼
[Training & Evaluation]
L_total = L_cmd + beta * L_axis_recon
评估：R_h / R_v / axis_recall
```

### 2.3 与完整 `constraint_fused_deepcad` 的差异

| 维度 | 完整版 | 简化版 |
| --- | --- | --- |
| 约束类型 | H/V/Parallel/Perpendicular/Collinear | 仅 H/V |
| 约束输入形式 | tag + token 双通路 | 仅 tag |
| 重建监督 | unary + pair | 仅 unary |
| 复杂度 | 中高 | 低 |
| 验证目标 | 约束感知 latent 的全量闭环 | 编码器侧 H/V 感知是否有效 |

### 2.4 目标目录结构

```text
constraint_fused_deepcad_simplify/
├─ __init__.py
├─ domain/
│  ├─ entities.py
│  └─ services.py
├─ sketch_preparation/
│  ├─ constraint_extractor_simplify.py
│  └─ batch_assembler_simplify.py
├─ encoding/
│  ├─ embeddings.py
│  ├─ encoder_simplify.py
│  └─ pooling.py
├─ generation/
│  ├─ decoder_adapter.py
│  └─ axis_recon_head.py
├─ application/
│  ├─ train_use_case.py
│  ├─ evaluate_axis_constraints.py
│  └─ loss_composer.py
├─ infrastructure/
│  ├─ dataset_simplify.py
│  └─ repository.py
└─ config/
   └─ config_constraint_fused_simplify.py
```

---

## 3. 领域模型设计

### 3.1 聚合根：`SketchSequenceAggregateSimplify`

#### 模块作用

统一持有一个样本在简化版中的全部视图：命令、H/V 约束、命令级标签和一元监督张量，避免标签与监督在多个脚本里重复推导。

#### 模块原理

该聚合根维护以下不变量：

1. 只有 `LINE` 命令才允许关联 `line_ref`。
2. `ConstraintRelationSimplify` 只允许 `HORIZONTAL` 或 `VERTICAL`。
3. `constraint_tags` 必须由约束关系自动投影生成。
4. `unary_gt` 必须与约束关系保持一致。

#### 代码

```python
from dataclasses import dataclass
from typing import List, Optional

import torch


class ConstraintTypeSimplify:
    HORIZONTAL = 0
    VERTICAL = 1


@dataclass
class CadCommand:
    command_id: int
    args: List[int]
    group_id: Optional[int] = None
    line_ref: Optional[int] = None


@dataclass
class ConstraintRelationSimplify:
    type_id: int
    line_idx: int


@dataclass
class SketchSequenceAggregateSimplify:
    commands: List[CadCommand]
    constraints: List[ConstraintRelationSimplify]
    constraint_tags: torch.Tensor   # (S, 2)
    unary_gt: torch.Tensor          # (L, 2)
    cmd_padding_mask: torch.Tensor
    line_count: int
```

#### 举例说明

若一个草图中第 1、4 条线分别为水平和竖直，则：

- `constraints` 中有两个关系：`HORIZONTAL(line=1)`、`VERTICAL(line=4)`。
- `constraint_tags` 只在对应 `LINE` 命令位置置位。
- `unary_gt[1] = [1, 0]`，`unary_gt[4] = [0, 1]`。

---

### 3.2 实体：`ConstraintRelationSimplify`

#### 模块作用

表达一条线的轴对齐属性，是简化版中唯一的约束实体。

#### 模块原理

与完整版不同，本版不再处理二元关系，因此约束实体只需要：

1. 约束类型。
2. 线段索引。

这种建模能够显著减少数据构造、监督张量、loss 和评估复杂度。

#### 代码

```python
@dataclass
class ConstraintRelationSimplify:
    type_id: int   # HORIZONTAL or VERTICAL
    line_idx: int
```

#### 举例说明

若 `line_idx=7` 对应一条沿 x 轴方向的线，则其关系表示为：

```python
ConstraintRelationSimplify(type_id=ConstraintTypeSimplify.HORIZONTAL, line_idx=7)
```

---

### 3.3 值对象：`AxisTagVector`

#### 模块作用

为每个命令位置提供一个轻量约束语义标签，让 encoder 在嵌入层就能感知该命令是否对应水平或竖直线。

#### 模块原理

`AxisTagVector` 固定为 2 维：

1. `horizontal`
2. `vertical`

对于非 `LINE` 命令，默认标签为 `[0, 0]`。

#### 代码

```python
@dataclass(frozen=True)
class AxisTagVector:
    horizontal: int
    vertical: int
```

#### 举例说明

一段 `LINE` 命令若被判定为竖直，则其 tag 为：

```python
AxisTagVector(horizontal=0, vertical=1)
```

---

## 4. 模块架构设计

### 4.1 模块：`constraint_extractor_simplify.py`

#### 模块作用

从 DeepCAD 的 `CADSequence` 或 `cad_vec` 中恢复所有 `Line`，自动判定每条线是否满足水平/竖直条件，并输出结构化约束列表。

#### 模块原理

1. 从 `CADSequence` 中收集所有 `Line`。
2. 计算每条线的二维方向向量。
3. 与 x 轴、y 轴单位向量做无向夹角比较。
4. 小于阈值则标记为 `horizontal` 或 `vertical`。

这里沿用已有评估口径：

- **水平**：与 x 轴单位向量的无向夹角 `< angle_thresh`
- **竖直**：与 y 轴单位向量的无向夹角 `< angle_thresh`

#### 代码

```python
def extract_axis_constraints(lines, angle_thresh):
    horizontal = []
    vertical = []
    ex = np.array([1.0, 0.0], dtype=np.float64)
    ey = np.array([0.0, 1.0], dtype=np.float64)

    for i, line in enumerate(lines):
        di = line_direction_xy(line)
        if undirected_angle_deg(di, ex) < angle_thresh:
            horizontal.append(i)
        if undirected_angle_deg(di, ey) < angle_thresh:
            vertical.append(i)
    return horizontal, vertical
```

#### 举例说明

若一个样本有 5 条线，其中第 0、2 条近似水平，第 3 条近似竖直，则输出为：

```json
{
  "horizontal": [0, 2],
  "vertical": [3]
}
```

---

### 4.2 模块：`batch_assembler_simplify.py`

#### 模块作用

把命令序列与 H/V 约束关系组装成模型前向可直接消费的 batch 数据，包括 `constraint_tags`、`unary_gt` 和 padding mask。

#### 模块原理

1. 遍历命令序列，给每条 `LINE` 命令分配 `line_ref`。
2. 根据 `ConstraintRelationSimplify` 把线级约束投影到命令级位置。
3. 构造 `unary_gt(line_idx, axis_type)` 监督矩阵。

#### 代码

```python
def build_constraint_tags(seq_len, commands_np, relations):
    tags = torch.zeros(seq_len, 2, dtype=torch.float32)
    line_to_pos = {}
    line_id = 0
    for t in range(seq_len):
        if int(commands_np[t]) == LINE_IDX:
            line_to_pos[line_id] = t
            line_id += 1

    for rel in relations:
        pos = line_to_pos.get(rel.line_idx)
        if pos is None:
            continue
        tags[pos, rel.type_id] = 1.0
    return tags
```

#### 举例说明

若第 6 个命令是 `LINE` 且其线索引为 `2`，同时 `line_idx=2` 被标记为水平，则 `constraint_tags[6] = [1, 0]`。

---

### 4.3 模块：`encoding/embeddings.py`

#### 模块作用

在 DeepCAD 原有命令 embedding 上，增加一个二维轴约束标签投影层。

#### 模块原理

本模块保留原始：

1. `command_embed`
2. `arg_embed`
3. 可选 `group_embed`
4. `pos_encoding`

并新增：

5. `AxisTagEmbedding(2 -> d_model)`

最终输出：

```text
E_cmd = command_emb + arg_emb + group_emb(optional) + axis_tag_emb
```

#### 代码

```python
class AxisTagEmbedding(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, tags):
        return self.proj(tags.float())
```

#### 举例说明

对同一个 `LINE` 命令：

- 若 tag 为 `[0, 0]`，模型只看到原始几何语义。
- 若 tag 为 `[1, 0]`，encoder 能在输入层获知“这是一条水平线”。

---

### 4.4 模块：`encoding/encoder_simplify.py`

#### 模块作用

用最少改动实现“编码器侧融合”。它接收带 H/V tags 的命令序列，经过 Transformer Encoder 和 pooling 输出约束感知 latent。

#### 模块原理

本模块与完整版最大的区别是：

1. 不拼接约束 token。
2. 不构造联合 mask。
3. 只对命令序列做 self-attention。
4. H/V 信息通过 command position 上的 tag embedding 注入。

这是一种**轻量 fused**：约束仍进入 encoder，但工程复杂度远低于双流联合编码。

#### 代码

```python
class EncoderSimplify(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.embedding = CADEmbeddingWithAxisTags(cfg, cfg.max_total_len)
        self.encoder = TransformerEncoder(...)
        self.pooling = MaskedMeanPooling()
        self.bottleneck = Bottleneck(...)

    def forward(self, commands, args, constraint_tags, cmd_padding_mask, groups=None):
        e_cmd = self.embedding(commands, args, groups, constraint_tags)
        memory = self.encoder(e_cmd, src_key_padding_mask=cmd_padding_mask)
        z_pre = self.pooling(memory, cmd_padding_mask)
        return self.bottleneck(z_pre)
```

#### 举例说明

若两个样本几何形状接近，但一个样本包含更多水平线，则该差异会通过 `constraint_tags` 在 encoder 中提前体现，而不是等到评估阶段才被看到。

---

### 4.5 模块：`generation/axis_recon_head.py`

#### 模块作用

从 latent 预测每条线的 H/V 属性，用于验证 latent 中是否真正保留了轴约束信息。

#### 模块原理

因为本版只关注 H/V，不再需要完整约束图重建。只需要从 `z` 预测：

```text
unary_pred: (N, max_lines, 2)
```

其中最后一个维度分别表示：

1. `is_horizontal`
2. `is_vertical`

#### 代码

```python
class AxisReconHead(nn.Module):
    def __init__(self, d_model: int, max_lines: int):
        super().__init__()
        self.max_lines = max_lines
        self.out = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, max_lines * 2),
        )

    def forward(self, z):
        logits = self.out(z.squeeze(0))
        return logits.view(z.shape[1], self.max_lines, 2)
```

#### 举例说明

若某 batch 中第一条样本第 5 条线真实为竖直，则训练时会推动：

```text
unary_pred[0, 5, 1] -> 1
```

---

### 4.6 模块：`application/loss_composer.py`

#### 模块作用

组合命令重建损失与 H/V 一元重建损失。

#### 模块原理

总损失定义为：

```text
L_total = L_cmd + beta * L_axis_recon
```

其中：

- `L_cmd`：DeepCAD 原有命令/参数预测损失
- `L_axis_recon`：对 `unary_pred` 与 `unary_gt` 做带 mask 的 BCE

#### 代码

```python
def compose_loss(cmd_loss, unary_pred, unary_gt, line_mask, beta):
    bce = F.binary_cross_entropy_with_logits(unary_pred, unary_gt, reduction="none")
    masked = bce * line_mask.unsqueeze(-1).float()
    axis_loss = masked.sum() / line_mask.sum().clamp(min=1)
    return cmd_loss + beta * axis_loss, axis_loss
```

#### 举例说明

若模型命令预测已经较准，但 `axis_loss` 仍较高，则说明模型能画出线，却还没学会保持其水平/竖直属性。

---

### 4.7 模块：`application/train_use_case.py`

#### 模块作用

封装一次训练 batch 的编排逻辑，隔离领域规则与训练脚本。

#### 模块原理

单 batch 训练流程如下：

1. dataset 提供命令、args、tags、`unary_gt`。
2. `EncoderSimplify` 输出 latent `z`。
3. decoder 根据 `z` 预测命令 logits。
4. `AxisReconHead` 根据 `z` 预测 H/V logits。
5. `LossComposer` 输出总损失。

#### 代码

```python
class TrainConstraintFusedSimplifyBatchUseCase:
    def execute(self, batch):
        z = self.encoder(
            commands=batch["command"].transpose(0, 1),
            args=batch["args"].transpose(0, 1),
            constraint_tags=batch["constraint_tags"].transpose(0, 1),
            cmd_padding_mask=batch["cmd_padding_mask"],
            groups=batch.get("groups"),
        )
        cmd_logits, args_logits = self.decoder(z)
        unary_pred = self.axis_recon_head(z)
        total, axis_loss = self.loss_composer(...)
        return {"loss": total, "axis_loss": axis_loss}
```

#### 举例说明

训练脚本不再直接知道 H/V 提取细节，只调用用例对象即可。这保证后续即使切换提取规则，也不会牵连训练入口。

---

### 4.8 模块：`application/evaluate_axis_constraints.py`

#### 模块作用

对齐现有评估口径，计算水平/竖直约束的恢复情况。

#### 模块原理

优先采用两个层次的指标：

1. **全局比值指标**
   - `R_h = sum(pred_h) / sum(gt_h)`
   - `R_v = sum(pred_v) / sum(gt_v)`
2. **样本级补充指标**
   - `axis_recall_mean`
   - `axis_precision_mean`

第一层指标直接兼容已有 `论文尝试/DeepCAD原始约束指标`。

#### 代码

```python
def aggregate_axis_metrics(sum_pred_h, sum_gt_h, sum_pred_v, sum_gt_v):
    return {
        "R_h": None if sum_gt_h == 0 else sum_pred_h / sum_gt_h,
        "R_v": None if sum_gt_v == 0 else sum_pred_v / sum_gt_v,
    }
```

#### 举例说明

若测试集 GT 中总共有 200 条水平线，模型预测结果中提取出 180 条水平线，则：

```text
R_h = 180 / 200 = 0.90
```

这说明模型整体保留了 90% 左右的水平线数量规模。

---

### 4.9 模块：`infrastructure/dataset_simplify.py`

#### 模块作用

把原始 DeepCAD 数据读取逻辑与简化版聚合根对接起来，形成可复用的独立数据访问层。

#### 模块原理

1. 复用原始 `cad_vec` 或 JSON 数据源。
2. 调用 `constraint_extractor_simplify.py` 得到 H/V 约束。
3. 调用 `batch_assembler_simplify.py` 组装聚合根。
4. 输出训练 batch 所需字段。

#### 代码

```python
sample = {
    "command": command_tensor,
    "args": args_tensor,
    "constraint_tags": agg.constraint_tags,
    "unary_gt": agg.unary_gt,
    "cmd_padding_mask": agg.cmd_padding_mask,
    "line_count": agg.line_count,
    "id": sample_id,
}
```

#### 举例说明

同一个训练脚本未来可替换 dataset，而模型侧接口基本不变。这保证独立包后续做扩展时不会绑死在旧实现上。

---

## 5. 应用服务与限界上下文映射

### 5.1 Sketch Preparation Context

负责：

1. 解析 DeepCAD 数据。
2. 提取 H/V 约束。
3. 构造 tags 与 `unary_gt`。

典型模块：

- `constraint_extractor_simplify.py`
- `batch_assembler_simplify.py`

### 5.2 Encoding Context

负责：

1. 命令 embedding
2. H/V tag 融合
3. encoder 编码
4. pooling + bottleneck

典型模块：

- `encoding/embeddings.py`
- `encoding/encoder_simplify.py`

### 5.3 Generation Context

负责：

1. 基于 `z` 重建 CAD 命令
2. 基于 `z` 重建 H/V 线级属性

典型模块：

- `generation/decoder_adapter.py`
- `generation/axis_recon_head.py`

### 5.4 Training & Evaluation Context

负责：

1. 训练单 batch 编排
2. loss 组合
3. 指标评估
4. checkpoint / 日志记录

典型模块：

- `application/train_use_case.py`
- `application/loss_composer.py`
- `application/evaluate_axis_constraints.py`

---

## 6. 关键张量约定

| 名称 | 含义 | 典型形状 |
| --- | --- | --- |
| `commands` | 命令序列 | `(S, N)` |
| `args` | 参数序列 | `(S, N, n_args)` |
| `constraint_tags` | 命令级 H/V 标签 | `(S, N, 2)` |
| `cmd_padding_mask` | 命令 padding mask | `(N, S)` |
| `z` | 约束感知 latent | `(1, N, d_model)` |
| `unary_gt` | 线级 H/V 真值 | `(N, max_lines, 2)` |
| `unary_pred` | 线级 H/V 预测 | `(N, max_lines, 2)` |

---

## 7. 分阶段落地建议

### 7.1 Phase 1：数据与标签

打通：

1. H/V 提取
2. `constraint_tags`
3. `unary_gt`
4. dataset 输出

### 7.2 Phase 2：最小模型改造

打通：

1. `AxisTagEmbedding`
2. `EncoderSimplify`
3. `AxisReconHead`
4. `L_total`

### 7.3 Phase 3：验证闭环

打通：

1. 单 batch 训练
2. 评估脚本
3. `R_h / R_v`
4. 对比基线结果

---

## 8. 风险与取舍

### 8.1 优势

1. 范围小，开发快。
2. 指标简单，易解释。
3. 仍保留“约束进入 encoder”的核心思想。
4. 后续可平滑升级到完整 `constraint_fused_deepcad`。

### 8.2 风险

1. 只做 H/V，无法证明模型对 pair 约束也有效。
2. 仅使用 tag 注入，可能低估完整 Fused 联合序列建模的收益。
3. `R_h / R_v` 只能反映轴对齐数量尺度，不能完全反映几何质量。

### 8.3 取舍结论

对于“快速验证方案是否值得继续”的目标，这种简化是合理的：

- 它保留了**最关键的假设**：约束进入 encoder 能否改善 latent。
- 它删除了**最耗时的部分**：token 双流编码、pair loss、decoder cross attention。

---

## 9. 总结

`constraint_fused_deepcad_simplify` 的定位不是完整替代 `constraint_fused_deepcad`，而是一个**低复杂度、强可验证、独立目录**的实验版本。

它的核心特点如下：

1. 只处理**水平/竖直**两类约束。
2. 只保留**命令级 tag 融合**这一条最短 encoder 侧约束注入路径。
3. 只保留**一元约束重建**，不处理 pair 约束。
4. 用 `R_h / R_v` 与 `axis_recon_loss` 做快速效果验证。
5. 未来若效果成立，可按“加 token 编码、加 pair_gt、加 Cross-Attn”的顺序向完整方案演进。

如果后续开始真正编码实现，建议把本文件作为 `constraint_fused_deepcad_simplify` 的**唯一设计基线**，任务推进以配套任务清单为准，HTML 架构图用于对齐模块边界与信息流。
