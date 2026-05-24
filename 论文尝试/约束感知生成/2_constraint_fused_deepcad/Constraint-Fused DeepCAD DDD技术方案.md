# Constraint-Fused DeepCAD 详细设计（DDD 版）

本文档以《[Constraint-Fused DeepCAD 技术方案](./Constraint-Fused%20DeepCAD%20技术方案.md)》为**唯一技术事实来源**，在同一目标与模块边界下，按领域驱动设计（DDD, Domain-Driven Design）重组为**可落地的详细设计**：限界上下文、领域模型、应用服务、基础设施与训练/推理用例。撰写口径对齐项目 `.cursor/rules/TechnicalProposal.mdc`（方案目标、整体架构、模块含作用/原理/代码/举例、总结）。

**与 Constraint-Aware DeepCAD 的关系**：Fused 方案**完整继承**基线的约束定义、离线提取逻辑、评估口径与「命令 token 表示不变」等约定；演进点为约束在**编码器侧与命令联合建模**，使瓶颈 \(z\) 显式携带约束语义，与 Latent GAN 及「推理不强制依赖外部 \(C\)」一致。主技术方案中**第 3.0、3.8 节与基线共用**；DDD 文档中对应映射为 **Sketch Preparation Context（提取/标准化）** 与 **评估用例**。

<!-- markdownlint-disable MD024 -->

---

## 1. 方案目标

### 1.1 业务问题与技术问题

当前 DeepCAD 及其常见 Constraint-Aware 变体存在一个根本问题：**约束主要在解码阶段注入，而不在编码阶段进入瓶颈表示**。这导致系统表面上“知道约束”，但真正可复用的 latent code 并不稳定地携带约束语义。

其直接后果如下：

1. **隐空间生成不可控**：Latent GAN 只学习 \(z\) 的分布，若 \(z\) 中缺少约束信息，则采样生成无法继承几何关系。
2. **训练与推理路径割裂**：训练依赖外部约束 \(C\)，推理却常常只依赖 \(z\)，二者的信息路径不一致。
3. **监督难以闭环**：解码器被要求“服从约束”，但编码器未被要求“理解约束”，监督无法有效压回瓶颈层。
4. **系统难以扩展**：约束相关逻辑常散落在数据脚本、embedding、decoder 和 loss 中，缺少统一领域抽象，工程维护成本高。

### 1.2 设计目标

| 目标 | 说明 | 验收标准 |
| --- | --- | --- |
| 约束感知隐空间 | 使 \(z\) 同时压缩命令几何与约束关系 | 仅输入 \(z\) 时仍具备较稳定约束满足率 |
| 训练推理一致性 | 推理不以外部约束序列 \(C\) 为必需输入 | Latent GAN 路径不调用 constraint encoder |
| DDD 分层清晰 | 领域逻辑、编排逻辑、基础设施隔离 | 代码目录与接口职责明确，不跨层耦合 |
| 兼容现有 DeepCAD | 保持命令格式、数据格式与主训练框架可渐进迁移 | 原训练脚本可分阶段接入 |
| 可观测可验证 | 对约束提取、融合、重建、解码全过程可单测与评估 | 每个关键模块均有输入/输出契约与指标 |

### 1.3 核心设计原则

1. **约束进入编码器，而不是只进入解码器**。
2. **领域模型先行，神经网络模块是领域对象的实现细节**。
3. **训练编排与模型领域逻辑解耦**，避免把业务流程硬编码进 `forward()`。
4. **保持 DeepCAD 命令语义稳定**，约束融合以增强为主，不推翻原格式。
5. **推理路径以 latent-only 为主路径**，任何外部约束访问都只能是可选增强。

### 1.4 约束类型范围（2D 草图，与基线一致）

与 Constraint-Aware 文档一致，优先覆盖计算简单、易自动判定的五类关系（用于提取器、tag 与 token 类型 id）：

| 中文 | 英文 |
| --- | --- |
| 水平 | Horizontal |
| 竖直 | Vertical |
| 平行 | Parallel |
| 垂直 | Perpendicular |
| 共线 | Collinear |

### 1.5 继承自 Constraint-Aware 的基线目标（仍适用于本方案数据与评估）

以下四条在 Fused 方案中**仍然成立**，仅「约束主注入位置」由解码器扩展为「编码器 + 可选解码器」：

1. **数据与标注**：完全复用 DeepCAD 官方 JSON 命令数据；约束由几何自动推导，**不引入新标注、不另建数据集**。
2. **表示与模型（命令侧）**：保持 DeepCAD 命令 token 表示不变；在基线中约束经嵌入 + 解码器交叉注意力 + 可选预测头**最小侵入**接入；本方案在此基础上增加编码器侧融合与 \(z\) 侧重建。
3. **优化目标**：总损失在命令预测损失基础上叠加约束相关项（基线为 `L_cmd + α·L_constraint`；本方案见 4.9 节扩展形式），使生成分布在统计意义上更贴近训练集中由几何导出的约束结构。
4. **评估与落地**：可计算水平/竖直/平行/垂直/共线等满足率，并结合 Chamfer/Hausdorff、拓扑合法性等指标，与原版流程对齐（见 4.10 节）。

---

## 2. DDD 视角下的整体架构

### 2.1 限界上下文划分

本方案将系统拆分为 4 个核心限界上下文（Bounded Context）：

| 限界上下文 | 职责 | 典型模块 |
| --- | --- | --- |
| `Sketch Preparation Context` | 从 CAD 序列和约束字典中提取领域对象 | `constraint_extractor.py`, `constraint_tokenizer.py` |
| `Constraint-Fused Encoding Context` | 将命令与约束联合编码为约束感知 latent | `constraint_tag_embedding.py`, `constraint_token_encoder.py`, `encoder_fused.py` |
| `Generation Context` | 基于 \(z\) 重建或生成 CAD 命令序列 | `decoder`, `bottleneck`, `latent generator` |
| `Training Orchestration Context` | 组织训练、评估、日志、配置与实验生命周期 | `train.py`, `trainer/`, `evaluate/` |

这样划分的原因是：**数据准备、领域建模、生成推理、训练编排**在概念与变化频率上不同，强行混写会导致模型演进时频繁牵连整个系统。

### 2.2 上下文映射（Context Map）

```text
Sketch Preparation Context
        │ 输出：SketchSequenceAggregate / ConstraintGraph / TrainingBatch
        ▼
Constraint-Fused Encoding Context
        │ 输出：ConstraintAwareLatent
        ▼
Generation Context
        │ 输出：DecodedCadSequence / ConstraintPrediction
        ▼
Training Orchestration Context
        │ 负责 loss 组合、反向传播、评估、日志、checkpoint
        └───────────────────────────────────────────────┐
                                                        │
                     评估反馈、实验配置、指标回写 ────────┘
```

上下文协作规则如下：

1. `Sketch Preparation Context` 不关心模型结构，只负责构造干净、可消费的领域数据。
2. `Constraint-Fused Encoding Context` 不直接负责训练步骤，只负责“如何把命令与约束转成 \(z\)”。
3. `Generation Context` 只消费 `ConstraintAwareLatent`，不依赖约束字典原始格式。
4. `Training Orchestration Context` 只编排，不承担具体的领域计算细节。

### 2.3 分层架构

每个上下文内部统一采用 DDD 常见四层：

| 层级 | 责任 | 在本方案中的体现 |
| --- | --- | --- |
| `Domain Layer` | 领域对象、领域服务、领域规则 | 约束图、命令序列、融合规则、约束重建规则 |
| `Application Layer` | 用例编排、事务边界、流程控制 | 训练一个 batch、推理一个 latent、评估一个样本 |
| `Infrastructure Layer` | 数据加载、配置、日志、模型持久化 | dataset、repo、checkpoint、tensorboard |
| `Interface Layer` | 对外输入输出、脚本入口 | `train.py`, `evaluate/*.py`, 配置文件 |

### 2.4 核心信息流

```text
CAD cmds/args + constraint dict
        │
        ▼
[Preparation Context]
ConstraintExtractor
ConstraintTokenizer
TrainingBatchAssembler
        │
        ▼
[Encoding Context]
CADEmbeddingFused
ConstraintTokenEncoder
JointTransformerEncoder
ConstraintAwarePooling
Bottleneck（与 DeepCAD 对齐，建议与 Encoder 解耦调用）
        │
        ▼
ConstraintAwareLatent z
        │
        ├──────────────► [Generation Context] Decoder → CAD Sequence
        │
        └──────────────► [Constraint Reconstruction] unary/pair heads
        │
        ▼
[Training Orchestration Context]
LossComposer + Trainer + Evaluator
```

### 2.4.1 与主技术方案一致的信息流（编码器约束融合）

下列框图与《Constraint-Fused DeepCAD 技术方案》§2.1 对齐，强调 **约束在嵌入层与编码器联合序列即参与计算**，\(z\) 聚合命令与约束 token；解码器侧 Cross-Attn 为**可选增强**。

```text
                    ┌──────────────────────────────────────┐
                    │            Encoder（约束融合）           │
  CAD cmds/args ──► │ CADEmbedding + constraint_tags         │
  constraint dict ─►│ + ConstraintTokenEncoder + SegmentEmb  │
                    │        Self-Attn ×L（联合序列）         │
                    │        Constraint-Aware Pooling        │
                    │              Bottleneck                │
                    └─────────────────┬────────────────────┘
                                      z（约束感知）
                                      │
                    ┌─────────────────▼────────────────────┐
                    │  Decoder（与 DeepCAD 同构或可增强）      │
                    │  Self-Attn → Global-Inject(z) → FFN    │
                    │  （可选）Cross-Attn(C)，推理时可关闭       │
                    └─────────────────┬────────────────────┘
                                      │
              Cmd/Arg Head +（可选）Constraint Pred Head
```

### 2.4.2 基线 Constraint-Aware：数据与训练管线（复用前半段）

本方案**复用**「JSON → 提取 → 约束字典 / 约束 token」；差异在于 Encoder 内融合与 \(z\) 重建，Decoder 对 \(C\) 的依赖降为可选。

```mermaid
flowchart LR
  subgraph offline["离线（一次性）"]
    JSON[DeepCAD JSON 命令]
    EX[约束提取]
    G[约束图 / 约束字典]
    JSON --> EX --> G
  end
  subgraph train["训练（基线概念）"]
    DS[Dataset 加载命令 + 约束]
    ENC[Encoder]
    DEC[Decoder + 约束 Cross-Attn]
    L1[L_cmd]
    L2[L_constraint]
    DS --> ENC --> DEC
    DEC --> L1
    DEC --> L2
  end
  G --> DS
```

**基线模型侧信息流（对照）**：输入为 CAD 命令序列；并行支路将约束经 **Constraint Embedding** 得到 \(C\)；解码器隐状态对 \(C\) 做 **Cross-Attention**；损失为 `L_total = L_cmd + α * L_constraint`。Fused 方案中 \(C\) 同时进入 Encoder 联合序列，并增加 `L_constraint_recon`（及可选 `L_constraint_pred`）。

### 2.5 DATA SHAPES

| 符号 | 含义 | 典型形状 |
| --- | --- | --- |
| \(S\) | 命令序列长度 | `(S, N)` |
| \(N\) | batch size | `scalar` |
| \(T_c\) | 约束 token 数 | `(T_c, N)` |
| `d_model` | 模型通道数 | `256` |
| `constraint_tags` | 命令级约束标记 | `(S, N, 5)` |
| `E_cmd` | 命令 embedding | `(S, N, d_model)` |
| `E_con` | 约束 token embedding | `(T_c, N, d_model)` |
| `E_joint` | 联合序列 embedding | `(S + T_c, N, d_model)` |
| `mask_joint` | 联合 padding mask | `(N, S + T_c)` |
| `memory` | 编码器输出 | `(S + T_c, N, d_model)` |
| \(z\) | 约束感知 latent | `(1, N, d_model)` |
| `unary_gt` | 一元约束真值 | `(N, max_lines, 2)` |
| `pair_gt` | 二元约束真值 | `(N, max_lines, max_lines, 3)` |

**张量符号对照**：基线 Constraint-Aware 文档常用 batch 维 `B`、命令长度 `T`；DeepCAD 实现约定多为序列长度 `S`、batch `N`。含义对应为 \(T \leftrightarrow S\)、\(B \leftrightarrow N\)。

### 2.6 与原方案差异

| 维度 | 原方案 | DDD 详细设计后 |
| --- | --- | --- |
| 关注点 | 主要是模型结构 | 同时覆盖模型、领域对象、流程、接口与落地 |
| 约束主入口 | 解码器为主 | 编码器为主，解码器仅可选增强 |
| 数据角色 | 张量为主 | 张量背后有明确领域语义对象 |
| 训练逻辑 | 容易耦合在脚本 | 由应用服务统一编排 |
| 工程扩展性 | 改动范围大 | 可替换单个上下文或服务 |

### 2.7 与原 Constraint-Aware 思路对比（摘要）

| 维度 | 原 Constraint-Aware 思路 | Constraint-Fused（本文） |
| --- | --- | --- |
| 约束主注入位置 | 解码器 Cross-Attn | 编码器 Embedding + 联合 Self-Attn |
| \(z\) 是否含约束 | 否 | 是 |
| Latent GAN 推理 | 难依赖外部 \(C\) | 仅需 \(z\) |
| 解码器 | 常需 Cross-Attn | 可保持原版；Cross-Attn **可选** |

---

## 3. 领域模型设计

### 3.1 聚合根：`SketchSequenceAggregate`

#### 模块作用

作为核心聚合根，统一管理一个样本中的命令序列、约束关系、命令级标记、约束 token、重建标签以及掩码信息，确保训练和推理看到的是**同一个领域对象**的不同视图。

#### 模块原理

聚合根封装以下不变量：

1. 命令序列与线段索引映射必须一致。
2. 每个 `ConstraintRelation` 引用的线段索引必须落在 `max_lines` 范围内。
3. `constraint_tags` 必须可由 `ConstraintGraph` 推导得到，而不是自由构造。
4. `constraint_tokens`、`unary_gt`、`pair_gt` 必须与同一约束图保持一致。

#### 代码

```python
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class CadCommand:
    command_id: int
    args: List[int]
    group_id: Optional[int]
    line_ref: Optional[int] = None


@dataclass
class ConstraintRelation:
    type_id: int
    line_a: int
    line_b: int


@dataclass
class SketchSequenceAggregate:
    commands: List[CadCommand]
    constraints: List[ConstraintRelation]
    constraint_tags: "Tensor"
    constraint_tokens: "Tensor"
    unary_gt: "Tensor"
    pair_gt: "Tensor"

    def validate(self, max_lines: int) -> None:
        for rel in self.constraints:
            assert 0 <= rel.line_a < max_lines
            assert 0 <= rel.line_b < max_lines
```

#### 举例说明

一个样本包含 40 条命令，其中第 2、5、7 条命令各对应一条线段；约束字典中存在 `PARALLEL(0, 2)`。则该约束既会体现在 `constraints` 中，也会被投影为命令级 `constraint_tags`，并进一步衍生出 `constraint_tokens` 与 `pair_gt`。训练时不应分别从多个脚本“重复推导”这些结果，而应由聚合统一持有。

---

### 3.2 实体：`CadCommand`

#### 模块作用

表达一条 CAD 命令及其参数，是整个系统最基础的领域实体，也是与原 DeepCAD 格式兼容的核心载体。

#### 模块原理

`CadCommand` 不是普通 token，它携带三个层面的信息：

1. **语法语义**：命令类型与参数。
2. **几何语义**：是否对应草图线段或轮廓元素。
3. **约束锚点语义**：是否可被约束关系引用。

#### 代码

```python
@dataclass
class CadCommand:
    command_id: int
    args: List[int]
    group_id: Optional[int] = None
    line_ref: Optional[int] = None

    @property
    def is_line_command(self) -> bool:
        return self.line_ref is not None
```

#### 举例说明

`LINE(start_x, start_y, end_x, end_y)` 可映射为 `line_ref=3`，表示它在该样本的线段索引空间中编号为 3。后续 `PARALLEL(3, 8)`、`VERTICAL(3)` 等约束都通过这个锚点与命令实体关联。

---

### 3.3 实体：`ConstraintRelation`

#### 模块作用

统一表达一元与二元约束，是约束图中的基本边或节点属性，用于驱动 tag 构造、token 编码和重建监督。

#### 模块原理

约束关系可分为两类：

1. **Unary 约束**：如水平、竖直，可视为线段节点属性。
2. **Pair 约束**：如平行、垂直、共线，可视为线段节点之间的边。

当前详细设计中，**真实约束类型共 5 类**：`HORIZONTAL`、`VERTICAL`、`PARALLEL`、`PERPENDICULAR`、`COLLINEAR`。此外保留一个 `NONE` 类型，仅用于约束 token 的 padding / 占位，不参与命令级 `ConstraintTagVector` 的五维定义。

#### 代码

```python
class ConstraintType:
    HORIZONTAL = 0
    VERTICAL = 1
    PARALLEL = 2
    PERPENDICULAR = 3
    COLLINEAR = 4
    NONE = 5


@dataclass
class ConstraintRelation:
    type_id: int
    line_a: int
    line_b: int = 0

    def is_unary(self) -> bool:
        return self.type_id in {ConstraintType.HORIZONTAL, ConstraintType.VERTICAL}
```

#### 举例说明

`VERTICAL(线3)` 在实现上可表示为 `ConstraintRelation(type_id=1, line_a=3, line_b=3)`，这样 token 编码、padding 与 pair/unary 映射逻辑能够统一处理，而不需要为 unary 和 pair 维护两套完全不同的数据通道。

---

### 3.4 值对象：`ConstraintTagVector`、`ConstraintAwareLatent`

#### 模块作用

值对象不强调身份，强调语义稳定性。这里用它们表达“命令是否参与某类约束”和“已经携带约束语义的 latent code”。

#### 模块原理

1. `ConstraintTagVector` 是命令级弱约束表示，重点是告诉 embedding 该命令处于什么约束语境中。
2. `ConstraintAwareLatent` 是聚合后的强约束表示，重点是为解码与采样提供统一输入。

其中 `ConstraintTagVector` 固定为 **5 维真实约束语义**；token 级约束词表则为 **5 类真实约束 + 1 类 `NONE/PAD`**，二者职责不同，不混用。

#### 代码

```python
@dataclass(frozen=True)
class ConstraintTagVector:
    horizontal: int
    vertical: int
    parallel: int
    perpendicular: int
    collinear: int


@dataclass(frozen=True)
class ConstraintAwareLatent:
    tensor: "Tensor"  # (1, N, d_model)
```

#### 举例说明

命令 3 若同时参与水平与平行关系，则其 tag 可表达为 `[1, 0, 1, 0, 0]`。经过联合编码与池化后，该局部信息会演化为全局 latent 中的一部分，成为 `ConstraintAwareLatent`。

---

### 3.5 领域服务：`ConstraintFusionDomainService`

#### 模块作用

定义“命令流 + 约束流如何融合”的核心业务规则，是本方案最重要的领域服务之一。

#### 模块原理

该服务不关心训练循环，只关心一件事：如何把 **DataLoader 产出的张量 batch**（或与 `SketchSequenceAggregate` 等价的 collate 视图）转成约束感知表示。其步骤包括：

1. 构造命令 embedding。
2. 构造命令级约束 tag embedding。
3. 构造约束 token embedding。
4. 拼接联合序列并加 segment embedding。
5. 通过 joint transformer 编码。
6. 先以 `ConstraintAwarePooling` 聚合得到 `z_pre`（与主方案 §3.3 策略 A/B 一致）。
7. 再经 `BottleneckAdapter`（**建议与 `EncoderFused` 类解耦**，与原版 DeepCAD 训练脚本对齐）得到与 decoder 兼容的 `ConstraintAwareLatent`。

#### 代码

```python
class ConstraintFusionDomainService:
    def __init__(self, encoder_fused, bottleneck):
        self.encoder_fused = encoder_fused
        self.bottleneck = bottleneck

    def fuse(self, batch_tensors):
        """batch_tensors：commands, args, constraint_tags, c_types, c_line_a, c_line_b,
        cmd_padding_mask, constraint_padding_mask, groups(optional) 等与 EncoderFused.forward 一致。"""
        z_pre = self.encoder_fused(**batch_tensors)
        z = self.bottleneck(z_pre)
        return ConstraintAwareLatent(z)
```

#### 举例说明

对同一个样本，如果不启用 `ConstraintFusionDomainService`，则命令与约束仅在 decoder 中相遇；启用后，它们在 encoder 阶段就被统一放入同一个注意力空间，最终约束语义被吸收进 \(z\)。

---

### 3.6 领域服务：`ConstraintReconstructionDomainService`

#### 模块作用

负责从 \(z\) 重建约束图，并输出约束重建监督结果，用于保证瓶颈层确实保存了约束语义。

#### 模块原理

它将“约束是否被编码成功”从隐含假设变成显式任务。只要重建失败，梯度就会持续施压 encoder 与 bottleneck。

#### 代码

```python
class ConstraintReconstructionDomainService:
    def __init__(self, recon_head):
        self.recon_head = recon_head

    def reconstruct(self, latent):
        unary_pred, pair_pred = self.recon_head(latent.tensor.squeeze(0))
        return unary_pred, pair_pred
```

#### 举例说明

若样本中存在 `PERPENDICULAR(线2, 线5)`，但 `pair_pred[2,5,1]` 长期无法接近 1，则说明当前 \(z\) 没有稳定保存该关系，系统会通过 `L_constraint_recon` 强化编码器对该约束的表达。

---

### 3.7 仓储接口：`SketchRepository` 与 `ModelCheckpointRepository`

#### 模块作用

仓储用于屏蔽底层数据存取细节，使应用层不直接依赖文件结构、pickle、npz、checkpoint 命名规则等基础设施细节。

#### 模块原理

1. `SketchRepository` 输出 `SketchSequenceAggregate`。
2. `ModelCheckpointRepository` 管理权重版本、最优模型与恢复训练。

#### 代码

```python
class SketchRepository:
    def load(self, sample_id: str) -> SketchSequenceAggregate:
        raise NotImplementedError


class ModelCheckpointRepository:
    def save(self, epoch: int, state: dict) -> None:
        raise NotImplementedError

    def load_latest(self) -> dict:
        raise NotImplementedError
```

#### 举例说明

如果后续把数据来源从本地离线文件切换为 LMDB 或 parquet，应用层与领域层都不需要修改，只替换 `SketchRepository` 的基础设施实现即可。

---

## 4. 模块架构

以下模块均按“模块作用、模块原理、代码、举例说明”展开，并映射到 DDD 分层。

---

### 4.1 模块：离线约束提取与标准化（与主方案 §3.0 对齐）

#### 模块作用

位于 `Sketch Preparation Context`，从 DeepCAD **JSON 命令序列**解析几何实体（线段等），用统一数值阈值自动判定五类关系，输出**结构化约束字典**；再经 tokenizer / collate 转为 `ConstraintRelation`、`constraint_tags`、约束 token、`unary_gt` / `pair_gt` 及 mask。全程**无需人工标注**。本模块是相对 JSON 格式的**防腐层（Anti-Corruption Layer）**。

#### 模块原理

1. **输入**：单条样本的 DeepCAD 命令 JSON（与官方格式一致）。
2. **中间**：执行命令语义等价的几何构造，得到带 id 的线段及端点、方向向量。
3. **判定**（统一阈值，与主方案一致）：
   - **水平**：方向向量 \(z\) 分量近似为 0。
   - **竖直**：\(x、y\) 分量近似为 0（视具体坐标约定与 DeepCAD 线向定义一致微调）。
   - **平行 / 垂直**：两线方向向量夹角与 \(0^\circ\) 或 \(90^\circ\) 比较。
   - **共线**：先满足近似平行，再判一点到另一直线的距离小于阈值。
4. **输出**：五类约束的列表或线对列表 → 映射为领域对象与张量标签。

翻译规则（collate 侧）：

1. 约束字典映射为 `ConstraintRelation` 列表。
2. 按 `line_ref` 将关系投影为命令级 `constraint_tags`（五维参与向量）。
3. 关系离散为约束 token 序列，供 Encoder 联合 Self-Attn。
4. 派生 `unary_gt` / `pair_gt` 供 `L_constraint_recon`。

#### 代码

**约束字典结构（输出 schema，与主方案一致）：**

```json
{
  "horizontal": [线ID列表],
  "vertical": [线ID列表],
  "parallel": [[线A, 线B], ...],
  "perpendicular": [[线A, 线B], ...],
  "collinear": [[线A, 线B], ...]
}
```

**阈值与线段表示（提取器内部）：**

```python
# 阈值（论文常用量级）
ANGLE_THRESH = 2.0       # 角度误差 < 2°
DIST_THRESH = 1e-3       # 距离误差 < 0.001
EPS = 1e-5

# 线段表示（示例）
line = {
    "id": int,
    "start": (x, y, z),
    "end": (x, y, z),
    "vec": (dx, dy, dz),
}
```

**判定要点（伪代码级，与主方案一致）：**

```python
import math

def angle_deg(u, v):
    cos_t = abs(dot(u, v)) / (norm(u) * norm(v) + 1e-12)
    cos_t = min(1.0, max(-1.0, cos_t))
    return math.degrees(math.acos(cos_t))

# 水平：abs(line.vec.z) < EPS
# 竖直：abs(line.vec.x) < EPS and abs(line.vec.y) < EPS
# 平行：angle_deg(v1, v2) < ANGLE_THRESH
# 垂直：abs(angle_deg(v1, v2) - 90.0) < ANGLE_THRESH
# 共线：平行 + 点 p3 到直线 (p1,p2) 距离 < DIST_THRESH
```

**领域组装（示意）：**

```python
class ConstraintExtractor:
    def build_relations(self, raw_constraint_dict):
        relations = []
        for item in raw_constraint_dict:
            relations.append(
                ConstraintRelation(
                    type_id=item["type_id"],
                    line_a=item["line_a"],
                    line_b=item.get("line_b", item["line_a"]),
                )
            )
        return relations


class ConstraintBatchAssembler:
    def assemble(self, commands, relations, max_lines, max_constraints):
        constraint_tags = build_constraint_tags(commands, relations)
        constraint_tokens = build_constraint_tokens(relations, max_constraints)
        unary_gt, pair_gt = build_recon_targets(relations, max_lines)
        return SketchSequenceAggregate(
            commands=commands,
            constraints=relations,
            constraint_tags=constraint_tags,
            constraint_tokens=constraint_tokens,
            unary_gt=unary_gt,
            pair_gt=pair_gt,
        )
```

#### 举例说明

假设解析后有三条线：`L0` 沿 x 轴，`L1` 沿 y 轴，`L2` 与 `L0` 同向且共线。则可能得到：`horizontal: [0, 2]`，`vertical: [1]`，`perpendicular: [[0, 1], ...]`，`collinear: [[0, 2]]`。该字典经 `constraint_tokenizer` / `collate` 转为 `constraint_tags`、`c_types/c_line_a/c_line_b` 及重建 GT。

原始字段若包含 `{type: "parallel", entities: [1, 4]}`，则转换为 `ConstraintRelation(type_id=2, line_a=1, line_b=4)`；命令中 `line_ref` 为 1、4 的位置在 tag 中 `parallel=1`，并生成对应 token 与 `pair_gt[1,4,0]=1`（维序与主方案一致即可）。

---

### 4.2 模块：命令级约束标记嵌入

#### 模块作用

在命令 embedding 阶段注入局部约束先验，使模型从一开始就知道某条命令所处的约束语境。

#### 模块原理

每条命令对应一个五维参与向量 \(p_i \in \{0,1\}^5\)，通过 MLP 投影到 `d_model` 后，与原始命令 embedding 相加。

五维定义如下：

| 维度 | 语义 |
| --- | --- |
| 0 | 水平 |
| 1 | 竖直 |
| 2 | 平行 |
| 3 | 垂直 |
| 4 | 共线 |

#### 代码

```python
class ConstraintTagEmbedding(nn.Module):
    def __init__(self, n_constraint_types=5, d_model=256):
        super().__init__()
        self.tag_proj = nn.Sequential(
            nn.Linear(n_constraint_types, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, constraint_tags):
        """
        constraint_tags: (S, N, 5) float，约束参与向量
        return: (S, N, d_model)
        """
        return self.tag_proj(constraint_tags)


class CADEmbeddingFused(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.command_embed = nn.Embedding(cfg.n_commands, cfg.d_model)
        self.arg_embed = nn.Embedding(cfg.args_dim + 1, 64, padding_idx=0)
        self.embed_fcn = nn.Linear(64 * cfg.n_args, cfg.d_model)
        self.pos_encoding = PositionalEncodingLUT(cfg.d_model, max_len=cfg.max_len)
        self.use_group = cfg.use_group_emb
        if self.use_group:
            self.group_embed = nn.Embedding(cfg.group_len + 2, cfg.d_model)
        self.constraint_tag = ConstraintTagEmbedding(5, cfg.d_model)

    def forward(self, commands, args, groups=None, constraint_tags=None):
        src = self.command_embed(commands.long()) + self.embed_fcn(
            self.arg_embed((args + 1).long()).view(commands.shape[0], commands.shape[1], -1)
        )
        if self.use_group and groups is not None:
            src = src + self.group_embed(groups.long())
        if constraint_tags is not None:
            src = src + self.constraint_tag(constraint_tags)
        return self.pos_encoding(src)
```

融合顺序（与主方案 §3.1 一致，在 `pos_encoding` 之前）：`command + args [+ group] + constraint_tag` → `pos_encoding`。

#### 举例说明

如果命令 3 对应一条水平线，且它与命令 7 对应线段满足平行关系，那么命令 3 的 tag 可设为 `[1,0,1,0,0]`。**extrude** 等非线段命令则令 \(p_i = \mathbf{0}\)，避免引入虚假约束信号（主方案 §3.1）。

---

### 4.3 模块：约束 Token 编码与联合序列构造

#### 模块作用

把“约束是什么关系”显式编码为 token，使模型不仅知道一条线“有约束”，还知道“它和谁构成了何种关系”。与主方案 §3.2 一致：命令级 tag 不表达关系结构，约束 token 序列在联合 Self-Attn 中补全「线 A–线 B–关系类型」细粒度依赖。

#### 模块原理

1. `type_embed` 编码约束类型。
2. `line_embed` 编码参与关系的线段索引。
3. `pair_fuse` 融合线对语义。
4. `segment_embed` 区分命令段与约束段。
5. 命令序列与约束序列在 encoder 入口拼接；`key_padding_mask` 覆盖命令 pad 与约束 pad。

编码器层可与原版一致（如 `TransformerEncoderLayerImproved`），**不新增层类型**，仅序列长度变为 \(S + T_c\)。

**与基线 Constraint Embedding 的对应（主方案 §3.2）**：基线将约束字典转为 \(C \in \mathbb{R}^{B \times T_c \times d_\text{model}}\)，一条实例一个 token；Fused 中 `ConstraintTokenEncoder` 语义等价，但 token **进入 Encoder 联合序列**。若需与旧实现对齐，可采用基线风格 `ConstraintEmbedding`（`pair_mask` 区分单线/线对）：

```python
import torch
import torch.nn as nn

class ConstraintEmbedding(nn.Module):
    def __init__(self, n_types, n_line_ids, d_model, d_ent=64):
        super().__init__()
        self.type_emb = nn.Embedding(n_types, d_model)
        self.line_emb = nn.Embedding(n_line_ids, d_ent)
        self.pair_mlp = nn.Sequential(
            nn.Linear(2 * d_ent, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        self.single_proj = nn.Linear(d_ent, d_model)
        self.out = nn.LayerNorm(d_model)

    def forward(self, type_ids, line_a, line_b, pair_mask):
        et = self.type_emb(type_ids)
        ea, eb = self.line_emb(line_a), self.line_emb(line_b)
        ep = self.pair_mlp(torch.cat([ea, eb], dim=-1))
        es = self.single_proj(ea)
        fused = torch.where(pair_mask.unsqueeze(-1), ep, es)
        return self.out(et + fused)
```

`ConstraintTokenEncoder` 的 `n_types=6`：`HORIZONTAL=0, VERTICAL=1, PARALLEL=2, PERPENDICULAR=3, COLLINEAR=4, NONE=5`。**`NONE/PAD`** 仅用于定长 batch，不作为真实监督统计。

#### 代码

```python
class ConstraintTokenEncoder(nn.Module):
    def __init__(self, n_types=6, max_lines=64, d_model=256):
        super().__init__()
        self.type_embed = nn.Embedding(n_types, d_model)
        self.line_embed = nn.Embedding(max_lines, d_model // 2)
        self.pair_fuse = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU())
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, c_types, c_line_a, c_line_b):
        e_type = self.type_embed(c_types)
        e_a = self.line_embed(c_line_a)
        e_b = self.line_embed(c_line_b)
        e_lines = self.pair_fuse(torch.cat([e_a, e_b], dim=-1))
        return self.norm(self.out_proj(e_type + e_lines))


class SegmentEmbedding(nn.Module):
    def __init__(self, d_model=256):
        super().__init__()
        self.embed = nn.Embedding(2, d_model)  # 0: command, 1: constraint

    def forward(self, seg_ids):
        return self.embed(seg_ids)
```

拼接与 mask（与主方案一致）：

```text
E_joint = cat([E_cmd, E_con], dim=0) + SegmentEmbedding(cat([seg_cmd, seg_con]))
mask_joint = cat([cmd_padding_mask, constraint_padding_mask], dim=1)
memory = Encoder(E_joint, src_key_padding_mask=mask_joint)
```

#### 举例说明

样本中若同时存在 `PARALLEL(1,5)` 与 `PERPENDICULAR(2,5)`，则两个 constraint token 会在同一注意力空间中连接到命令 1、2、5；模型能够学习“线 5 是多个关系的共享节点”这一结构信息。

**基线-only 补充**：某样本约束 token 顺序为 `[VERT(L3), HORIZ(L1), PARALLEL(L1,L4)]`，则 \(T_c=3\)；batch 内另一样本约束较少时 pad 到 \(T_c^{\max}\)，padding 在 `key_padding_mask` 为 `True`（PyTorch MHA 约定）。Fused 下 \(C\) 同时进入 Encoder 联合序列。

---

### 4.4 模块：融合编码器 `EncoderFused`

#### 模块作用

把命令 embedding（含 tag）、约束 token、segment embedding、联合 Self-Attention 与**池化**串联为单一编码路径，输出 **`z_pre`（池化后、Bottleneck 前）**。与主方案 §3.6 一致：**Bottleneck 建议放在类外**（由 `ConstraintFusionDomainService` 或训练脚本调用），以便与原版 DeepCAD `Bottleneck` 模块对齐。

#### 模块原理

`EncoderFused` 只承担 **Encoder 侧融合编码**，不承担 Bottleneck、Decoder 与损失。

流程如下：

1. 构造 `E_cmd`（含可选 `groups` 与 `constraint_tags`）。
2. 构造 `E_con`。
3. 构造 `seg_ids` 与 `mask_joint`。
4. 拼接为 `E_joint`。
5. 经 `TransformerEncoder` 得到 `memory`。
6. 经 `ConstraintAwarePooling`（默认 masked mean，主方案策略 A）得到 `z_pre` 并返回。

#### 代码

```python
class EncoderFused(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.embedding = CADEmbeddingFused(cfg)
        self.constraint_token_enc = ConstraintTokenEncoder(6, cfg.max_lines, cfg.d_model)
        self.segment_embed = SegmentEmbedding(cfg.d_model)
        enc_layer = TransformerEncoderLayerImproved(
            cfg.d_model,
            cfg.n_heads,
            cfg.dim_feedforward,
            cfg.dropout,
        )
        self.encoder = TransformerEncoder(
            enc_layer,
            cfg.n_layers,
            nn.LayerNorm(cfg.d_model),
        )

    def forward(
        self,
        commands,
        args,
        constraint_tags,
        c_types,
        c_line_a,
        c_line_b,
        cmd_padding_mask,
        constraint_padding_mask,
        groups=None,
    ):
        e_cmd = self.embedding(commands, args, groups, constraint_tags)
        e_con = self.constraint_token_enc(c_types, c_line_a, c_line_b)
        s, n, _ = e_cmd.shape
        t_c = e_con.shape[0]

        seg = torch.cat([
            torch.zeros(s, n, dtype=torch.long, device=e_cmd.device),
            torch.ones(t_c, n, dtype=torch.long, device=e_cmd.device),
        ], dim=0)

        e_joint = torch.cat([e_cmd, e_con], dim=0) + self.segment_embed(seg)
        mask_joint = torch.cat([cmd_padding_mask, constraint_padding_mask], dim=1)
        memory = self.encoder(e_joint, src_key_padding_mask=mask_joint)
        valid = (~mask_joint).unsqueeze(-1).transpose(0, 1).float()
        z_pre = (memory * valid).sum(dim=0, keepdim=True) / valid.sum(dim=0, keepdim=True).clamp(min=1e-6)
        return z_pre
```

#### 举例说明

样本 A 的 \(S=40, T_c=10\)，样本 B 的 \(S=25, T_c=32\)（约束截断上限）。`mask_joint` 分别屏蔽各自 padding，池化只对真实命令与真实约束 token 求平均，避免把 padding 当零约束「稀释」\(z\)（主方案 §3.6）。

---

### 4.5 模块：约束感知池化与 Bottleneck

#### 模块作用

将联合序列输出压缩为固定长度的全局表示，再经 Bottleneck 得到供解码器 Global-Inject 使用的 \(z\)。池化逻辑可在 `EncoderFused` 内联（主方案策略 A）或抽取为 `MaskedMeanPooling` 模块；**Bottleneck** 与原版 DeepCAD 对齐，放在 **Encoding 上下文出口**（领域服务或 `train` 编排处），避免与 `EncoderFused` 强耦合。

#### 模块原理

推荐分两级：

1. **池化**：从 `memory` 与 `mask_joint` 得到 `z_pre`。
2. **BottleneckAdapter**：与现有 DeepCAD `Bottleneck` 一致，将 `z_pre` 映射为 decoder 期望的 `ConstraintAwareLatent`。

池化支持两种策略（主方案 §3.3）：

1. **策略 A（推荐起步）**：masked mean pooling，**默认**；约束 token 自然参与 \(z\)。
2. **策略 B（可选）**：双流门控 `DualStreamPooling`，在「约束过强压制几何细节」时做消融。

#### 代码

```python
class MaskedMeanPooling(nn.Module):
    def forward(self, memory, mask_joint):
        valid = (~mask_joint).transpose(0, 1).unsqueeze(-1).float()
        return (memory * valid).sum(dim=0, keepdim=True) / valid.sum(dim=0, keepdim=True).clamp(min=1e-6)


class DualStreamPooling(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid(),
        )

    def forward(self, memory, cmd_mask, con_mask):
        z_cmd = (memory * cmd_mask).sum(0) / cmd_mask.sum(0).clamp(min=1)
        z_con = (memory * con_mask).sum(0) / con_mask.sum(0).clamp(min=1)
        gate = self.gate(torch.cat([z_cmd, z_con], dim=-1))
        z = gate * z_cmd + (1 - gate) * z_con
        return z.unsqueeze(0)


class BottleneckAdapter(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.proj(z)
```

#### 举例说明

当前主实现先采用 masked mean pooling，优先保证编码路径简单、稳定且与原始 DeepCAD 接口对齐；若实验发现约束 token 占比过高导致几何细节被抹平，再切换到策略 B，让门控自动调节几何流与约束流比例（主方案 §3.3）。

---

### 4.6 模块：约束重建头与辅助损失

#### 模块作用

从 \(z\) 反向恢复约束图，用辅助损失迫使瓶颈层真正编码约束，而不是只让 decoder 在局部路径上“临时利用约束”。

#### 模块原理

重建任务拆分为：

1. `UnaryReconHead`：预测每条线是否水平/竖直。
2. `PairReconHead`：预测线对之间是否平行/垂直/共线。
3. `WeightedBCELoss`：解决稀疏标签不平衡问题。

#### 代码

```python
class ConstraintReconHead(nn.Module):
    def __init__(self, dim_z=256, max_lines=64):
        super().__init__()
        self.max_lines = max_lines
        self.unary_head = nn.Sequential(
            nn.Linear(dim_z, 256),
            nn.GELU(),
            nn.Linear(256, max_lines * 2),
        )
        self.pair_head = nn.Sequential(
            nn.Linear(dim_z, 512),
            nn.GELU(),
            nn.Linear(512, max_lines * max_lines * 3),
        )

    def forward(self, z):
        n = z.size(0)
        unary = self.unary_head(z).view(n, self.max_lines, 2)
        pair = self.pair_head(z).view(n, self.max_lines, self.max_lines, 3)
        return torch.sigmoid(unary), torch.sigmoid(pair)


def weighted_bce(pred, target, pos_weight=5.0):
    weight = torch.where(target > 0.5, pos_weight, 1.0)
    return F.binary_cross_entropy(pred, target, weight=weight)
```

#### 举例说明

若 `pair_gt[3,8,0]=1` 表示线 3 与线 8 平行，但模型始终无法在 `pair_pred[3,8,0]` 上给出高响应，那么训练将持续把这部分误差传回 encoder，使瓶颈不得不保留相关约束语义。

---

### 4.7 模块：解码器、可选约束 Cross-Attention 与约束预测头

#### 模块作用

在 **\(z\) 已约束感知** 的前提下（主方案 §3.5）：解码器可**完全沿用 DeepCAD**（Self-Attn → Global-Inject(\(z\)) → FFN），保证与 Latent GAN 一致；若需更强训练期监督，可**可选**加入对约束 memory \(C\) 的 Cross-Attn，**推理时必须不依赖 \(C\)**（关闭模块、权重置零或 `training_dropout→1`）。同时可用 **约束预测头** 提供 `L_constraint_pred`。

#### 模块原理

1. **基础路径**：`linear_global(z)` 将约束语义注入每层解码器，推理只需 \(z\)。
2. **可选增强路径**：训练时以一定概率对约束序列做 Cross-Attn；配合 dropout schedule，逐步提高「仅用 \(z\)」的比例，减轻 train–infer 差异（主方案 §3.5 `OptionalConstraintCrossAttn`）。
3. **基线对照**：Constraint-Aware 中解码器隐状态 \(H\) 为 Q，\(C\) 为 K/V 的标准 MHA；本方案中该通路为**可选**，用于 Phase 3 消融而非默认必需。
4. **辅助监督**：`ConstraintPredHead` 对 decoder 隐状态预测约束相关标签，与交叉注意力的隐式融合互补。

**基线 Cross-Attn 块（与主方案一致，供 Generation 上下文基础设施实现）：**

```python
import torch.nn as nn

class ConstraintCrossAttentionBlock(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mha = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.drop = nn.Dropout(dropout)

    def forward(self, h, c, constraint_key_padding_mask):
        q = self.norm(h)
        k = self.norm(c)
        attn_out, _ = self.mha(q, k, k, key_padding_mask=constraint_key_padding_mask)
        return h + self.drop(attn_out)
```

**可选 Cross-Attn + 训练期随机跳过（主方案）：**

```python
class OptionalConstraintCrossAttn(nn.Module):
    def __init__(self, d_model, nhead, training_dropout=0.5):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, nhead)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(training_dropout)
        self.training_dropout = training_dropout

    def forward(self, tgt, constraint_memory, constraint_mask=None):
        if not self.training:
            return tgt
        if torch.rand(1).item() < self.training_dropout:
            return tgt
        t2, _ = self.cross_attn(
            self.norm(tgt), constraint_memory, constraint_memory,
            key_padding_mask=constraint_mask,
        )
        return tgt + self.dropout(t2)
```

```python
class ConstraintPredHead(nn.Module):
    def __init__(self, d_model, n_constraint_types):
        super().__init__()
        self.proj = nn.Linear(d_model, n_constraint_types)

    def forward(self, h_step):
        return self.proj(h_step)
```

#### 代码（适配器：默认 latent-only，可选挂接 Cross-Attn）

```python
class ConstraintAwareDecoderAdapter(nn.Module):
    def __init__(self, decoder, constraint_pred_head, optional_cross_attn=None):
        super().__init__()
        self.decoder = decoder
        self.constraint_pred_head = constraint_pred_head
        self.optional_cross_attn = optional_cross_attn

    def forward(self, latent, constraint_memory=None, constraint_mask=None, targets=None):
        hidden_states, cmd_logits = self.decoder(latent, targets=targets)
        if self.optional_cross_attn is not None and constraint_memory is not None:
            hidden_states = self.optional_cross_attn(
                hidden_states, constraint_memory, constraint_mask
            )
        constraint_pred_logits = self.constraint_pred_head(hidden_states)
        return cmd_logits, constraint_pred_logits
```

> **实现说明**：上表适配器将 `optional_cross_attn` 接在 decoder 一次前向之后，仅为**接口占位**。与主方案及 DeepCAD 解码器层序一致的生产实现应为每层内 `Self-Attn → Global-Inject(z) → [可选 Cross-Attn(C)] → FFN`；Cross-Attn 应作为解码子模块插入层栈，而非在整个 decoder 末尾单次调用。

#### 举例说明

**阶段 1**：`training_dropout=0`，模型强烈依赖 \(C\) 与 \(z\) 双路信息，约束满足率上升快。**阶段 2**：线性增大 `training_dropout` 至接近 1，推理关闭 Cross-Attn，若生成质量与约束指标仍稳定，说明 \(z\) 已吸收原需 Cross-Attn 传递的约束信息（主方案 §3.5）。

在**仅 latent-only** 配置下，decoder 不访问外部 constraint memory，训练与推理路径一致；`ConstraintPredHead` 仍可从隐状态提供 `L_constraint_pred`，与 `L_constraint_recon` 共同约束「约束可学性」。

---

### 4.8 模块：可微几何约束评估器与几何一致性损失

#### 模块作用

将 decoder 预测的**命令参数分布**可微地恢复为线段几何，并直接对 GT 约束计算几何残差，使“平行 / 垂直 / 共线”等关系不再只停留在离散标签层，而是**真实回拉到输出参数值**。该模块与 `L_constraint_recon` 的职责不同：后者保证 \(z\) “记住约束”，本模块保证预测参数“画出约束”。

#### 模块原理

训练期增加一条几何闭环路径：

```text
cmd/arg logits
    -> soft dequantization
    -> DifferentiableSketchInterpreter
    -> SoftLine(start, end, dir, unit, valid)
    -> DifferentiableConstraintEvaluator
    -> L_geom_constraint
```

其核心思想如下：

1. **不对参数做 argmax**：对每个参数 bin 分布取期望，得到连续值 `arg_soft`，避免离散采样切断梯度。
2. **只解释约束相关几何**：第一阶段仅对 `LINE` 命令恢复 2D 线段，不强求整个 `CADSequence` 全可微解释。
3. **不做硬阈值判定**：不同于离线 `ConstraintExtractor` 的角度阈值/距离阈值，本模块直接输出连续残差。
4. **teacher-forced 起步**：建议先用 GT `command_id` 确定哪些位置是 line，仅让几何损失回拉参数头；稳定后再扩展到命令类型也参与软选择。

设线段端点为 \(p_1, p_2 \in \mathbb{R}^2\)，方向向量 \(d = p_2 - p_1\)，单位向量 \(u = d / (\|d\| + \varepsilon)\)。对两条线段 \(a, b\) 的单位方向向量 \(u, v\)，定义几何残差：

- **水平**：`r_horizontal = u_y^2`
- **竖直**：`r_vertical = u_x^2`
- **平行**：`r_parallel = 1 - (u · v)^2`
- **垂直**：`r_perpendicular = (u · v)^2`
- **共线**：`r_collinear = r_parallel + λ_dist · r_line_distance`

其中 `r_line_distance` 为两条线段端点到对方支撑直线的平均平方距离，用于刻画“平行但不在同一直线”与“真正共线”的差异。

在 batch 内，几何一致性损失定义为：

```text
L_geom_constraint = (1 / |R_valid|) · Σ_k w(type_k) · r_k
```

其中 `R_valid` 为当前样本可计算残差的约束集合，`w(type_k)` 为按约束类型设置的权重或反频率权重。

#### 代码

```python
class DifferentiableSketchInterpreter(nn.Module):
    """
    将 decoder 输出的参数分布恢复为可微线段几何。
    第一阶段仅处理 LINE，命令位置可先由 GT command_id 提供。
    """
    def __init__(self, n_bins, coord_range=(-1.0, 1.0), eps=1e-6):
        super().__init__()
        self.n_bins = n_bins
        self.coord_range = coord_range
        self.eps = eps

    def soft_dequantize(self, arg_logits):
        probs = torch.softmax(arg_logits, dim=-1)  # (..., n_bins)
        bins = torch.arange(self.n_bins, device=arg_logits.device, dtype=arg_logits.dtype)
        soft_idx = (probs * bins).sum(dim=-1)
        lo, hi = self.coord_range
        return lo + (hi - lo) * soft_idx / max(self.n_bins - 1, 1)

    def forward(self, arg_logits, line_cmd_mask, line_start_index):
        """
        arg_logits: (T, B, n_args, n_bins)
        line_cmd_mask: (T, B) bool/float，指示该位置是否为线段命令
        line_start_index: (T, B) long，当前线段在 line_ref 空间中的索引；非线段位置可置 -1
        """
        arg_cont = self.soft_dequantize(arg_logits)  # (T, B, n_args)

        # 示例：假设 LINE 的前 4 个参数依次为 x1, y1, x2, y2
        p1 = arg_cont[..., 0:2]
        p2 = arg_cont[..., 2:4]
        d = p2 - p1
        norm = torch.norm(d, dim=-1, keepdim=True).clamp_min(self.eps)
        unit = d / norm

        return {
            "start": p1,
            "end": p2,
            "dir": d,
            "unit": unit,
            "valid": line_cmd_mask.float(),
            "line_index": line_start_index,
        }


class DifferentiableConstraintEvaluator(nn.Module):
    def __init__(self, lambda_collinear_dist=1.0, eps=1e-6):
        super().__init__()
        self.lambda_collinear_dist = lambda_collinear_dist
        self.eps = eps

    def _dot_sq(self, u, v):
        return (u * v).sum(dim=-1).pow(2)

    def _point_to_line_sq(self, p, a, u):
        ap = p - a
        proj = (ap * u).sum(dim=-1, keepdim=True) * u
        ortho = ap - proj
        return (ortho ** 2).sum(dim=-1)

    def horizontal_residual(self, u):
        return u[..., 1].pow(2)

    def vertical_residual(self, u):
        return u[..., 0].pow(2)

    def parallel_residual(self, u, v):
        return 1.0 - self._dot_sq(u, v)

    def perpendicular_residual(self, u, v):
        return self._dot_sq(u, v)

    def collinear_residual(self, a1, a2, b1, b2, u, v):
        r_parallel = self.parallel_residual(u, v)
        r_dist = 0.25 * (
            self._point_to_line_sq(b1, a1, u)
            + self._point_to_line_sq(b2, a1, u)
            + self._point_to_line_sq(a1, b1, v)
            + self._point_to_line_sq(a2, b1, v)
        )
        return r_parallel + self.lambda_collinear_dist * r_dist

    def forward(self, soft_lines, relations, type_weights=None):
        losses = []
        stats = {}
        type_weights = type_weights or {}

        for rel in relations:
            a = rel.line_a
            b = rel.line_b
            p1 = soft_lines["start"][:, a]
            p2 = soft_lines["end"][:, a]
            u = soft_lines["unit"][:, a]

            if rel.type_id.name == "HORIZONTAL":
                r = self.horizontal_residual(u)
                w = type_weights.get("HORIZONTAL", 1.0)
            elif rel.type_id.name == "VERTICAL":
                r = self.vertical_residual(u)
                w = type_weights.get("VERTICAL", 1.0)
            else:
                q1 = soft_lines["start"][:, b]
                q2 = soft_lines["end"][:, b]
                v = soft_lines["unit"][:, b]
                if rel.type_id.name == "PARALLEL":
                    r = self.parallel_residual(u, v)
                    w = type_weights.get("PARALLEL", 1.0)
                elif rel.type_id.name == "PERPENDICULAR":
                    r = self.perpendicular_residual(u, v)
                    w = type_weights.get("PERPENDICULAR", 1.0)
                else:
                    r = self.collinear_residual(p1, p2, q1, q2, u, v)
                    w = type_weights.get("COLLINEAR", 1.0)

            losses.append(w * r)
            stats.setdefault(rel.type_id.name, []).append(r)

        if len(losses) == 0:
            zero = soft_lines["start"].new_tensor(0.0)
            return zero, {"n_valid_constraints": 0}

        loss = torch.stack(losses, dim=0).mean()
        metrics = {"n_valid_constraints": len(losses)}
        for k, vals in stats.items():
            metrics[f"geom_{k.lower()}"] = torch.stack(vals, dim=0).mean()
        return loss, metrics
```

> **实现说明**：若现有 DeepCAD 向量格式中的 line 起点由“前一条曲线终点”隐式决定，则 `DifferentiableSketchInterpreter` 需要维护一个 torch 版“当前笔尖状态”；上例用 `x1, y1, x2, y2` 仅演示梯度路径。工程落地时应镜像 `cadlib/sketch.py` 中 `Loop.from_vector()` 的顺序语义，但避免使用 `numpy` / 硬分支。

#### 举例说明

假设 GT 中存在 `PARALLEL(线3, 线8)`。若 decoder 预测出的两条线方向夹角仍有偏差，则 `r_parallel = 1 - (u·v)^2 > 0`；该误差会通过 `unit -> end/start -> soft_dequantize -> arg_logits` 反传，使参数头直接修正线段方向。若两条线几乎平行但存在侧向偏移，则 `COLLINEAR(3, 8)` 的 `r_parallel` 已很小，但 `r_line_distance` 仍较大，从而继续推动端点坐标靠近同一直线。

---

### 4.9 模块：总损失与训练规则模块

#### 模块作用

统一定义训练优化目标，避免损失函数散落在脚本中难以维护。

#### 模块原理

总损失定义为：

```text
L_total = L_cmd + α · L_constraint_pred + β · L_constraint_recon + γ · L_geom_constraint
```

其中：

| 项 | 含义 |
| --- | --- |
| `L_cmd` | 原 DeepCAD 命令/参数交叉熵 |
| `L_constraint_pred` | 解码器侧约束预测损失（**可选**；与主方案一致，保留预测头时计入） |
| `L_constraint_recon` | 从 \(z\) 重建约束图的辅助损失 |
| `L_geom_constraint` | 对预测线段几何直接计算的可微约束残差 |

权重可 **schedule**：例如前期较大 \(\beta\) 迫使 \(z\) 学约束；`γ` 采用 warmup，从 0 逐步升至目标值，待主任务语法/参数基本稳定后，再强化几何约束闭环。

**与基线 Constraint-Aware 的关系**：基线形式为 `L_total = L_cmd + α * L_constraint`，其中 `L_constraint` 为约束分类 BCE/CE 等与预测头一致；本方案在保留 `L_cmd` 与（可选）`L_constraint_pred` 的前提下增加 `L_constraint_recon` 与 `L_geom_constraint`，分别把一部分「约束可学性」压到**瓶颈 \(z\)**，以及把一部分「约束达成性」压回**输出参数几何**。

```python
def total_loss_baseline(logits_cmd, targets_cmd, logits_cstr, targets_cstr, alpha=0.2):
    loss_cmd = cross_entropy_cmd(logits_cmd, targets_cmd)
    loss_c = constraint_loss(logits_cstr, targets_cstr)
    return loss_cmd + alpha * loss_c
```

若 \(\alpha\) 过小，约束满足率提升不明显；若 \(\gamma\) 过大，可能在主任务尚未稳定时过早拉扯参数空间，导致命令困惑度上升。建议在验证集上同时监控 **命令 token 准确率**、**约束满足率** 与 **几何残差**（与 4.10 节一致）。

#### 代码

```python
class LossComposer:
    def __init__(self, alpha=0.1, beta=0.5, gamma=0.2, pos_weight=5.0):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.pos_weight = pos_weight

    def compose(self, cmd_loss, pred_loss, unary_pred, pair_pred, unary_gt, pair_gt, geom_loss):
        recon_loss = weighted_bce(unary_pred, unary_gt, self.pos_weight)
        recon_loss = recon_loss + weighted_bce(pair_pred, pair_gt, self.pos_weight)
        return cmd_loss + self.alpha * pred_loss + self.beta * recon_loss + self.gamma * geom_loss
```

#### 举例说明

若实验中发现 `L_recon` 一直高而 `L_cmd` 很低，说明模型更会“背答案”而不会“记约束”，此时可上调 `beta`。若 `L_geom_constraint` 长期不下降，优先检查可微解释器与 `line_ref` 对齐是否正确，再考虑提高 `gamma`。若 `L_cmd` 明显恶化，则说明辅助约束压制了主生成任务，可降低 `beta` / `gamma`，或推迟启用几何闭环（主方案 §3.7、§4.3 分阶段）。

---

### 4.10 模块：评估指标与工程校验（与主方案 §3.8 对齐）

#### 模块作用

量化**约束达成情况**与**几何/拓扑质量**，便于与原生 DeepCAD 及 Constraint-Aware 基线对比与消融（去约束分支、去损失、去 cross-attn、去编码器融合等）。映射到 DDD：由 **Training Orchestration** 或 **独立评估接口** 调用 `EvaluateConstraintSatisfactionUseCase` 与几何指标工具。

#### 模块原理

- **约束满足率**：相对 GT（由 4.1 节提取器得到的约束）；水平/竖直可按线段计数；平行/垂直/共线可按线对集合 precision/recall 或「满足数 / GT 总数」。
- **几何**：Chamfer / Hausdorff（若将草图栅格化或采样为点集）。
- **拓扑**：闭合性、自交、非法环等规则检测（依产品定义实现）。

#### 代码

```python
def horizontal_rate(lines):
    """示例：水平线段占比。"""
    n_h = sum(1 for L in lines if abs(L["vec"][2]) < 1e-5)
    return n_h / max(len(lines), 1)


def pair_satisfaction(pairs_gt, judge_fn, pairs_pred_geometry):
    """示例：线对约束满足数 / GT 对数。"""
    ok = sum(1 for p in pairs_gt if judge_fn(p, pairs_pred_geometry))
    return ok / max(len(pairs_gt), 1)
```

#### 举例说明

GT 有 10 对平行约束；生成几何解析后，其中 7 对同时满足角度阈值，则**平行满足率**可为 0.7（定义需与论文表述一致，可改为 F1）。

---

## 5. 应用服务设计

### 5.1 用例：训练一个 Batch

#### 模块作用

这是 `Training Orchestration Context` 的核心应用服务，负责组织一次前向、损失计算、反向传播与指标记录。

#### 模块原理

应用服务只编排，不直接实现领域规则。它依赖：

1. `SketchRepository`
2. `ConstraintFusionDomainService`
3. `DecoderAdapter`
4. `ConstraintReconstructionDomainService`
5. `DifferentiableConstraintEvaluator`
6. `LossComposer`

#### 代码

```python
class TrainConstraintFusedBatchUseCase:
    def __init__(self, encoder_service, decoder_adapter, recon_service, geom_evaluator, loss_composer):
        self.encoder_service = encoder_service
        self.decoder_adapter = decoder_adapter
        self.recon_service = recon_service
        self.geom_evaluator = geom_evaluator
        self.loss_composer = loss_composer

    def execute(self, batch):
        """
        batch 需包含：
        - encoder_kwds: 传入 ConstraintFusionDomainService.fuse 的字典（与 EncoderFused.forward + 可选 groups 一致）
        - command_targets / cad_loss_targets：主任务标签
        - constraint_tags, unary_gt, pair_gt：辅助监督
        - 可选 constraint_memory, constraint_mask：仅当 decoder 启用 OptionalConstraintCrossAttn 时传入
        """
        latent = self.encoder_service.fuse(batch["encoder_kwds"])
        cmd_logits, constraint_pred_logits = self.decoder_adapter(
            latent.tensor,
            constraint_memory=batch.get("constraint_memory"),
            constraint_mask=batch.get("constraint_mask"),
            targets=batch["command_targets"],
        )
        unary_pred, pair_pred = self.recon_service.reconstruct(latent)
        cmd_loss = cad_sequence_loss(cmd_logits, batch)
        pred_loss = constraint_prediction_loss(constraint_pred_logits, batch["constraint_tags"])
        geom_loss, geom_metrics = self.geom_evaluator(
            soft_lines=batch["soft_lines"],
            relations=batch["constraint_relations"],
        )
        total_loss = self.loss_composer.compose(
            cmd_loss,
            pred_loss,
            unary_pred,
            pair_pred,
            batch["unary_gt"],
            batch["pair_gt"],
            geom_loss,
        )
        return total_loss
```

#### 举例说明

训练脚本只需构造 `batch`（通常由 `collate` 与 `SketchRepository` 产出），调用 `execute()`；**默认**不向 decoder 传入 `constraint_memory`，即 latent-only 主路径。若启用几何闭环，`batch` 还需提供 `soft_lines` 所需的参数分布或可微解释输入，以及 `constraint_relations`。Phase 3 若启用可选 Cross-Attn，在同一份 batch 中附带 `constraint_memory`（如约束 token 的 embedding 或并行编码支路输出）与 `constraint_mask` 即可，推理阶段省略这两项即与主方案「推理不强制 \(C\)」一致。

---

### 5.2 用例：仅基于 Latent 生成 CAD 序列

#### 模块作用

提供论文和实际演示最关键的推理用例：**不依赖外部约束输入，仅用 \(z\) 生成序列**。

#### 模块原理

该用例有两种来源：

1. 自编码路径：`z = EncoderFused(x, C)`。
2. 采样路径：`z ~ LatentGAN`。

无论哪种来源，生成侧接口保持一致。

#### 代码

```python
class GenerateFromLatentUseCase:
    def __init__(self, decoder):
        self.decoder = decoder

    def execute(self, latent):
        return self.decoder.decode(latent.tensor)
```

#### 举例说明

当使用 Latent GAN 采样到一个新 \(z\) 时，该用例直接调用 decoder 得到 CAD 序列，而不会要求再提供约束字典。这正是本方案与“解码器依赖外部约束”的老路径最本质的区别。

---

### 5.3 用例：评估约束满足率

#### 模块作用

把模型输出重新映射到约束空间，评估“生成了什么”以及“是否满足预期约束结构”。

#### 模块原理

评估分三步：

1. 从输出 CAD 序列恢复线段与拓扑。
2. 重新抽取预测约束。
3. 与 GT 或参考统计分布比较。

#### 代码

```python
class EvaluateConstraintSatisfactionUseCase:
    def __init__(self, extractor, metrics):
        self.extractor = extractor
        self.metrics = metrics

    def execute(self, decoded_sequence, reference_constraints):
        pred_constraints = self.extractor.build_relations(decoded_sequence)
        return self.metrics.compare(pred_constraints, reference_constraints)
```

#### 举例说明

如果目标样本要求存在线 1 与线 4 平行、线 2 与线 4 垂直，则评估时会检查模型输出重构出的几何是否满足这些关系，而不是只看 token 级交叉熵是否下降。

---

## 6. 基础设施设计

### 6.1 目录建议

与主技术方案 §4.5「`constraint_deepcad/` 扁平目录」可**一一映射**：本节采用 DDD 分层目录；若仓库保持单包结构，可将 `domain/`、`application/` 等折叠为同级模块文件，但**限界上下文边界与 import 方向**仍建议遵守下文原则。

```text
constraint_fused_deepcad/
├── application/
│   ├── use_cases/
│   │   ├── train_constraint_fused_batch.py
│   │   ├── generate_from_latent.py
│   │   └── evaluate_constraint_satisfaction.py
│   └── services/
│       └── loss_composer.py
├── domain/
│   ├── aggregates/
│   │   └── sketch_sequence_aggregate.py
│   ├── entities/
│   │   ├── cad_command.py
│   │   └── constraint_relation.py
│   ├── value_objects/
│   │   ├── constraint_tag_vector.py
│   │   └── constraint_aware_latent.py
│   ├── services/
│   │   ├── constraint_fusion_service.py
│   │   └── constraint_reconstruction_service.py
│   └── repositories/
│       ├── sketch_repository.py
│       └── checkpoint_repository.py
├── infrastructure/
│   ├── data/
│   │   ├── constraint_extractor.py
│   │   ├── constraint_tokenizer.py
│   │   └── deepcad_dataset_repository.py
│   ├── model/
│   │   ├── constraint_tag_embedding.py
│   │   ├── constraint_token_encoder.py
│   │   ├── encoder_fused.py
│   │   ├── pooling.py
│   │   ├── bottleneck.py
│   │   ├── recon_head.py
│   │   ├── constraint_pred_head.py
│   │   ├── optional_constraint_cross_attn.py
│   │   └── decoder_adapter.py
│   ├── persistence/
│   │   └── checkpoint_repository_fs.py
│   └── monitoring/
│       ├── tensorboard_logger.py
│       └── experiment_tracker.py
├── artifacts/
│   ├── experiments/
│   │   └── <exp_id>/
│   │       ├── manifest.json
│   │       ├── train_metrics.csv
│   │       ├── eval_metrics.json
│   │       ├── best_checkpoint.txt
│   │       └── qualitative_cases.json
│   └── paper_exports/
│       ├── tables/
│       └── figures/
├── interfaces/
│   ├── train.py
│   ├── infer.py
│   └── evaluate.py
└── config/
    └── constraint_fused_config.py
```

### 6.2 基础设施实现原则

1. `infrastructure/data` 只负责把外部数据映射为领域对象。
2. `infrastructure/model` 只放神经网络具体实现，不直接写训练流程。
3. `interfaces/*.py` 只作为入口，不实现核心业务规则。
4. 配置应统一收敛到 `config/`，避免超参数散落。

### 6.3 配置项（KEY PARAMS）

| 参数 | 建议值 | 说明 |
| --- | --- | --- |
| `d_model` | 256 | 模型宽度，建议先与原版一致 |
| `n_layers` | 4 | 编码器层数 |
| `n_heads` | 8 | 多头注意力头数 |
| `n_constraint_types` | 6 | 5 类真实约束 + 1 类 `NONE/PAD` |
| `max_constraints` | 32 | 约束 token 最大条数 |
| `max_lines` | 64 | 线段索引最大值 |
| `alpha` | 0.1 | 约束预测损失权重 |
| `beta` | 0.5 | 约束重建损失权重 |
| `gamma` | 0.2 | 几何一致性损失权重，建议 warmup 启用 |
| `pos_weight` | 5.0 | 稀疏约束正样本权重 |
| `pooling_strategy` | `masked_mean` | 主方案策略 A；可选 `dual_stream_gate`（策略 B） |
| `enable_constraint_pred_head` | `true` | 启用 decoder 侧约束预测监督（可与主方案 `L_constraint_pred` 一起关闭做消融） |
| `enable_decoder_cross_attn` | `false` | **默认关闭**；与主方案 Phase 3 一致，可改为 `true` 并配置 `training_dropout` 做可选增强 |
| `constraint_cross_attn_dropout` | `0.5` | 训练期跳过 Cross-Attn 的概率上界（schedule 时逐步增大） |
| `geom_loss_warmup_epochs` | `10` | 前若干 epoch 线性升高 `γ`，避免过早扰动主任务 |

### 6.4 日志与监控

建议记录以下指标：

1. `L_cmd`
2. `L_constraint_recon_unary`
3. `L_constraint_recon_pair`
4. `L_geom_constraint`
5. `constraint_satisfaction_rate`
6. `latent_recon_consistency`
7. `decoder_only_infer_score`

这些指标共同回答三个问题：

1. 模型会不会生成？
2. 模型记没记住约束？
3. 不给外部约束时还能不能生成合理结果？

这些在线指标主要服务于训练监控与调参；若目标是后续发论文、做消融和复现实验，则还需要一套“实验数据记录与论文证据链”方案，把配置、checkpoint、评估结果与图表导出关联起来。

### 6.5 实验数据记录与论文证据链

#### 模块作用

把一次实验从“跑过了”提升到“可复现、可对比、可写论文”的证据单元。该模块负责统一记录以下信息：

1. **实验身份信息**：`exp_id`、时间、实验阶段、实验目的、负责人。
2. **可复现实验条件**：配置快照、数据集切分、随机种子、设备环境、代码版本。
3. **训练过程数据**：step/epoch 级损失、约束满足率、最优 checkpoint、早停点。
4. **论文结果数据**：主表指标、消融指标、定性案例、图表导出路径。

#### 模块原理

该方案采用“三层记录模型”，保证训练日志、评估结果与论文素材可以通过同一个 `exp_id` 串起来。

1. **Run-Level（实验级）**：用 `manifest.json` 固化一次实验的不可变上下文，包括模型配置、数据版本、随机种子、机器环境和实验目标。它回答“这次实验到底是什么”。
2. **Process-Level（过程级）**：用 `train_metrics.csv` 或 TensorBoard 标量流持续记录训练中的关键数值，包括 `L_cmd`、`L_constraint_recon_unary`、`L_constraint_recon_pair`、`L_geom_constraint`、`constraint_satisfaction_rate` 等。它回答“这次实验是怎么收敛的”。
3. **Evidence-Level（证据级）**：用 `eval_metrics.json`、`qualitative_cases.json`、论文表格与图像导出文件保存最终结论。它回答“这次实验最终能支撑哪一张表、哪一张图、哪一个结论”。

建议每次实验都遵守以下主键关联规则：

| 记录对象 | 主键 / 关联键 | 作用 |
| --- | --- | --- |
| `manifest.json` | `exp_id` | 固化配置、环境、数据切分、实验目标 |
| `train_metrics.csv` | `exp_id + epoch + step` | 追踪训练曲线与最优点 |
| `eval_metrics.json` | `exp_id + checkpoint_tag` | 保存单次评估结果 |
| `best_checkpoint.txt` | `exp_id` | 记录最佳模型路径 |
| `qualitative_cases.json` | `exp_id + sample_id` | 记录论文可视化案例 |
| `paper_exports/tables/*.csv` | `table_id + source_exp_ids` | 生成论文主表、消融表 |

为避免后期“知道最优结果但不知道怎么来的”，`manifest.json` 至少应包含以下字段：

| 字段 | 说明 |
| --- | --- |
| `exp_id` | 实验唯一编号，如 `cfdeepcad_p2_ablation_seed3` |
| `stage` | 所属阶段，如 `baseline`、`fusion`、`ablation`、`final` |
| `hypothesis` | 本次实验要验证的核心假设 |
| `dataset_name` / `dataset_split` | 数据集名称与训练/验证/测试切分 |
| `model_config` | 关键模型参数快照 |
| `loss_config` | `alpha`、`beta`、`pos_weight` 等损失超参 |
| `seed` | 随机种子 |
| `device_env` | GPU、CUDA、PyTorch 等环境 |
| `code_version` | git commit、分支名或手工版本标签 |
| `resume_from` | 如为续训，记录来源 checkpoint |
| `paper_tags` | 标记该实验对应主实验、消融、可视化或附录 |

为了服务论文写作，建议在 `eval_metrics.json` 中把指标按论文口径分为四组：

1. **重建质量**：命令 token 准确率、参数误差、序列合法率。
2. **约束质量**：约束识别 F1、约束满足率、pair constraint recall。
3. **生成能力**：latent-only 有效样本率、多样性、失败率。
4. **工程成本**：训练时长、显存峰值、参数量、吞吐量。

这样做的目的不是让所有指标都进主文，而是保证后续可以快速筛选：

1. 主表保留核心指标。
2. 消融表比较模块开关差异。
3. 附录放训练曲线、失败案例和更多可视化。

#### 代码

```python
from dataclasses import asdict, dataclass
from pathlib import Path
import csv
import json
from datetime import datetime


@dataclass
class ExperimentManifest:
    exp_id: str
    stage: str
    hypothesis: str
    dataset_name: str
    dataset_split: dict
    model_config: dict
    loss_config: dict
    seed: int
    device_env: dict
    code_version: str
    paper_tags: list[str]
    resume_from: str | None = None


class ExperimentTracker:
    def __init__(self, root: str, exp_id: str):
        self.exp_dir = Path(root) / exp_id
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_csv = self.exp_dir / "train_metrics.csv"

    def save_manifest(self, manifest: ExperimentManifest):
        payload = asdict(manifest)
        payload["created_at"] = datetime.now().isoformat()
        (self.exp_dir / "manifest.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def log_train_metrics(self, epoch: int, step: int, metrics: dict):
        file_exists = self.metrics_csv.exists()
        fieldnames = ["epoch", "step"] + list(metrics.keys())
        with self.metrics_csv.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow({"epoch": epoch, "step": step, **metrics})

    def save_eval_metrics(self, checkpoint_tag: str, metrics: dict):
        payload = {
            "checkpoint_tag": checkpoint_tag,
            "metrics": metrics,
        }
        (self.exp_dir / "eval_metrics.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
```

上述实现刻意保持简单：TensorBoard 负责在线看曲线，`ExperimentTracker` 负责把论文真正需要的结构化数据沉淀到文件系统。二者不是替代关系，而是“监控 + 归档”的组合。

#### 举例说明

例如，做“是否引入约束重建头”的消融实验时，可定义以下两组实验：

1. `cfdeepcad_baseline_seed1`
2. `cfdeepcad_recon_head_seed1`

两组实验都写入各自的 `manifest.json`，明确记录：

1. baseline 关闭 `enable_constraint_pred_head` 与 `L_constraint_recon`
2. recon 版本开启 `ConstraintReconstructionDomainService`
3. 两组使用相同训练集切分、相同 seed、相同训练轮数

训练结束后：

1. `train_metrics.csv` 用于画收敛曲线。
2. `eval_metrics.json` 用于提取主表中的约束满足率、序列合法率和 latent-only 有效样本率。
3. `qualitative_cases.json` 用于保存论文中的可视化案例，例如“生成结果满足平行/垂直关系的成功样本”和“失败样本”。
4. `paper_exports/tables/table_main.csv` 则从多个 `exp_id` 汇总形成论文主表。

最终，论文中的每一行实验结果都能回溯到：

1. 当时的配置是什么；
2. 最优模型是哪一个 checkpoint；
3. 指标是在什么数据切分下算出来的；
4. 是否有对应的定性案例和训练曲线可支撑结论。

---

## 7. 时序流程设计

### 7.1 训练时序

```text
Train Script
   │
   ├── SketchRepository.load(batch_ids)
   │        │
   │        └── 返回 SketchSequenceAggregateBatch
   │
   ├── TrainConstraintFusedBatchUseCase.execute(batch)
   │        │
   │        ├── ConstraintFusionDomainService.fuse(batch["encoder_kwds"])
   │        ├── DecoderAdapter(latent[, optional C]) -> cmd_logits + constraint_pred_logits
   │        ├── ConstraintReconstructionDomainService.reconstruct(latent)
   │        ├── LossComposer.compose(...)
   │        └── 返回 total_loss
   │
   ├── backward()
   ├── optimizer.step()
   └── logger / checkpoint
```

### 7.2 推理时序

```text
LatentGAN / (EncoderFused → Bottleneck)
        │
        ▼
ConstraintAwareLatent
        │
        ▼
GenerateFromLatentUseCase
        │
        ▼
Decoder
        │
        ▼
CAD Sequence
```

### 7.3 评估时序

```text
Decoded Sequence
    │
    ▼
ConstraintExtractor
    │
    ▼
Pred Constraint Graph
    │
    ▼
MetricCalculator.compare()
    │
    ▼
Constraint Satisfaction / Reconstruction Metrics
```

---

## 8. 测试与验收设计

### 8.1 单元测试

应优先覆盖以下领域规则：

1. `ConstraintExtractor` 能正确把原始字典转换为 `ConstraintRelation`。
2. `build_constraint_tags()` 对 unary/pair 约束映射正确。
3. `ConstraintTokenEncoder` 输出 shape 正确，padding 行为正确。
4. `EncoderFused` 在不同 `S`、`T_c` 下输出稳定 shape。
5. `ConstraintReconHead` 的 unary/pair 输出维度正确。

### 8.2 集成测试

建议覆盖以下用例：

1. 单 batch 训练可正常前后向。
2. 在 latent-only decoder 路径下推理可正常运行。
3. 使用随机 latent 可输出合法命令序列。
4. `SketchRepository -> Aggregate -> Encoder -> Decoder` 全链路不丢字段。

### 8.3 验收指标

| 指标 | 目标 |
| --- | --- |
| 重建任务收敛稳定性 | `L_cmd`、`L_constraint_recon` 与 `L_geom_constraint` 可协同下降 |
| 约束满足率 | 相比原 DeepCAD 或 decoder-only 注入方案提升 |
| latent-only 推理可用性 | 不依赖外部 \(C\) 仍可生成有效 CAD 序列 |
| 工程可维护性 | 新增约束类型时修改范围局部可控 |

### 8.4 风险点与缓解策略

| 风险 | 说明 | 缓解策略 |
| --- | --- | --- |
| 约束标签噪声 | 原始 constraint dict 不完整或映射不准 | 先做 extractor 可视化抽检与小样本回放 |
| 约束稀疏导致训练不稳 | pair_gt 极度稀疏 | 使用 `pos_weight`、采样平衡或分阶段启用 |
| 联合序列过长 | \(S + T_c\) 拉高 encoder 开销 | 控制 `max_constraints`，先做截断实验 |
| 辅助监督压制主任务 | `L_constraint_pred`、`L_constraint_recon` 或 `L_geom_constraint` 过强会拖慢主重建 | 通过 `alpha`、`beta`、`gamma` 调权与 schedule，并优先保证 `L_cmd` 稳定 |
| 可选 Cross-Attn 致 train–infer 差 | 训练依赖 \(C\) 而推理关闭 | 使用 `training_dropout` 渐进关闭，并以 latent-only 为验收主路径 |
| latent 容量不足 | \(z\) 无法同时容纳几何与约束 | 先提升 bottleneck 宽度或引入双流池化 |

---

## 9. 总结

### 9.1 方案小结

本详细设计与《Constraint-Fused DeepCAD 技术方案》**目标、损失形式、分阶段路线、评估口径与「推理不强制 \(C\)」约束**保持一致，并在此基础上用 DDD 重组为可编码结构：

1. 用 `SketchSequenceAggregate`（及 collate 批契约）把命令、约束与监督目标统一为核心聚合。
2. 用 `Constraint-Fused Encoding Context` 将 **Constraint-Fused Encoding**（tag + 约束 token + 联合 Self-Attn + 池化）与 **Bottleneck** 衔接，把约束压入 \(z\)。
3. 用应用服务编排训练、推理、评估，**可选**解码器 Cross-Attn 仅在 Phase 3 作为训练增强。
4. 用基础设施层隔离提取器、tokenizer、checkpoint 与实验证据链。

最终能力：**模型上** \(z\) 为约束感知瓶颈；**工程上** 可与基线 Constraint-Aware 五步路线对照消融，并分阶段落地（9.2、9.5）。

### 9.2 分阶段落地路线（与主方案 §4.3 对齐）

| 阶段 | 内容 | 目的 |
| --- | --- | --- |
| Phase 1 | 仅 `ConstraintTagEmbedding` + `CADEmbedding` 改造 | 最小改动验证 \(z\) 是否更可重建约束 |
| Phase 2 | 约束 token + 段嵌入 + 池化 + `L_recon` | 完整编码器侧融合 |
| Phase 3 | 可选解码器 Cross-Attn + dropout schedule | 进一步约束满足率，同时保证推理仅用 \(z\) |
| （工程递进） | 应用服务化、仓储抽象、实验证据链（第 6 章） | 可维护性与论文可复现性 |

### 9.3 明确不做的范围（与主方案 §4.6 对齐）

1. 不改变 DeepCAD 命令表示与数据集格式；不新增人工标注；不更换官方数据集划分与格式（与基线一致）。
2. 不引入 Pointer、扩散、点云等替代生成范式（本方案聚焦 DeepCAD 命令 + 约束融合）。
3. 草图级五类约束为主，不扩展到复杂 3D 特征约束。
4. **推理不将外部约束 \(C\) 作为必需输入**；默认解码主路径为 latent-only，Cross-Attn 仅可选且训练期可渐进关闭。

### 9.4 论文式表述（可选直接使用，与主方案 §4.7 一致）

> 现有 CAD 序列生成模型常在解码器侧引入约束注意力，导致瓶颈隐变量不含约束语义，在隐空间采样生成时约束无法延续。本文提出 Constraint-Fused Encoding：在编码器嵌入与序列层面融合命令级约束标记与约束 token，经联合自注意力建模，并以约束重建任务约束隐空间；解码器可由全局 \(z\) 注入获得约束感知生成，推理无需外部约束序列，训练与推理路径一致，并兼容 DeepCAD 与 Latent GAN 流程。

**DDD 补充一句（若正文需强调工程组织）**：上述能力在实现上映射为 Sketch Preparation、Constraint-Fused Encoding、Generation 与 Training Orchestration 等限界上下文，以聚合根与张量批契约隔离数据准备与模型细节。

### 9.5 基线 Constraint-Aware 最简复现路线（主方案 §4.2，消融对照）

独立复现或消融**仅解码器注入**的 Constraint-Aware 时：

1. 加载 DeepCAD 官方 JSON，跑通 `ConstraintExtractor`，落盘或缓存约束字典。
2. Dataset `collate` 产出 \(C\) 与 `key_padding_mask`。
3. 在 Decoder 中接入 `Cross-Attn(C)`（优先双路：`z` 与 \(C\) 分列）。
4. 实现约束预测头与 `L_constraint`，总损失为 `L_cmd + α * L_constraint`。
5. 复用原版训练循环，增加评估脚本（4.10 节指标）。

在此基础上接入 Fused 时，将步骤 2 的产出同时用于 `constraint_tags`、联合序列与重建 GT，并按 9.2 节分阶段启用编码器融合、`L_recon` 与 `L_geom_constraint`。

---

*文档版本：DDD 详细设计版；技术要点与《Constraint-Fused DeepCAD 技术方案》同步，并扩展为限界上下文、领域模型、应用服务与基础设施的可编码设计。*
