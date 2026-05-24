# DeepCAD：`Loop.to_vector` 与 `Profile.to_vector` 编码说明

本文档说明仓库中 **[`cadlib/sketch.py`](../../../../cadlib/sketch.py)** 里 **闭环（Loop）** 与 **轮廓（Profile，多环）** 的序列化约定：如何把多条曲线步打成二维矩阵，如何用 **`SOL` / `EOS`** 分隔环与整廓，以及与整条挤压步 [`Extrude.to_vector`](../../../../cadlib/extrude.py) 的配合。  
**不替代**同目录下的 [`DeepCAD原始技术方案_详细版.md`](./DeepCAD原始技术方案_详细版.md)；单条曲线布局见 [`DeepCAD_直线Line_to_vector_编码说明.md`](./DeepCAD_直线Line_to_vector_编码说明.md)。

---

## 一、方案目标

1. **Loop**：把一个 **封闭环**（有序连接的曲线序列）编成形状为 `(T, 1 + N_ARGS)` 的数组；可选在首部加 **`SOL`**、尾部加 **`EOS`**，并可按 `max_len` **截断拒绝**或 **用 `EOS` 定长填充**。  
2. **Profile**：把 **多个 Loop**（外环在前，见 `Profile.reorder`）顺序拼成一条更长序列，**仅在整廓末尾**加一个 **`EOS`**；多环之间靠每个子环首部的 **`SOL`** 切分；可选校验 `max_n_loops` / `max_len_loop` 并做 `EOS` 填充。

---

## 二、整体关系

| 层次 | 编码入口 | 与上下层关系 |
|------|-----------|----------------|
| 曲线一步 | [`curves.*.to_vector`](../../../../cadlib/curves.py) | 每行 `1 + N_ARGS`，首列为命令类型；见 [直线编码说明](./DeepCAD_直线Line_to_vector_编码说明.md)。 |
| 环 | `Loop.to_vector` | 沿第 0 维堆叠各曲线行，再可选加 `SOL_VEC`、`EOS_VEC`。 |
| 轮廓 | `Profile.to_vector` | 各环先 `to_vector(None, add_eos=False)` 再 `concatenate`，最后追加一行 `EOS_VEC`。 |
| 挤压步 | [`Extrude.to_vector`](../../../../cadlib/extrude.py) | 对 `Profile.to_vector(..., pad=False)` 的结果再插入 `Ext` 行等；Profile 层通常 **`pad=False`**。 |

**分隔符语义**（定义在 [`cadlib/macro.py`](../../../../cadlib/macro.py)）：

- **`SOL_VEC`**：`[SOL_IDX, PAD_VAL, …]`，标记 **新环开始**。  
- **`EOS_VEC`**：`[EOS_IDX, PAD_VAL, …]`，在 Loop 内可表示 **本环结束**；在 Profile 整条末尾表示 **本轮廓结束**。

---

## 三、模块说明

### 3.1 `Loop.to_vector` — 模块作用

将 **单环内** 所有曲线的向量沿时间维叠成矩阵，形成可被 `h5` / batch 消费的 **环级子序列**；与 **`Loop.from_vector`** / **`Profile.from_vector`** 的分段规则一致。

### 3.2 `Loop.to_vector` — 模块原理

1. **`loop_vec = stack(curve.to_vector())`**：第 `i` 行对应环内第 `i` 条曲线，列数为 **`1 + N_ARGS`**（与 Line 文档一致）。  
2. **`add_sol=True`（默认）**：在 **最前** 拼接一行 **`SOL_VEC`**，表示该环起点（多环轮廓里用于切环）。  
3. **`add_eos=True`（默认）**：在 **最后** 拼接一行 **`EOS_VEC`**，表示该环结束。  
4. **`max_len=None`**：不截断不填充，直接返回当前长度 `T`。  
5. **`max_len` 为整数**：若实际长度 **`> max_len`**，返回 **`None`**（数据过复杂或超配置）；若 **`< max_len`**，在尾部用 **`EOS_VEC` 重复** 填到 **`max_len`** 行（与全序列尾部 pad 语义一致）。

**对 `Profile.to_vector` 的特例**：拼接多环时对每个环调用 **`add_eos=False`**，避免环与环之间出现多余的 `EOS`，整廓只保留 **最后一个 `EOS`**。

### 3.3 `Loop.to_vector` — 代码

```189:203:d:\DeepCAD\DeepCAD\cadlib\sketch.py
    def to_vector(self, max_len=None, add_sol=True, add_eos=True):
        loop_vec = np.stack([curve.to_vector() for curve in self.children], axis=0)
        if add_sol:
            loop_vec = np.concatenate([SOL_VEC[np.newaxis], loop_vec], axis=0)
        if add_eos:
            loop_vec = np.concatenate([loop_vec, EOS_VEC[np.newaxis]], axis=0)
        if max_len is None:
            return loop_vec

        if loop_vec.shape[0] > max_len:
            return None
        elif loop_vec.shape[0] < max_len:
            pad_vec = np.tile(EOS_VEC, max_len - loop_vec.shape[0]).reshape((-1, len(EOS_VEC)))
            loop_vec = np.concatenate([loop_vec, pad_vec], axis=0) # (max_len, 1 + N_ARGS)
        return loop_vec
```

**常量**：[`SOL_VEC` / `EOS_VEC`](../../../../cadlib/macro.py) 由 `SOL_IDX`、`EOS_IDX` 与 `PAD_VAL` 填充的 `N_ARGS` 个槽组成。

### 3.4 `Loop.to_vector` — 举例说明

| 设定 | 环内曲线 | 输出形状（默认 `add_sol=True, add_eos=True`） | 行序列含义（首列命令） |
|------|-----------|-----------------------------------------------|-------------------------|
| 2 条直线 | `Line, Line` | `(4, 1+N_ARGS)` | `SOL, Line, Line, EOS` |
| `add_sol=False` | 同上 | `(3, …)` | `Line, Line, EOS` |
| `max_len=6`，实际 4 行 | 同上 | `(6, …)` | 前 4 行同上，后 2 行为 **`EOS` 填充行** |
| `max_len=3`，实际需 4 行 | — | **`None`** | 超长拒绝 |

---

### 3.5 `Profile.to_vector` — 模块作用

将一个 **Profile**（`children` 为多个 [`Loop`](../../../../cadlib/sketch.py)）编码为 **单条二维序列**：多环顺接，**环与环之间仅用下一个环前的 `SOL` 分界**，全局 **仅一个尾部 `EOS`**；可选与 **`MAX_N_LOOPS` / `MAX_N_CURVES`**（训练配置中常映射为 `max_len_loop`）对齐做合法性检查与填充。

### 3.6 `Profile.to_vector` — 模块原理

1. **逐环编码**：`loop.to_vector(None, add_eos=False)` — 每环保留 **`SOL` + 各曲线**，**不带环尾 `EOS`**。  
2. **约束**：若给定 **`max_n_loops`** 且环数过多 → **`None`**；若某环行数超过 **`max_len_loop`** → **`None`**。  
3. **拼接**：`np.concatenate(loop_vecs, axis=0)`，再 **`concatenate` 一行 `EOS_VEC`** 作为 **轮廓结束**。  
4. **`pad=True`**：在尾部再重复 **`EOS_VEC`**，使总行数为 **`max_n_loops * max_len_loop`**。  
   - **注意**：实现上直接计算 `max_n_loops * max_len_loop - profile_vec.shape[0]`，若二者任一为 **`None`** 会无法在 Python 中正确相乘；实际管线里 **[`Extrude.to_vector`](../../../../cadlib/extrude.py) 对 Profile 使用 `pad=False`**，在 Extrude 层再做与 `max_n_loops * max_len_loop` 相关的 padding。

### 3.7 `Profile.to_vector` — 代码

```251:263:d:\DeepCAD\DeepCAD\cadlib\sketch.py
    def to_vector(self, max_n_loops=None, max_len_loop=None, pad=True):
        loop_vecs = [loop.to_vector(None, add_eos=False) for loop in self.children]
        if max_n_loops is not None and len(loop_vecs) > max_n_loops:
            return None
        for vec in loop_vecs:
            if max_len_loop is not None and vec.shape[0] > max_len_loop:
                return None
        profile_vec = np.concatenate(loop_vecs, axis=0)
        profile_vec = np.concatenate([profile_vec, EOS_VEC[np.newaxis]], axis=0)
        if pad:
            pad_len = max_n_loops * max_len_loop - profile_vec.shape[0]
            profile_vec = np.concatenate([profile_vec, EOS_VEC[np.newaxis].repeat(pad_len, axis=0)], axis=0)
        return profile_vec
```

**解码对称性**：[`Profile.from_vector`](../../../../cadlib/sketch.py) 在首个 `EOS` 之前，用 **`SOL_IDX` 出现位置** 把向量切成多段，每段再补上 `EOS` 后交给 **`Loop.from_vector`**，与上述编码结构对应。

### 3.8 `Profile.to_vector` — 举例说明

设每个环默认带 **`SOL`**，单环内仅 **`Line`** 一步以简化记号（真实每行仍为 `1+N_ARGS`）。

| Profile 结构 | `loop_vecs` 拼接（不含最后全局 EOS） | 最终 `profile_vec` 命令列（首列） |
|----------------|--------------------------------------|-----------------------------------|
| 1 个环，2 条边 | `SOL, Line, Line` | `SOL, Line, Line, **EOS**` |
| 2 个环，各 1 条边 | `SOL, Line, SOL, Line` | `SOL, Line, SOL, Line, **EOS**` |

与 **`Extrude.to_vector`** 的关系（摘自 [`extrude.py`](../../../../cadlib/extrude.py)）：Profile 结果去掉 **最后一个 EOS** 的那段会与 **Ext 行** 再拼接，最后仍以 **EOS** 结束整条挤压向量，细节以 Extrude 注释 `NOTE: last one is EOS` 为准。

---

## 四、总结

- **`Loop.to_vector`**：环 = **`[可选 SOL] + 曲线行序列 + [可选 EOS]`**；`max_len` 用于 **超长丢弃** 或 **不足用 EOS 行填充**。  
- **`Profile.to_vector`**：轮廓 = **各环（无环尾 EOS）顺序拼接 + 末尾单一 EOS**；`SOL` 标明下一环起点，供 `from_vector` 切分。  
- 训练相关的默认上限见 [`macro.py`](../../../../cadlib/macro.py) 中 **`MAX_N_LOOPS`、`MAX_N_CURVES`**；整条 CAD 序列拼接见 [`CADSequence.to_vector`](../../../../cadlib/extrude.py) 与 [`dataset/json2vec.py`](../../../../dataset/json2vec.py)。
