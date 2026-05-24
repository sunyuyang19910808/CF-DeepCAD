# DeepCAD：直线 `Line.to_vector` 编码说明

本文档独立整理仓库中 **`cadlib/curves.py`** 里直线一步的向量编码与解码约定，便于查阅；**不替代**同目录下的 [`DeepCAD原始技术方案.md`](./DeepCAD原始技术方案.md) 与 [`DeepCAD原始技术方案_详细版.md`](./DeepCAD原始技术方案_详细版.md)。

---

## 一、方案目标

说明 DeepCAD 如何将一条 **2D 直线段** 压成与 `Arc`、`Circle`、`Ext` 等命令 **同形状** 的单步整数向量，使序列可写入 h5、批训练，并与 `CMD_ARGS_MASK`、嵌入层 `+1` 偏移一致。

---

## 二、整体关系

- **编码**：`Line.to_vector()` → 长度 `1 + N_ARGS` 的 `numpy` 一维数组（首元为命令类型，其余为参数槽）。  
- **解码**：`Line.from_vector(vec, start_point)`，起点来自链式上下文，**不从 `vec` 读取**。  
- **分派**：`construct_curve_from_vector` 根据 `vec[0] == LINE_IDX` 调用 `Line.from_vector`。

---

## 三、模块说明（`Line.to_vector`）

### 3.1 模块作用

将一条直线编码为与 `Arc`、`Circle`、`Ext` 等 **相同 layout** 的单步表示：`command` 一维 + **`N_ARGS`** 个参数槽；直线只写入终点两维，其余槽用 **`PAD_VAL`** 填充，便于矩阵堆叠与参数损失掩码。

### 3.2 模块原理

1. **第 0 维**：曲线类型 **`LINE_IDX`**，等于 `macro.ALL_COMMANDS` 中 `'Line'` 的下标（通常为 **0**）。  
2. **第 1、2 维**：终点 **`(x, y)`**（经 `numericalize` 后为格点整数）。  
3. **起点**：**不写入本步向量**；解码时由 `from_vector(vec, start_point)` 使用上一曲线终点或环起点等上下文。  
4. **定长填充**：在 `[LINE_IDX, end_x, end_y]` 之后追加 **`PAD_VAL`（-1）**，数量为 **`1 + N_ARGS - len(vec)`**（此处 `len(vec)=3`），使整步长度为 **`1 + N_ARGS`**。默认 **`N_ARGS = N_ARGS_SKETCH + N_ARGS_EXT = 16`**，故一步共 **17** 个数。  
5. **与损失一致**：`CMD_ARGS_MASK` 中 `Line` 行对草图参数槽前两维为 1，与只编码终点两维一致。

### 3.3 代码

`to_vector` 与 `from_vector`：

```140:142:d:\DeepCAD\DeepCAD\cadlib\curves.py
    def to_vector(self):
        vec = [LINE_IDX, self.end_point[0], self.end_point[1]]
        return np.array(vec + [PAD_VAL] * (1 + N_ARGS - len(vec)))
```

```107:108:d:\DeepCAD\DeepCAD\cadlib\curves.py
    def from_vector(vec, start_point, is_numerical=True):
        return Line(start_point, vec[1:3])
```

相关常量（`N_ARGS`、`PAD_VAL`）：

```16:22:d:\DeepCAD\DeepCAD\cadlib\macro.py
PAD_VAL = -1
N_ARGS_SKETCH = 5 # sketch parameters: x, y, alpha, f, r
N_ARGS_PLANE = 3 # sketch plane orientation: theta, phi, gamma
N_ARGS_TRANS = 4 # sketch plane origin + sketch bbox size: p_x, p_y, p_z, s
N_ARGS_EXT_PARAM = 4 # extrusion parameters: e1, e2, b, u
N_ARGS_EXT = N_ARGS_PLANE + N_ARGS_TRANS + N_ARGS_EXT_PARAM
N_ARGS = N_ARGS_SKETCH + N_ARGS_EXT
```

`Line` 在掩码中的两行（前两维有效）：

```27:28:d:\DeepCAD\DeepCAD\cadlib\macro.py
CMD_ARGS_MASK = np.array([[1, 1, 0, 0, 0, *[0]*N_ARGS_EXT],  # line
                          [1, 1, 1, 1, 0, *[0]*N_ARGS_EXT],  # arc
```

### 3.4 举例说明

| 设定 | 结果 |
|------|------|
| `LINE_IDX = 0`，量化后终点 `(10, 20)` | 前 3 个数为 `[0, 10, 20]`，再补 `1 + 16 - 3 = 14` 个 `-1`，一步长度 **17**。 |
| 解码 | `from_vector(上述向量, start_point=(0,0))` → 直线从 `(0,0)` 到 `(10,20)`。 |
| 序列侧 | `vec[0]==LINE_IDX` 时由 `construct_curve_from_vector` 分派到 `Line.from_vector`，与 JSON→向量管线一致。 |

---

## 四、总结

- 直线一步：**类型 + 终点 xy + 填充**，总长 **`1 + N_ARGS`**。  
- 起点依赖 **链式 `start_point`**，与 `to_vector` 不写起点的设计成对使用。  
- **`-1` 填充**与嵌入层、损失中的 `PAD_VAL` 处理一致。
