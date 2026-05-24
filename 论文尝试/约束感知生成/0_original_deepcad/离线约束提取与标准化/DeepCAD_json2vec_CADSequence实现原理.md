# DeepCAD 离线 JSON → 向量序列：`CADSequence` 实现原理

本文说明原始 DeepCAD 数据管线中，将 Fusion360 风格 CAD JSON 转为整数序列向量（[`dataset/json2vec.py`](../../../../dataset/json2vec.py) 中 `process_one` 四步）的实现原理。实现分布在 [`cadlib/extrude.py`](../../../../cadlib/extrude.py)（`CADSequence` / `Extrude`）、[`cadlib/macro.py`](../../../../cadlib/macro.py)（词表与上限）、[`cadlib/sketch.py`](../../../../cadlib/sketch.py)（`Profile` / `Loop`，由 `Extrude` 间接使用）等。

**延伸阅读（同目录）**：[挤压步 `Extrude.to_vector`](./DeepCAD_Extrude_to_vector_编码说明.md)、[轮廓与环 `Profile`/`Loop`](./DeepCAD_Loop与Profile_to_vector_编码说明.md)、[命令词表与 `N_ARGS`](./DeepCAD_macro_命令词表与索引实现说明.md)、[直线一步编码](./DeepCAD_直线Line_to_vector_编码说明.md)。**不替代** [`DeepCAD原始技术方案_详细版.md`](./DeepCAD原始技术方案_详细版.md)。

---

## 1. 目标与数据流

**目标**：把单个零件的 JSON 描述转换为二维整数数组 `cad_vec`（每行为「命令类型 + 参数槽」，宽度 `1 + N_ARGS`），供序列模型训练或写入 HDF5。

**数据流**：

1. [`CADSequence.from_dict(data)`](../../../../cadlib/extrude.py)：解析 JSON，得到拉伸序列与整体包围盒。  
2. [`cad_seq.normalize()`](../../../../cadlib/extrude.py)：按包围盒将几何缩放到约 **[-1, 1]³**（含安全因子 [`NORM_FACTOR`](../../../../cadlib/macro.py)）。  
3. [`cad_seq.numericalize()`](../../../../cadlib/extrude.py)：将浮点量离散到 `0 … n-1`（默认 `n=256`）。  
4. [`cad_seq.to_vector(...)`](../../../../cadlib/extrude.py)：按固定词表与参数布局拼接为 `numpy` 数组；[`json2vec.py`](../../../../dataset/json2vec.py) 中 **`pad=False`**，随后用 [`MAX_TOTAL_LEN`](../../../../cadlib/macro.py) 过滤超长样本。

---

## 2. `CADSequence.from_dict(data)`

**输入**：`data` 为 `json.load` 得到的字典，需包含 `sequence`、`entities`、`properties.bounding_box` 等字段。

**行为**（见 [`CADSequence.from_dict`](../../../../cadlib/extrude.py)）：

- 遍历 `data["sequence"]`，对 `type == "ExtrudeFeature"` 的项，根据 `entity` 在 `data["entities"]` 中解析，调用 [`Extrude.from_dict`](../../../../cadlib/extrude.py) 构造一个或多个 `Extrude` 对象（草图 [`Profile`](../../../../cadlib/sketch.py)、草图平面 [`CoordSystem`](../../../../cadlib/extrude.py)、拉伸操作类型、延伸类型与距离、草图位姿与尺寸等）。**多 profile 的拉伸**会在 `Extrude.from_dict` 中被拆成多条记录。  
- 从 `properties.bounding_box` 的 `max_point` / `min_point` 构造 `bbox`，供归一化使用。

**输出**：`CADSequence(seq, bbox)`，其中 `seq` 为 `Extrude` 的列表。

**概念示例**：若特征序列为「拉伸 A → 拉伸 B」，则 `seq` 可能为 `[Extrude_A, Extrude_B]`；若某次拉伸含多个轮廓，则可能扩展为更多 `Extrude` 元素。

---

## 3. `cad_seq.normalize()`

**作用**：将整个 CAD 模型统一缩放到单位立方体附近，避免数值量级过大，并为后续增广留余量。

**原理**（[`CADSequence.normalize`](../../../../cadlib/extrude.py)）：  
`scale = size * NORM_FACTOR / max(|bbox|)`，再对序列中每个 `Extrude` 调用 `transform(0.0, scale)`，同步缩放草图平面原点、草图全局位置、拉伸距离、草图尺寸等。[`NORM_FACTOR`](../../../../cadlib/macro.py) 默认为 **`0.75`**，用于防止归一化后增广时溢出离散区间。

**示例**：若包围盒各坐标绝对值最大为 100（单位与 JSON 一致），则整体约缩小为原来的 `0.75/100` 倍，几何量落在 **[-1, 1]** 附近。

---

## 4. `cad_seq.numericalize()`

**前提**：须在 `normalize()` 之后调用（[`Extrude.numericalize`](../../../../cadlib/extrude.py) 的注释假定拉伸距离等已在归一化范围内）。

**作用**：将浮点几何与姿态量映射为整数，默认 `n=256`，即参数落在 `0 … 255`。包括草图曲线参数、平面姿态（θ, φ, γ 等）、拉伸距离、草图位置与尺寸等；操作类型、extent 类型等类别索引转为 `int`。

**示例**：对落在 **[-1, 1]** 内的标量，典型映射为 `round(((x + 1) / 2) * n)` 并裁剪到 `[0, n-1]`，供嵌入层或分类头使用。

---

## 5. `cad_seq.to_vector(MAX_N_EXT, MAX_N_LOOPS, MAX_N_CURVES, MAX_TOTAL_LEN, pad=False)`

**宏参数**（[`cadlib/macro.py`](../../../../cadlib/macro.py)）：

| 符号 | 含义 |
|------|------|
| `MAX_N_EXT` | 最大拉伸步数（默认 10） |
| `MAX_N_LOOPS` | 每个草图最大 loop 数（默认 6） |
| `MAX_N_CURVES` | 每个 loop 最大曲线相关长度预算（默认 15），作为 `max_len_loop` 传入 [`Extrude.to_vector`](../../../../cadlib/extrude.py) / [`Profile.to_vector`](../../../../cadlib/sketch.py) |
| `MAX_TOTAL_LEN` | 整条序列最大长度（默认 60），在 [`json2vec.py`](../../../../dataset/json2vec.py) 中用于**事后过滤**，`pad=False` 时本处不把序列填到该长度 |

**行为**（[`CADSequence.to_vector`](../../../../cadlib/extrude.py)）：

- 若 `len(seq) > MAX_N_EXT`，返回 `None`。  
- 否则对每个 `Extrude` 调用 [`item.to_vector(max_n_loops, max_len_loop, pad=False)`](../../../../cadlib/extrude.py)：草图部分为 `Line` / `Arc` / `Circle` 等命令行，以及 `SOL`、`EOS` 等分隔；每条挤压步内含一行 **`Ext`**，携带平面姿态、草图位置与尺寸、拉伸距离与操作/延伸类型等（布局见 [`N_ARGS_SKETCH`](../../../../cadlib/macro.py)、`N_ARGS_PLANE`、`N_ARGS_TRANS`、`N_ARGS_EXT_PARAM` 及 [挤压编码说明](./DeepCAD_Extrude_to_vector_编码说明.md)）。  
- 将各段向量拼接时，**去掉每段末尾的轮廓 `EOS`**（`vec[:-1]`）再连接，最后在**整条序列末尾**追加**一个全局 `EOS`**。  
- **`pad=False`**：不用 `EOS` 将序列填充到 `MAX_TOTAL_LEN`；实际长度由草图复杂度与拉伸次数决定。

核心逻辑摘录：

```263:282:d:\DeepCAD\DeepCAD\cadlib\extrude.py
    def to_vector(self, max_n_ext=10, max_n_loops=6, max_len_loop=15, max_total_len=60, pad=False):
        if len(self.seq) > max_n_ext:
            return None
        vec_seq = []
        for item in self.seq:
            vec = item.to_vector(max_n_loops, max_len_loop, pad=False)
            if vec is None:
                return None
            vec = vec[:-1] # last one is EOS, removed
            vec_seq.append(vec)

        vec_seq = np.concatenate(vec_seq, axis=0)
        vec_seq = np.concatenate([vec_seq, EOS_VEC[np.newaxis]], axis=0)

        # add EOS padding
        if pad and vec_seq.shape[0] < max_total_len:
            pad_len = max_total_len - vec_seq.shape[0]
            vec_seq = np.concatenate([vec_seq, EOS_VEC[np.newaxis].repeat(pad_len, axis=0)], axis=0)

        return vec_seq
```

**结构示意**（单挤压、简化）：`SOL` → 若干草图命令 → `Ext` →（步末 `EOS` 在拼接前被去掉）→ … → 最终全局 `EOS`。多挤压则为多段「草图 + `Ext`」拼接后再接最终 `EOS`。若某草图超出 loop/曲线上限，[`item.to_vector`](../../../../cadlib/extrude.py) 可能返回 `None`，则整条 `to_vector` 失败。

---

## 6. 与 [`dataset/json2vec.py`](../../../../dataset/json2vec.py) 的衔接

[`process_one`](../../../../dataset/json2vec.py) 在成功执行四步后：

- 若 `cad_vec` 超长或无效则丢弃样本；否则将 `cad_vec` 写入 `*.h5` 的 `vec` 数据集。

入口代码：

```26:46:d:\DeepCAD\DeepCAD\dataset\json2vec.py
    try:
        cad_seq = CADSequence.from_dict(data)
        cad_seq.normalize()
        cad_seq.numericalize()
        cad_vec = cad_seq.to_vector(MAX_N_EXT, MAX_N_LOOPS, MAX_N_CURVES, MAX_TOTAL_LEN, pad=False)

    except Exception as e:
        print("failed:", data_id)
        return

    if MAX_TOTAL_LEN < cad_vec.shape[0] or cad_vec is None:
        print("exceed length condition:", data_id, cad_vec.shape[0])
        return

    save_path = os.path.join(SAVE_DIR, data_id + ".h5")
    ...
    with h5py.File(save_path, 'w') as fp:
        fp.create_dataset("vec", data=cad_vec, dtype=int)
```

**注意**：上述 `if` 条件当前写作 `MAX_TOTAL_LEN < cad_vec.shape[0] or cad_vec is None`。若 `to_vector` 返回 `None`，会先计算 `cad_vec.shape[0]` 而触发异常；更稳妥的写法是先判断 **`cad_vec is None`**，再比较长度。

因此四步合起来的语义是：**JSON → 归一化且离散化的 CAD 命令序列 → 在长度与结构约束下的整数矩阵**。

---

## 7. 关键代码位置（便于对照阅读）

| 内容 | 路径 |
|------|------|
| 批处理入口、`h5` 写出 | [`dataset/json2vec.py`](../../../../dataset/json2vec.py) |
| `CADSequence` / `Extrude` / `CoordSystem`、序列 `to_vector` / `from_vector` | [`cadlib/extrude.py`](../../../../cadlib/extrude.py) |
| 词表、`PAD_VAL`、`N_ARGS_*`、`MAX_*`、`EOS_VEC` | [`cadlib/macro.py`](../../../../cadlib/macro.py) |
| `Profile` / `Loop` 与草图向量 | [`cadlib/sketch.py`](../../../../cadlib/sketch.py) |
| 曲线一步 `to_vector` | [`cadlib/curves.py`](../../../../cadlib/curves.py) |

---

*文档说明对象：原始 DeepCAD 向量序列构建；与约束融合扩展可在其他文档中单独描述。*
