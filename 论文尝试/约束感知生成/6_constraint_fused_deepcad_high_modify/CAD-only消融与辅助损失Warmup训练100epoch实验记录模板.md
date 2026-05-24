# CAD-only 消融与辅助损失 Warmup 实验记录模板

## 1. 实验命名


| 实验 ID | `exp_name`                         | 目的                       | 状态  |
| ----- | ---------------------------------- | ------------------------ | --- |
| H0    | `cf_high_modify`                   | 当前问题基线                   | 已冻结 |
| H1    | `cf_high_modify_cad_only_100`      | CAD-only 主路径上限           | 训练至 epoch 52 中途，latest=epoch 51，已阶段评估 |
| H2    | `cf_high_modify_warmup_mild_100`   | 温和辅助损失 warmup            | 训练至 epoch 32 手动停止，已做 5/10/…/30/latest 阶段评估 |
| H4    | `cf_high_modify_no_geom_100`       | 关闭 soft geometry 的辅助损失消融 | 待训练 |
| H3    | `cf_high_modify_warmup_strong_100` | 强辅助损失 warmup，可选          | 待决策 |


## 2. H0 当前基线


| 字段                                   | 路径或数值                                                                                                                                                         |
| ------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 配置                                   | `proj_log/constraint_fused_deepcad_simplify_modify2_low_riskbased_high_modify/cf_high_modify/config.txt`                                                      |
| 约束评估                                 | `proj_log/constraint_fused_deepcad_simplify_modify2_low_riskbased_high_modify/cf_high_modify/artifacts/test_eval_latest_20260517_0715/summary.json`           |
| 重建精度                                 | `proj_log/constraint_fused_deepcad_simplify_modify2_low_riskbased_high_modify/cf_high_modify/artifacts/reconstruction_test_latest_20260517_0715_acc_stat.txt` |
| Checkpoint                           | `proj_log/constraint_fused_deepcad_simplify_modify2_low_riskbased_high_modify/cf_high_modify/model/latest.pth`                                                |
| `ACC_cmd`                            | 0.96712562355405                                                                                                                                              |
| `ACC_param`                          | 0.8689659342507456                                                                                                                                            |
| `ratio_h`                            | 0.8965216050968504                                                                                                                                            |
| `ratio_v`                            | 0.9148407987048031                                                                                                                                            |
| `parallel_recall_index_aligned`      | 0.7376895938244429                                                                                                                                            |
| `perpendicular_recall_index_aligned` | 0.8594641628556396                                                                                                                                            |
| `n_parse_fail_pred`                  | 315                                                                                                                                                           |
| `n_samples_extrude_count_mismatch`   | 523                                                                                                                                                           |


## 3. 训练命令记录


| 实验  | 命令                                                                                                                                                                                                                                                                                                                                                 | 开始时间                     | 结束时间   | Checkpoint                                                                                                        |
| --- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------ | ------ | ----------------------------------------------------------------------------------------------------------------- |
| H1  | `python -m constraint_fused_deepcad_simplify_modify2_low_riskbased_high_modify.train --data_root data --proj_dir proj_log/constraint_fused_deepcad_simplify_modify2_low_riskbased_high_modify --exp_name cf_high_modify_cad_only_100 --batch_size 64 --nr_epochs 100 --alpha 0 --beta 0 --gamma 0 --disable_soft_geometry --save_frequency 5 --continue --ckpt latest -g 0` | `2026-05-17 08:24 UTC+8` | `2026-05-18 20:13 UTC+8 手动暂停评估`  | `proj_log/constraint_fused_deepcad_simplify_modify2_low_riskbased_high_modify/cf_high_modify_cad_only_100/model/latest.pth`（`epoch=51, step=128520`） |
| H2  | `python -m constraint_fused_deepcad_simplify_modify2_low_riskbased_high_modify.train --data_root data --proj_dir proj_log/constraint_fused_deepcad_simplify_modify2_low_riskbased_high_modify --exp_name cf_high_modify_warmup_mild_100 --batch_size 64 --num_workers 0 --nr_epochs 100 --aux_schedule warmup --aux_warmup_start_epoch 0 --aux_warmup_end_epoch 10 --alpha 1.0 --beta 0.5 --gamma 1.0 --save_frequency 5 --init_weights_from proj_log/constraint_fused_deepcad_simplify_modify2_low_riskbased_high_modify/cf_high_modify_cad_only_100/model/ckpt_epoch50.pth -g 0` | `2026-05-19`（自 H1 `ckpt_epoch50` 初始化） | `2026-05-20 04:01` 手动停止 | `proj_log/constraint_fused_deepcad_simplify_modify2_low_riskbased_high_modify/cf_high_modify_warmup_mild_100/model/latest.pth`（`epoch=32, step=78500`） |
| H4  | `____`                                                                                                                                                                                                                                                                                                                                             | `____`                   | `____` | `____`                                                                                                            |
| H3  | `____`                                                                                                                                                                                                                                                                                                                                             | `____`                   | `____` | `____`                                                                                                            |


## 4. 统一评估结果


| 实验             | epoch | `ACC_cmd` | `ACC_param` | `ratio_h` | `ratio_v` | `parallel` | `perpendicular` | `parse_fail` | `ext_mismatch` | 结论     |
| -------------- | ----- | --------- | ----------- | --------- | --------- | ---------- | --------------- | ------------ | -------------- | ------ |
| 原始 DeepCAD     | 1000  | 0.9936    | 0.9759      | 0.9510    | 0.9574    | 0.8617     | 0.9279          | 29           | 64             | 参考基线   |
| H0 current     | 25    | 0.9671    | 0.8690      | 0.8965    | 0.9148    | 0.7377     | 0.8595          | 315          | 523            | 问题基线   |
| H1 CAD-only    | 5     | 0.9502    | 0.8629      | 0.8850    | 0.8955    | 0.7171     | 0.8652          | 381          | 570            | `ckpt_epoch5.pth` |
| H1 CAD-only    | 10    | 0.9615    | 0.8865      | 0.8920    | 0.8974    | 0.7300     | 0.8761          | 285          | 426            | `ckpt_epoch10.pth` |
| H1 CAD-only    | 15    | 0.9682    | 0.8972      | 0.8949    | 0.9051    | 0.7363     | 0.8781          | 194          | 396            | `ckpt_epoch15.pth` |
| H1 CAD-only    | 20    | 0.9714    | 0.9050      | 0.9004    | 0.9044    | 0.7400     | 0.8821          | 163          | 314            | `ckpt_epoch20.pth` |
| H1 CAD-only    | 25    | 0.9732    | 0.9098      | 0.9032    | 0.9118    | 0.7562     | 0.8807          | 167          | 278            | `ckpt_epoch25.pth` |
| H1 CAD-only    | 30    | 0.9761    | 0.9142      | 0.9071    | 0.9117    | 0.7526     | 0.8850          | 118          | 242            | `ckpt_epoch30.pth` |
| H1 CAD-only    | 35    | 0.9761    | 0.9174      | 0.9116    | 0.9166    | 0.7583     | 0.8881          | 148          | 244            | `ckpt_epoch35.pth` |
| H1 CAD-only    | 40    | 0.9774    | 0.9201      | 0.9128    | 0.9167    | 0.7656     | 0.8890          | 161          | 207            | `ckpt_epoch40.pth` |
| H1 CAD-only    | 45    | 0.9790    | 0.9221      | 0.9097    | 0.9143    | 0.7596     | 0.8832          | 114          | 211            | `ckpt_epoch45.pth` |
| H1 CAD-only    | 50    | 0.9787    | 0.9239      | 0.9129    | 0.9155    | 0.7689     | 0.8824          | 97           | 234            | `ckpt_epoch50.pth` |
| H1 CAD-only    | 51    | 0.9787    | 0.9243      | 0.9120    | 0.9185    | 0.7651     | 0.8830          | 134          | 179            | `latest.pth` |
| H1 CAD-only    | 100   | 待填        | 待填          | 待填        | 待填        | 待填         | 待填              | 待填           | 待填             | 主路径上限  |
| H2 warmup mild | 5     | 0.9780    | 0.9241      | 0.9177    | 0.9204    | 0.7779     | 0.8858          | 160          | 194            | `ckpt_epoch5.pth` |
| H2 warmup mild | 10    | 0.9794    | 0.9241      | 0.9191    | 0.9258    | 0.7811     | 0.8883          | 119          | 221            | `ckpt_epoch10.pth`；aux warmup 结束 |
| H2 warmup mild | 15    | 0.9801    | 0.9249      | 0.9214    | 0.9270    | 0.7799     | 0.8883          | 122          | 186            | `ckpt_epoch15.pth` |
| H2 warmup mild | 20    | 0.9808    | 0.9255      | 0.9237    | 0.9309    | 0.7868     | 0.8903          | 135          | 178            | `ckpt_epoch20.pth` |
| H2 warmup mild | 25    | 0.9807    | 0.9269      | 0.9247    | 0.9289    | 0.7888     | 0.8909          | 112          | 190            | `ckpt_epoch25.pth` |
| H2 warmup mild | 30    | 0.9821    | 0.9278      | 0.9305    | 0.9330    | 0.7990     | 0.8997          | 110          | 174            | `ckpt_epoch30.pth` |
| H2 warmup mild | 32    | 0.9820    | 0.9285      | 0.9272    | 0.9303    | 0.7763     | 0.8965          | 125          | 173            | `latest.pth` 中断点 |
| H2 warmup mild | 50    | —         | —           | —         | —         | —          | —               | —            | —              | 未训练 |
| H2 warmup mild | 75    | —         | —           | —         | —         | —          | —               | —            | —              | 未训练 |
| H2 warmup mild | 100   | —         | —           | —         | —         | —          | —               | —            | —              | 未训练 |
| H4 no geom     | 50    | 待填        | 待填          | 待填        | 待填        | 待填         | 待填              | 待填           | 待填             | 几何损失消融 |
| H4 no geom     | 100   | 待填        | 待填          | 待填        | 待填        | 待填         | 待填              | 待填           | 待填             | 几何损失消融 |


## 5. 判定摘要

- CAD-only 主路径结论：`H1 关闭 alpha/beta/gamma 与 soft geometry 后，ACC_cmd/ACC_param 相比 H0 明显提升，但截至 latest=epoch 51 仍低于原始 DeepCAD 早期/最终基线；继续单纯延长训练收益偏慢，主路径结构仍需消融定位。`
- Warmup 有效性结论：`H2 自 H1 epoch50 初始化，训练至 epoch32 停止。测试集 ACC_param 由 epoch5 0.9241 升至 latest 0.9285，约束 ratio_h/v 与 parallel/perpendicular 整体优于 H0，但 parallel 在 latest 略回落；ACC 仍低于原始 DeepCAD，需继续训练或调参后再判 warmup 最终有效性。`
- Soft geometry 是否保留：`H1 当前关闭 soft geometry 后性能优于 H0，说明早期 soft geometry/辅助项可能干扰主任务；是否最终保留需等待 H2/H4 对照。`
- 下一阶段推荐配置：`优先执行主路径消融 A1：仅去掉 constraint_tags；若 A1 无明显提升，再做不拼 constraint token、masked_mean、dim_z=256 等消融。`


## 6. 统一训练 loss（每 5 epoch，epoch 内均值）

统计口径：各实验 `train_metrics.csv` 按 `step` 去重后按 epoch 求均值（见各实验 `artifacts/train_metrics_per_epoch.csv`）。`loss_args` 已含 `loss_args_weight=2.0`。

读表说明：

- **H0**：`α=3, β=1, γ=3` 恒定；总 `loss` 含加权辅助项。早期 CSV 无 `aux_*` 列，表中按配置填写。
- **H1**：`α=β=γ=0` 且关闭 soft geometry；**进反传总 loss = `loss_cmd`**；`geom_loss=0`，`pred/recon/geom_*` 仅日志。
- **H2**：自 H1 `ckpt_epoch50` 初始化；epoch 1–10 `aux_*` 线性 warmup（0.1/0.05/0.1 → 1.0/0.5/1.0），epoch 11+ 固定 1.0/0.5/1.0；epoch 32 为中断时 `latest`（非完整 epoch）。

| 实验 | epoch | loss | loss_cmd | loss_cmd_only | loss_args | pred_loss | recon_loss | unary_recon | pair_recon | geom_loss | geom_h | geom_v | geom_para | geom_perp | aux_α | aux_β | aux_γ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| H0 | 5 | 3.9964 | 3.0836 | 0.3388 | 2.7448 | 0.0996 | 0.3712 | 0.2134 | 0.1579 | 0.0810 | 0.0464 | 0.0404 | 0.0699 | 0.1025 | 3.00 | 1.00 | 3.00 |
| H0 | 10 | 3.3167 | 2.6134 | 0.2693 | 2.3441 | 0.0760 | 0.2758 | 0.1593 | 0.1165 | 0.0665 | 0.0349 | 0.0295 | 0.0544 | 0.0872 | 3.00 | 1.00 | 3.00 |
| H0 | 15 | 2.8620 | 2.3202 | 0.2172 | 2.1030 | 0.0603 | 0.2063 | 0.1155 | 0.0908 | 0.0515 | 0.0252 | 0.0200 | 0.0381 | 0.0712 | 3.00 | 1.00 | 3.00 |
| H0 | 20 | 2.5950 | 2.1452 | 0.1850 | 1.9603 | 0.0532 | 0.1680 | 0.0961 | 0.0718 | 0.0407 | 0.0200 | 0.0153 | 0.0301 | 0.0563 | 3.00 | 1.00 | 3.00 |
| H0 | 25 | 2.4924 | 2.0605 | 0.1780 | 1.8825 | 0.0499 | 0.1571 | 0.0899 | 0.0672 | 0.0416 | 0.0199 | 0.0159 | 0.0302 | 0.0577 | 3.00 | 1.00 | 3.00 |
| H1 | 5 | 2.2314 | 2.2314 | 0.2497 | 1.9817 | 0.7923 | 2.9602 | 1.8378 | 1.1224 | 0.0000 | 0.7769 | 0.1715 | 0.0621 | 0.9546 | 0.00 | 0.00 | 0.00 |
| H1 | 10 | 1.8686 | 1.8686 | 0.1824 | 1.6862 | 0.7906 | 2.9700 | 1.8479 | 1.1222 | 0.0000 | 0.7768 | 0.1776 | 0.0677 | 0.9503 | 0.00 | 0.00 | 0.00 |
| H1 | 15 | 1.6948 | 1.6948 | 0.1532 | 1.5416 | 0.7745 | 2.9649 | 1.8482 | 1.1167 | 0.0000 | 0.7718 | 0.1836 | 0.0716 | 0.9466 | 0.00 | 0.00 | 0.00 |
| H1 | 20 | 1.5104 | 1.5104 | 0.1261 | 1.3843 | 0.7752 | 2.9650 | 1.8433 | 1.1217 | 0.0000 | 0.7677 | 0.1863 | 0.0805 | 0.9401 | 0.00 | 0.00 | 0.00 |
| H1 | 25 | 1.4520 | 1.4520 | 0.1203 | 1.3317 | 0.7660 | 2.9617 | 1.8387 | 1.1230 | 0.0000 | 0.7699 | 0.1864 | 0.0840 | 0.9362 | 0.00 | 0.00 | 0.00 |
| H1 | 30 | 1.3810 | 1.3810 | 0.1123 | 1.2687 | 0.7681 | 2.9572 | 1.8431 | 1.1142 | 0.0000 | 0.7743 | 0.1850 | 0.0922 | 0.9306 | 0.00 | 0.00 | 0.00 |
| H1 | 35 | 1.3215 | 1.3215 | 0.1034 | 1.2181 | 0.7608 | 2.9774 | 1.8533 | 1.1242 | 0.0000 | 0.7692 | 0.1883 | 0.0933 | 0.9290 | 0.00 | 0.00 | 0.00 |
| H1 | 40 | 1.2983 | 1.2983 | 0.1033 | 1.1951 | 0.7608 | 2.9520 | 1.8379 | 1.1140 | 0.0000 | 0.7695 | 0.1899 | 0.0938 | 0.9276 | 0.00 | 0.00 | 0.00 |
| H1 | 45 | 1.2481 | 1.2481 | 0.0980 | 1.1501 | 0.7561 | 2.9719 | 1.8493 | 1.1226 | 0.0000 | 0.7721 | 0.1902 | 0.0976 | 0.9250 | 0.00 | 0.00 | 0.00 |
| H1 | 50 | 1.2101 | 1.2101 | 0.0955 | 1.1146 | 0.7571 | 2.9533 | 1.8404 | 1.1129 | 0.0000 | 0.7708 | 0.1887 | 0.0968 | 0.9258 | 0.00 | 0.00 | 0.00 |
| H1 | 51 | 1.1928 | 1.1928 | 0.0907 | 1.1021 | 0.7566 | 2.9655 | 1.8439 | 1.1216 | 0.0000 | 0.7721 | 0.1893 | 0.0969 | 0.9252 | 0.00 | 0.00 | 0.00 |
| H2 | 5 | 1.4665 | 1.1923 | 0.0907 | 1.1015 | 0.1364 | 0.3865 | 0.2381 | 0.1484 | 0.2187 | 0.3344 | 0.2334 | 0.0802 | 0.2897 | 0.50 | 0.25 | 0.50 |
| H2 | 10 | 1.5567 | 1.2055 | 0.0958 | 1.1097 | 0.1005 | 0.2890 | 0.1759 | 0.1131 | 0.1062 | 0.1012 | 0.0758 | 0.0469 | 0.1536 | 1.00 | 0.50 | 1.00 |
| H2 | 15 | 1.4456 | 1.1676 | 0.0868 | 1.0808 | 0.0828 | 0.2298 | 0.1403 | 0.0895 | 0.0803 | 0.0796 | 0.0598 | 0.0390 | 0.1124 | 1.00 | 0.50 | 1.00 |
| H2 | 20 | 1.4240 | 1.1635 | 0.0872 | 1.0762 | 0.0790 | 0.2123 | 0.1294 | 0.0830 | 0.0754 | 0.0765 | 0.0577 | 0.0340 | 0.1068 | 1.00 | 0.50 | 1.00 |
| H2 | 25 | 1.3783 | 1.1343 | 0.0844 | 1.0499 | 0.0730 | 0.1988 | 0.1212 | 0.0776 | 0.0716 | 0.0741 | 0.0560 | 0.0331 | 0.1004 | 1.00 | 0.50 | 1.00 |
| H2 | 30 | 1.3457 | 1.1246 | 0.0794 | 1.0452 | 0.0670 | 0.1771 | 0.1056 | 0.0715 | 0.0655 | 0.0683 | 0.0536 | 0.0289 | 0.0926 | 1.00 | 0.50 | 1.00 |
| H2 | 32 | 1.2763 | 1.0761 | 0.0705 | 1.0057 | 0.0632 | 0.1539 | 0.0940 | 0.0599 | 0.0600 | 0.0679 | 0.0521 | 0.0229 | 0.0858 | 1.00 | 0.50 | 1.00 |
| H3 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| H4 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |

列名对照：`geom_h/v/para/perp` = `geom_horizontal` / `geom_vertical` / `geom_parallel` / `geom_perpendicular`；`unary_recon` / `pair_recon` = `unary_recon_loss` / `pair_recon_loss`。

阶段观察（对照 §4 测试指标）：

- **H0→H1（epoch 25/50）**：关闭辅助后 `loss` 明显下降（H0@25 2.49 vs H1@25 1.45），`ACC_param` 0.87→0.91，说明强辅助项拖累主任务。
- **H1**：`loss`（= `loss_cmd`）2.23→1.19；加权 `loss_args` 1.98→1.10；`ACC_param` 0.8629→0.9243（epoch 51）。
- **H2**：在 H1@50 初始化后 `loss` 约 1.47→1.28（至 epoch 32）；`ACC_param` 0.9241→0.9285；约束指标整体优于 H0。

## 7. 实验产物索引

| 实验 | 训练指标 CSV | 每 epoch 聚合 | 测试汇总 | 备注 |
| --- | --- | --- | --- | --- |
| H0 | `cf_high_modify/artifacts/train_metrics.csv` | — | `cf_high_modify/artifacts/test_eval_latest_20260517_0715/` | 训练至 epoch 25 |
| H1 | `cf_high_modify_cad_only_100/artifacts/train_metrics.csv` | `.../train_metrics_per_epoch.csv` | `.../h1_all_checkpoint_eval_summary.csv` | latest epoch 51 |
| H2 | `cf_high_modify_warmup_mild_100/artifacts/train_metrics.csv` | `.../train_metrics_per_epoch.csv` | `.../h2_all_checkpoint_eval_summary.csv` | 自 H1@50 初始化；latest epoch 32 |
| H3/H4 | — | — | — | 待训练 |

路径前缀均为 `proj_log/constraint_fused_deepcad_simplify_modify2_low_riskbased_high_modify/`。checkpoint 评估命名：H1 `reconstruction_test_h1_<ckpt>_acc`；H2 `reconstruction_test_h2_<ckpt>_acc`。

