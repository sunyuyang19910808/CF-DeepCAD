@echo off
REM Constraint-Fused DeepCAD Step — 训练命令（在仓库根目录执行）
cd /d "%~dp0.."

REM 按需修改数据根目录
set DATA_ROOT=D:\DeepCAD\DeepCAD\data
set PROJ_DIR=proj_log/constraint_fused_deepcad_step
set GPU=0

REM ---------------------------------------------------------------------------
REM S0: 原始 DeepCAD（无 L_geom 反传）
REM ---------------------------------------------------------------------------
REM python -m constraint_fused_deepcad_step.train ^
REM   --proj_dir %PROJ_DIR% ^
REM   --exp_name deepcad_step_s0_origin ^
REM   --data_root %DATA_ROOT% ^
REM   --batch_size 64 --nr_epochs 100 ^
REM   -g %GPU%

REM ---------------------------------------------------------------------------
REM S1: 几何日志（计算 geom，不反传 gamma_geom=0）
REM ---------------------------------------------------------------------------
REM python -m constraint_fused_deepcad_step.train ^
REM   --proj_dir %PROJ_DIR% ^
REM   --exp_name deepcad_step_s1_geom_log ^
REM   --data_root %DATA_ROOT% ^
REM   --enable_geom_loss --geom_log_only ^
REM   --dataset_cache disk --num_workers 0 ^
REM   --batch_size 64 --nr_epochs 100 ^
REM   -g %GPU%

REM ---------------------------------------------------------------------------
REM S2: 正关系 L_geom，gamma_geom 在 epoch 21–30 线性升至 0.1，epoch 31+ 满值
REM     epoch 1–20: gamma_geom=0
REM ---------------------------------------------------------------------------
python -m constraint_fused_deepcad_step.train ^
  --proj_dir %PROJ_DIR% ^
  --exp_name deepcad_step_s2_geom_pos_warmup21_31 ^
  --data_root %DATA_ROOT% ^
  --enable_geom_loss ^
  --gamma_geom 0.1 ^
  --geom_warmup_start_epoch 21 ^
  --geom_warmup_end_epoch 31 ^
  --dataset_cache disk ^
  --num_workers 4 ^
  --batch_size 64 ^
  --nr_epochs 100 ^
  --warmup_step 2000 ^
  -g %GPU%

REM 续训（同一 exp_name，从 checkpoint 接着跑）：
REM   在上面的命令末尾追加 --continue --ckpt latest

REM epoch 21 起立刻满 gamma_geom（无线性 ramp）：
REM   --geom_warmup_start_epoch 21 --geom_warmup_end_epoch 21
