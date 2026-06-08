@echo off
REM Constraint-Fused DeepCAD Step — 训练命令（在仓库根目录执行）
cd /d "%~dp0.."

REM 按需修改数据根目录
set DATA_ROOT=D:\DeepCAD\DeepCAD\data
set PROJ_DIR=proj_log/constraint_fused_deepcad_step
set GPU=0
REM 需 PyTorch>=1.12（scatter_reduce_）；勿用 Windows Store Python 3.8
set PYTHON=c:\Users\Administrator.SY-202303281050\AppData\Local\Programs\Python\Python313\python.exe

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
REM S2+: 正关系 L_geom（默认 angle_hinge，与 test angle_thresh=0.1° 对齐）
REM       geom_target_ratio 在 epoch 51–71 线性升至 0.1，gamma_geom 按 ratio 自适应（上限 0.1）
REM       固定 gamma：去掉 --geom_target_ratio，仅用 --gamma_geom + warmup
REM       回退旧 BCE：加 --geom_loss_mode bce
REM ---------------------------------------------------------------------------
"%PYTHON%" -m constraint_fused_deepcad_step.train ^
  --proj_dir %PROJ_DIR% ^
  --exp_name deepcad_step_s2plus_angle_hinge_warmup51_71 ^
  --data_root %DATA_ROOT% ^
  --enable_geom_loss ^
  --geom_loss_mode angle_hinge ^
  --angle_thresh 0.1 ^
  --gamma_geom 0.1 ^
  --geom_target_ratio 0.1 ^
  --geom_ratio_ema 0.99 ^
  --geom_warmup_start_epoch 51 ^
  --geom_warmup_end_epoch 71 ^
  --dataset_cache disk ^
  --num_workers 0 ^
  --batch_size 64 ^
  --nr_epochs 500 ^
  --warmup_step 2000 ^
  -g %GPU%

REM 续训 epoch 200 -> 500，gamma_geom=0.2（从 latest 接着跑，约 300 epoch）：
REM python -m constraint_fused_deepcad_step.train ^
REM   --proj_dir %PROJ_DIR% ^
REM   --exp_name deepcad_step_s2_geom_pos_warmup21_31 ^
REM   --data_root %DATA_ROOT% ^
REM   --enable_geom_loss ^
REM   --gamma_geom 0.2 ^
REM   --geom_warmup_start_epoch 21 ^
REM   --geom_warmup_end_epoch 31 ^
REM   --dataset_cache disk ^
REM   --num_workers 0 ^
REM   --batch_size 64 ^
REM   --nr_epochs 500 ^
REM   --warmup_step 2000 ^
REM   --continue --ckpt latest ^
REM   -g %GPU%

REM epoch 21 起立刻满 gamma_geom（无线性 ramp）：
REM   --geom_warmup_start_epoch 21 --geom_warmup_end_epoch 21
