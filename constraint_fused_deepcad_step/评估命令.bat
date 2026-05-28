@echo off
REM Constraint-Fused DeepCAD Step — 评估命令（TrainRules §4）
cd /d "%~dp0.."

set DATA_ROOT=D:\DeepCAD\DeepCAD\data
set PROJ_DIR=proj_log/constraint_fused_deepcad_step
set EXP_NAME=deepcad_step_s2_geom_pos_warmup21_31
set GPU=0

REM 一步：重建 + ACC + index-aligned 约束指标
python -m constraint_fused_deepcad_step.evaluate ^
  --proj_dir %PROJ_DIR% ^
  --exp_name %EXP_NAME% ^
  --data_root %DATA_ROOT% ^
  --ckpt latest ^
  --eval_split test ^
  -g %GPU%

REM 仅重算指标（已有 reconstruction 目录时）：
REM python -m constraint_fused_deepcad_step.evaluate ^
REM   --proj_dir %PROJ_DIR% ^
REM   --exp_name %EXP_NAME% ^
REM   --skip_reconstruct ^
REM   --reconstruction_dir %PROJ_DIR%/%EXP_NAME%/artifacts/reconstruction_test_latest ^
REM   -g %GPU%
