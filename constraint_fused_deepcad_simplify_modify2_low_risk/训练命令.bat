@echo off
cd /d "%~dp0.."
python -m constraint_fused_deepcad_simplify_modify2_low_risk.train --data_root data --proj_dir proj_log/constraint_fused_deepcad_simplify_modify2_low_risk --exp_name cf_simplify_modify2_low_risk --nr_epochs 100 -g 0 --num_workers 0
