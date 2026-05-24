@echo off
cd /d "%~dp0.."
python -m constraint_fused_deepcad_simplify_modify2_low_riskbased_high_modify.evaluate ^
  --data_root data ^
  --proj_dir proj_log/constraint_fused_deepcad_simplify_modify2_low_riskbased_high_modify ^
  --exp_name cf_high_modify ^
  --ckpt latest ^
  --eval_split test
