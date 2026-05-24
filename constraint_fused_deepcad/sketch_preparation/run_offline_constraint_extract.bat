@echo off
REM Offline constraint extraction -> sketch_preparation\ExtractData (see constraint_extractor.py)
REM Optional extra CLI args are forwarded, e.g.  --limit 100  or  --phase train

cd /d "%~dp0..\..\"
if not exist "data\train_val_test_split.json" (
    echo [ERROR] Missing data\train_val_test_split.json  -^> cwd: %CD%
    exit /b 1
)

set PYTHONWARNINGS=ignore
python -m constraint_fused_deepcad.sketch_preparation.constraint_extractor data --all-phases %*
exit /b %ERRORLEVEL%
