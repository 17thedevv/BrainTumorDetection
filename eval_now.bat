@echo off
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
venv\Scripts\python.exe -X utf8 eval_best_model.py --config experiments/ssl_experiment.yaml
pause
