@echo off
REM Wrapper script de chay SSL pipeline voi UTF-8 encoding tren Windows
REM Su dung: run_ssl.bat [config_path]

set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

set CONFIG=%1
if "%CONFIG%"=="" set CONFIG=experiments/ssl_experiment.yaml

echo [SSL Pipeline] Starting with config: %CONFIG%
echo [SSL Pipeline] UTF-8 mode enabled
echo.

venv\Scripts\python.exe -X utf8 main_ssl.py --config %CONFIG%
