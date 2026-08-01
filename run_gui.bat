@echo off
REM Wrapper script de chay Brain Tumor Detection GUI app voi virtual environment
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

echo [GUI App] Starting Brain Tumor Detection GUI using venv...
echo.

venv\Scripts\python.exe -X utf8 gui_app.py
