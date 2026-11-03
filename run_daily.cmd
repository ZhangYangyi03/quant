@echo off
cd /d "%~dp0"
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
python run_daily.py >> daily.out.log 2>&1
