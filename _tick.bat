@echo off
D:
cd "%~dp0" 2> "%~dp0_tick_err.log"
echo CWD=%CD%>> "%~dp0_tick_err.log"
python -B quant_tick.py 8 2>> "%~dp0_tick_err.log"
echo END %ERRORLEVEL% >> "%~dp0_tick_err.log"
