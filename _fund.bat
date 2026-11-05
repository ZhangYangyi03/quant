@echo off
D:
cd "%~dp0" || echo CDFAIL> "%~dp0_fund_err.log"
echo START>> "%~dp0_fund_err.log"
python -B fundpull.py 600
echo END %ERRORLEVEL%>> "%~dp0_fund_err.log"
