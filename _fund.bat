@echo off
D:
cd "%~dp0" || echo CDFAIL> "D:\Users\china\Desktop\项目_开发\quant\_fund_err.log"
echo START>> "D:\Users\china\Desktop\项目_开发\quant\_fund_err.log"
"C:\Users\china\miniconda3\python.exe" -B fundpull.py 600
echo END %ERRORLEVEL%>> "D:\Users\china\Desktop\项目_开发\quant\_fund_err.log"
