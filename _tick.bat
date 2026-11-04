@echo off
D:
cd "%~dp0" 2> "D:\Users\china\Desktop\项目_开发\quant\_tick_err.log"
echo CWD=%CD%>> "D:\Users\china\Desktop\项目_开发\quant\_tick_err.log"
"C:\Users\china\miniconda3\python.exe" -B quant_tick.py 8 2>> "D:\Users\china\Desktop\项目_开发\quant\_tick_err.log"
echo END %ERRORLEVEL% >> "D:\Users\china\Desktop\项目_开发\quant\_tick_err.log"
