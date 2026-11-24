@echo off
setlocal
cd /d "%~dp0"
py -m pip install -r requirements.txt
if errorlevel 1 exit /b 1
py vocab_atlas_15.py --out results
if errorlevel 1 exit /b 1
echo.
echo Finished. Open results\REPORT.md and results\overlap_heatmap.png
pause
