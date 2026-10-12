@echo off
setlocal
cd /d "%~dp0"
where py >nul 2>nul
if errorlevel 1 (
  python scripts\launch.py --mode local %*
) else (
  py -3 scripts\launch.py --mode local %*
)
if errorlevel 1 pause
