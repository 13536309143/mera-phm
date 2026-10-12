@echo off
setlocal
cd /d "%~dp0"
where py >nul 2>nul
if errorlevel 1 (
  python scripts\launch.py --mode api %*
) else (
  py -3 scripts\launch.py --mode api %*
)
if errorlevel 1 pause
