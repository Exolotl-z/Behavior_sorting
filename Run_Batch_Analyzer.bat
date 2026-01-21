@echo off
chcp 65001 > nul
echo ========================================
echo Batch Behavior Analyzer
echo ========================================
echo.
echo Starting batch analysis tool...
echo.

cd /d "%~dp0"

python Behavior_sorting\Behavior_sorting\batch_analyzer.py

if %errorlevel% neq 0 (
    echo.
    echo Error running program!
    echo.
    pause
)
