@echo off
chcp 65001 > nul
echo ========================================
echo 数据分析工具
echo ========================================
echo.
echo 此工具用于分析已保存的CSV数据文件
echo 无需重新打开视频
echo.

cd /d "%~dp0"

python Behavior_sorting\Behavior_sorting\data_analyzer.py

if %errorlevel% neq 0 (
    echo.
    echo 程序运行出错！
    echo.
    pause
)
