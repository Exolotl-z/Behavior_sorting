@echo off
chcp 65001 > nul
echo ========================================
echo 鼠标行为标记系统
echo ========================================
echo.
echo 正在启动程序...
echo.

cd /d "%~dp0"
python Behavior_sorting\Behavior_sorting\enhanced_main.py

if %errorlevel% neq 0 (
    echo.
    echo 程序运行出错！
    echo 请确保已安装所有依赖包。
    echo 可以运行 install_dependencies.bat 来安装依赖。
    echo.
    pause
)
