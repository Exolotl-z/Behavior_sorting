@echo off
chcp 65001 > nul
echo ========================================
echo 安装程序依赖包
echo ========================================
echo.

cd /d "%~dp0"

echo 正在安装依赖包，请稍候...
echo.

pip install -r requirements.txt

if %errorlevel% equ 0 (
    echo.
    echo ========================================
    echo 依赖包安装完成！
    echo 现在可以运行程序了。
    echo ========================================
) else (
    echo.
    echo ========================================
    echo 安装过程中出现错误！
    echo 请检查网络连接或Python环境。
    echo ========================================
)

echo.
pause
