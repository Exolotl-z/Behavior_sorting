@echo off
chcp 65001 > nul
echo ========================================
echo 鼠标行为标记系统 - 打包脚本
echo ========================================
echo.

echo [1/4] 检查Python环境...
python --version
if %errorlevel% neq 0 (
    echo 错误: 未找到Python，请先安装Python
    pause
    exit /b 1
)
echo.

echo [2/4] 安装依赖包...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo 警告: 依赖安装可能出现问题
)
echo.

echo [3/4] 安装PyInstaller...
pip install pyinstaller
if %errorlevel% neq 0 (
    echo 错误: PyInstaller安装失败
    pause
    exit /b 1
)
echo.

echo [4/4] 开始打包程序...
pyinstaller --clean MouseBehaviorAnnotator.spec
if %errorlevel% neq 0 (
    echo 错误: 打包失败
    pause
    exit /b 1
)
echo.

echo ========================================
echo 打包完成！
echo 可执行文件位置: dist\MouseBehaviorAnnotator.exe
echo ========================================
pause
