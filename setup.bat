@echo off
echo ==========================================
echo       2D Hydro Model Setup Script
echo ==========================================

echo [1/3] Checking Python...
python --version
if %errorlevel% neq 0 (
    echo Python not found! Please install Python 3.8+ and add it to PATH.
    pause
    exit /b 1
)

echo [2/3] Installing Python Dependencies...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo.
    echo [WARNING] Some dependencies failed to install. 
    echo If 'rasterio' failed, try installing it via Conda or download a pre-built wheel.
    echo.
)

echo [3/3] Compiling C++ Core Modules...
if exist "build.bat" (
    call build.bat
    if %errorlevel% neq 0 (
        echo.
        echo [ERROR] C++ Compilation Failed.
        echo Please ensure Visual Studio 2019/2022 C++ Build Tools are installed.
        echo You may need to edit 'build.bat' to point to your VsDevCmd.bat location.
    ) else (
        echo C++ Modules Compiled Successfully.
    )
) else (
    echo 'build.bat' not found! Skipping compilation.
)

echo.
echo ==========================================
echo             Setup Finished
echo ==========================================
echo.
pause
