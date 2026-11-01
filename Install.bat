@echo off
echo ==========================================
echo Instalando dependencias de SPARK (Global)...
echo ==========================================

:: Comprueba si Python esta instalado
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or is not in the PATH.
    echo Please, install Python and try again.
    pause
    exit /b
)

:: Instala las dependencias globalmente
echo Updating pip...
python -m pip install --upgrade pip

echo Installing packages from requirements.txt...
pip install -r requirements.txt

echo.
echo ==========================================
echo Installation completed!
echo ==========================================
pause
