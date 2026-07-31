@echo off
echo ==========================================
echo Instalando dependencias de SPARK (Global)...
echo ==========================================

:: Comprueba si Python esta instalado
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python no esta instalado o no esta en el PATH.
    echo Por favor, instala Python e intentalo de nuevo.
    pause
    exit /b
)

:: Instala las dependencias globalmente
echo Actualizando pip...
python -m pip install --upgrade pip

echo Instalando librerias desde requirements.txt...
pip install -r requirements.txt

echo.
echo ==========================================
echo Instalacion completada con exito!
echo ==========================================
pause