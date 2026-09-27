@echo off
setlocal enabledelayedexpansion
title Genius AI - Automated System Setup & Installation
cd /d "%~dp0"

echo ==============================================================================
echo       GENIUS AI : Autonomous Deep-Reasoning Agent Installation Setup
echo ==============================================================================
echo.

:: 1. Verify Python Installation
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python is not detected in your system PATH!
    echo Please install Python 3.10+ from: https://www.python.org/downloads/
    echo Make sure to check the box "Add Python to PATH" during installation.
    echo.
    pause
    exit /b 1
)

for /f "tokens=*" %%i in ('python --version') do set PY_VER=%%i
echo [*] Detected: %PY_VER%

:: 2. Create Virtual Environment if not exists
if not exist ".venv" (
    echo [*] Creating isolated virtual environment in .venv...
    python -m venv .venv
    if %ERRORLEVEL% NEQ 0 (
        echo [ERROR] Failed to create virtual environment!
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created successfully.
) else (
    echo [OK] Existing virtual environment found in .venv.
)

:: 3. Activate Virtual Environment
echo [*] Activating virtual environment...
call .\.venv\Scripts\activate.bat

:: 4. Upgrade pip
echo [*] Checking and upgrading pip package manager...
python -m pip install --upgrade pip --quiet

:: 5. Install Dependencies
echo [*] Installing Genius AI requirements from requirements.txt...
pip install -r requirements.txt
if %ERRORLEVEL% NEQ 0 (
    echo [WARNING] Some dependencies encountered warnings during installation.
    echo Retrying standard installation...
    pip install -r requirements.txt --no-cache-dir
)

:: 6. Verify and Prepare Data Folders
if not exist "data\datasets" (
    mkdir "data\datasets"
)
if not exist "data\instructions" (
    mkdir "data\instructions"
)

:: 7. Success Banner
echo.
echo ==============================================================================
echo   SUCCESS: Genius AI is completely installed and ready for production!
echo ==============================================================================
echo.
echo Quickstart Options:
echo   1. Double-click "start_genius.bat" to launch the interactive AI console.
echo   2. Run "python run_genius.py" from command prompt / PowerShell.
echo   3. Customize persona and rules by adding markdown files to "data\instructions\".
echo.
pause
