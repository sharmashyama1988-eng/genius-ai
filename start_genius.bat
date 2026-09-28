@echo off
title Genius AI — Smart Autonomous Agent Launcher
color 0b
chcp 65001 >nul 2>&1

:: Move to script directory
cd /d "%~dp0"

echo ======================================================================
echo    ⚡ GENIUS AI  —  Smart Autonomous Agent (Dual-Core Architecture)
echo ======================================================================
echo.

:: Check Python
where python >nul 2>&1
if %errorlevel% neq 0 (
    color 0c
    echo [ERROR] Python was not found in PATH!
    echo Please install Python 3.10+ and add it to your System PATH.
    echo.
    pause
    exit /b 1
)

:: Run Smart API Key Auto-Detector
python -m src.system.api_key_detector
if %errorlevel% neq 0 (
    echo.
    echo [NOTICE] Launcher continuing with active configuration...
)

echo.
echo ======================================================================
echo  Starting Genius AI Autonomous Shell...
echo ======================================================================
echo.

:: Launch Genius CLI with auto-loaded .env
python run_genius.py

if %errorlevel% neq 0 (
    echo.
    echo [EXIT] Genius session ended with exit code %errorlevel%.
    pause
)
