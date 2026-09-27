@echo off
title Genius: Autonomous Deep-Reasoning AI (xThinking + Wikipedia)
cd /d "%~dp0"
echo ========================================================
echo   Launching Genius: Autonomous Deep-Reasoning Agent
echo   Model: Dual-Core Qwen2.5 / Edge / Claude / Ollama
echo ========================================================
echo.

if exist ".\.venv\Scripts\python.exe" (
    .\.venv\Scripts\python.exe run_genius.py %*
) else (
    python run_genius.py %*
)
if %ERRORLEVEL% NEQ 0 pause
