@echo off
title Genius: Autonomous Deep-Reasoning AI (xThinking + Wikipedia)
cd /d "%~dp0"
echo ========================================================
echo   Launching Genius: Autonomous Deep-Reasoning Agent
echo   Model: Qwen2.5-0.5B-Chat ^| Multilingual Engine
echo ========================================================
echo.
.\.venv\Scripts\python.exe run_genius.py
pause
