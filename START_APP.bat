@echo off
REM ============================================================================
REM Washington DC Weather Dashboard - Windows Launcher
REM
REM This script runs the orchestrator which automates:
REM   1. Dependency checking
REM   2. Weather forecast generation
REM   3. Dashboard launch
REM
REM Usage:
REM   START_APP.bat              - Full workflow (with forecast generation)
REM   START_APP.bat --skip-forecast  - Skip forecast, use existing data
REM   START_APP.bat --port 8502      - Use custom port
REM
REM ============================================================================

echo.
echo ================================================================================
echo  Washington DC Weather Dashboard - Launcher
echo ================================================================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH
    echo.
    echo Please install Python 3.8 or higher from:
    echo   https://www.python.org/downloads/
    echo.
    echo Make sure to check "Add Python to PATH" during installation
    echo.
    pause
    exit /b 1
)

REM Check if run_weather_dashboard.py exists
if not exist "run_weather_dashboard.py" (
    echo [ERROR] run_weather_dashboard.py not found!
    echo.
    echo Make sure you're running this script from the project directory:
    echo   %CD%
    echo.
    pause
    exit /b 1
)

echo [INFO] Launching Weather Dashboard Orchestrator...
echo.

REM Run the orchestrator with any command-line arguments passed to this script
python run_weather_dashboard.py %*

REM Check if orchestrator failed
if errorlevel 1 (
    echo.
    echo ================================================================================
    echo  Dashboard Launch Failed
    echo ================================================================================
    echo.
    echo The orchestrator encountered an error. Check the output above for details.
    echo.
    echo Common issues:
    echo   - Missing dependencies: Run 'pip install -r requirements.txt'
    echo   - Port already in use: Try 'START_APP.bat --port 8502'
    echo   - Python not found: Make sure Python is in your PATH
    echo.
    pause
    exit /b 1
)

REM If we get here, the app was stopped normally (Ctrl+C)
echo.
echo ================================================================================
echo  Dashboard Stopped
echo ================================================================================
echo.
echo Thank you for using Weather Dashboard!
echo.
echo To run again, double-click START_APP.bat or run:
echo   START_APP.bat
echo.
pause
exit /b 0
