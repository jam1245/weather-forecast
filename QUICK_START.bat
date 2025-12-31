@echo off
REM ============================================================================
REM Washington DC Weather Dashboard - Quick Start (Windows)
REM
REM This script launches the dashboard WITHOUT generating new forecasts.
REM It uses existing data files for faster startup.
REM
REM Perfect for:
REM   - Quick testing during development
REM   - Viewing existing data without API calls
REM   - Faster launches when data is already fresh
REM
REM Usage:
REM   QUICK_START.bat            - Launch with existing data
REM   QUICK_START.bat --port 8502   - Use custom port
REM
REM For full workflow with forecast generation, use START_APP.bat instead
REM
REM ============================================================================

echo.
echo ================================================================================
echo  Weather Dashboard - QUICK START (Skip Forecast Generation)
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
    pause
    exit /b 1
)

REM Check if run_weather_dashboard.py exists
if not exist "run_weather_dashboard.py" (
    echo [ERROR] run_weather_dashboard.py not found!
    echo.
    echo Make sure you're running this script from the project directory.
    echo.
    pause
    exit /b 1
)

REM Check if data files exist
if not exist "weather_historical_forecast.csv" (
    echo [WARNING] weather_historical_forecast.csv not found!
    echo.
    echo You need to generate data first by running:
    echo   START_APP.bat
    echo.
    echo Or manually run:
    echo   python weather_forecast.py
    echo.
    set /p CONTINUE="Continue anyway? (y/n): "
    if /i not "%CONTINUE%"=="y" (
        echo.
        echo Cancelled. Run START_APP.bat to generate data first.
        pause
        exit /b 1
    )
)

echo [INFO] Launching dashboard with --skip-forecast...
echo [INFO] Using existing data (no forecast generation)
echo.

REM Run orchestrator with --skip-forecast flag and pass any additional arguments
python run_weather_dashboard.py --skip-forecast %*

REM Check if orchestrator failed
if errorlevel 1 (
    echo.
    echo ================================================================================
    echo  Quick Start Failed
    echo ================================================================================
    echo.
    echo Check the output above for error details.
    echo.
    echo Try running full workflow:
    echo   START_APP.bat
    echo.
    pause
    exit /b 1
)

REM Normal exit
echo.
echo ================================================================================
echo  Dashboard Stopped
echo ================================================================================
echo.
echo To run again:
echo   QUICK_START.bat       - Quick start (skip forecast)
echo   START_APP.bat         - Full workflow (with forecast)
echo.
pause
exit /b 0
