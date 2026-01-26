@echo off
echo ================================================
echo Behavior Scope Dashboard Launcher
echo ================================================
echo.
echo This script will:
echo 1. Start the FastAPI backend server
echo 2. Open a new window for the React frontend
echo.
echo Press any key to continue or Ctrl+C to cancel...
pause > nul

echo.
echo Starting backend API server...
echo Backend will run at: http://localhost:8000
echo.

start "Behavior Scope Frontend" cmd /k "cd dashboard && echo Starting React dashboard... && npm run dev"

timeout /t 2 /nobreak > nul

python start_dashboard.py

pause
