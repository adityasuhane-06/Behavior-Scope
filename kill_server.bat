@echo off
echo Stopping API server on port 8000...

REM Find the process using port 8000
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :8000 ^| findstr LISTENING') do (
    echo Found process: %%a
    taskkill /F /PID %%a
)

echo Done! You can now start the server again.
pause