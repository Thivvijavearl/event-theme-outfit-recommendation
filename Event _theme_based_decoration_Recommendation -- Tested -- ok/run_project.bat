@echo off
echo Starting StyleMate Project...

echo Starting Flask Backend...
start "" cmd /k "cd /d \"%~dp0\" && python backend\MobileApi_Color.py"

timeout /t 5 /nobreak > nul

echo Starting React Frontend...
start "" cmd /k "cd /d \"%~dp0frontend\" && npm run dev"

echo Both servers are starting. Frontend will be at http://localhost:5174/, Backend at http://localhost:5002/
pause