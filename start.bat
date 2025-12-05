@echo off
REM HFT Trading Bot - One-Click Starter für Windows

echo.
echo ╔══════════════════════════════════════════════════════╗
echo ║       🚀 HFT Trading Bot - Auto Starter 🚀          ║
echo ╚══════════════════════════════════════════════════════╝
echo.

REM Schritt 1: Python Dependencies
echo [1/5] Aktualisiere Python Dependencies...
poetry install --no-interaction
echo ✓ Python Dependencies aktualisiert
echo.

REM Schritt 2: Frontend Dependencies
echo [2/5] Aktualisiere Frontend Dependencies...
cd frontend
if not exist "node_modules" (
    echo   Installiere Node Modules...
    call npm install
) else (
    echo   Node Modules bereits installiert
)
cd ..
echo ✓ Frontend Dependencies aktualisiert
echo.

REM Schritt 3: Alte Prozesse beenden
echo [3/5] Beende alte Prozesse...
taskkill /F /IM python.exe /FI "WINDOWTITLE eq *uvicorn*" 2>nul
taskkill /F /IM node.exe /FI "WINDOWTITLE eq *vite*" 2>nul
timeout /t 2 >nul
echo ✓ Alte Prozesse beendet
echo.

REM Schritt 4: Backend starten
echo [4/5] Starte Backend...
start "HFT Backend" poetry run uvicorn backend.api.main:app --host 0.0.0.0 --port 8000 --reload
echo   Backend wird gestartet...
timeout /t 3 >nul
echo ✓ Backend läuft auf http://localhost:8000
echo.

REM Schritt 5: Frontend starten
echo [5/5] Starte Frontend...
cd frontend
start "HFT Frontend" npm run dev
cd ..
echo   Frontend wird gestartet...
timeout /t 3 >nul
echo ✓ Frontend läuft auf http://localhost:5173
echo.

REM Status anzeigen
echo ╔══════════════════════════════════════════════════════╗
echo ║              ✅ System erfolgreich gestartet!         ║
echo ╚══════════════════════════════════════════════════════╝
echo.
echo 📊 Dashboard:       http://localhost:5173
echo 📚 API Docs:        http://localhost:8000/docs
echo 💚 Health Check:    http://localhost:8000/api/health
echo.
echo 🛑 Stoppen:         stop.bat
echo.

REM Browser öffnen
timeout /t 2 >nul
start http://localhost:5173

echo ✨ HFT Trading Bot läuft! Viel Erfolg beim Trading! ✨
echo.
pause
