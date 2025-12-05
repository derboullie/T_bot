@echo off
REM HFT Trading Bot - Stopper Script für Windows

echo Stoppe HFT Trading Bot...
echo.

REM Backend stoppen
taskkill /F /IM python.exe /FI "WINDOWTITLE eq HFT Backend*" 2>nul
if %ERRORLEVEL% EQU 0 (
    echo ✓ Backend gestoppt
) else (
    echo ⚠ Backend läuft nicht
)

REM Frontend stoppen
taskkill /F /IM node.exe /FI "WINDOWTITLE eq HFT Frontend*" 2>nul
if %ERRORLEVEL% EQU 0 (
    echo ✓ Frontend gestoppt
) else (
    echo ⚠ Frontend läuft nicht
)

echo.
echo ✅ HFT Trading Bot gestoppt
pause
