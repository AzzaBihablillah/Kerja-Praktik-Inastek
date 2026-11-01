@echo off
REM CADANGAN EXPO: jalankan RVM di laptop (RTX 4050) bila Jetson bermasalah.
REM Kamera C922 = index 1, talang ESP32 = COM7 (otomatis dicatat saja bila tidak ada). Konveyor lewat PLC: area belt harus aman.
REM Tutup jendela ini untuk berhenti.
cd /d "%~dp0.."
start "" cmd /c "timeout /t 10 >nul & start http://localhost:8000/"
venv\Scripts\python -m app --plc
pause
