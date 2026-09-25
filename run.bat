@echo off
rem One-command launcher (Windows). Double-click it, or from the project folder run:
rem   run.bat            -> start the desktop app
rem   run.bat draw       -> click-to-draw zones in an OpenCV window (extra args are passed on,
rem                         e.g.  run.bat draw --source 0)
rem   run.bat test       -> run the test suite
rem   run.bat diagnose   -> check what is wrong if the app won't start (writes diagnose_report.txt)
rem First run creates .venv and installs requirements (takes a few minutes); later runs skip that.
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment...
    python -m venv .venv || (echo Could not create the virtual environment. Is Python installed and on PATH? & pause & exit /b 1)
)
call ".venv\Scripts\activate.bat"

if not exist ".venv\.deps_installed_v2" (
    echo Installing requirements...
    pip install -r requirements.txt || (echo Install failed. & pause & exit /b 1)
    type nul > ".venv\.deps_installed_v2"
)

if /i "%~1"=="test" (
    python -m unittest discover -s tests
    goto :end
)
if /i "%~1"=="diagnose" (
    python diagnose.py
    pause
    goto :end
)
if /i "%~1"=="draw" (
    shift
    python draw_zones.py %1 %2 %3 %4 %5 %6
    goto :end
)
python -X faulthandler -m desktop_app.main
if errorlevel 1 (
    echo.
    echo The app exited with an error - see the messages above. Run  run.bat diagnose  for a full check.
    pause
)

:end
