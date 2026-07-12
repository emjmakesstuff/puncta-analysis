@echo off
REM One-time setup for Puncta Analysis (Windows).
cd /d "%~dp0"

echo ===================================================
echo   Setting up Puncta Analysis...
echo   This may take a few minutes. Please wait.
echo ===================================================

if not exist ".venv" (
    echo Creating virtual environment...
    python -m venv .venv
)

call .venv\Scripts\activate
echo Installing packages...
python -m pip install --upgrade pip
pip install -e .

echo.
echo ===================================================
echo   Setup complete!
echo   You can now double-click 'launch_windows.bat'
echo   to start the app.
echo ===================================================
pause