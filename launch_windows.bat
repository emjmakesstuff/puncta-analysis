@echo off
REM Double-clickable launcher for Windows.
cd /d "%~dp0"
call .venv\Scripts\activate
streamlit run app.py
pause