#!/bin/bash
# Double-clickable launcher for Mac.
# Navigates to the app folder, activates the environment, starts the app.

cd "$(dirname "$0")"           # go to this script's folder
source .venv/bin/activate      # activate the virtual environment
streamlit run app.py           # launch the app (opens browser)