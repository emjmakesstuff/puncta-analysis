#!/bin/bash
# One-time setup for Puncta Analysis (Mac).
# Creates a virtual environment and installs the tool + dependencies.

cd "$(dirname "$0")"           # go to this script's folder

echo "==================================================="
echo "  Setting up Puncta Analysis..."
echo "  This may take a few minutes. Please wait."
echo "==================================================="

# Create the virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate and install
source .venv/bin/activate
echo "Installing packages..."
pip install --upgrade pip
pip install -e .

echo ""
echo "==================================================="
echo "  ✓ Setup complete!"
echo "  You can now double-click 'launch_mac.command'"
echo "  to start the app."
echo "==================================================="
echo ""
echo "Press any key to close this window."
read -n 1