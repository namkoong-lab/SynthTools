#!/bin/bash

set -x

# Setup venv directory to be current files' folder
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "Script directory is $SCRIPT_DIR"

VENV_DIR="$SCRIPT_DIR/.venv"

echo "Creating virtual environment in $VENV_DIR"
uv venv $VENV_DIR --python 3.12.7

echo "Activating virtual environment..."
source $VENV_DIR/bin/activate

echo "Installing required packages..."
uv pip sync requirements.txt
uv pip install -e .

echo "Setup complete! Run 'source $VENV_DIR/bin/activate' to activate the virtual environment."