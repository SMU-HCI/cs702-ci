#!/bin/bash

# Exit on error
set -e

echo "Installing uv..."
pip install uv || { echo "Failed to install uv"; exit 1; }

echo "Installing Python packages from pyproject.toml..."
python -c "import tomllib; print(' '.join(tomllib.load(open('pyproject.toml', 'rb'))['project']['dependencies']))" | xargs uv pip install || { echo "Failed to install Python packages"; exit 1; }

echo "Installing IPOPT solver..."
conda install -y -c conda-forge ipopt || { echo "Failed to install IPOPT"; exit 1; }

echo "Installing Spot..."
conda install -y -c conda-forge spot
echo "All dependencies installed successfully!"