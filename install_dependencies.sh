#!/bin/bash

# Exit on error
set -e

echo "Installing uv..."
pip install uv==0.5.16 || { echo "Failed to install uv"; exit 1; }

echo "Installing Python packages from requirements.txt..."
uv pip install -r requirements.txt || { echo "Failed to install Python packages"; exit 1; }

echo "Installing IPOPT solver..."
conda install -y -c conda-forge ipopt || { echo "Failed to install IPOPT"; exit 1; }

echo "Installing Spot..."
conda install -y -c conda-forge spot
echo "All dependencies installed successfully!"