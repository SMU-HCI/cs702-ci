#!/bin/bash

echo "Updating system and installing Java..."
export DEBIAN_FRONTEND=noninteractive
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
    openjdk-17-jre-headless \
    curl \
    ca-certificates \
    tar 


ARCH=$(uname -m)
VERSION="4.9"

if [ "$ARCH" = "x86_64" ]; then
    PRISM_ARCH="linux64-x86"
elif [ "$ARCH" = "aarch64" ]; then
    PRISM_ARCH="linux64-arm"
else
    echo "Unsupported architecture: $ARCH"
    exit 1
fi

# Remove any previous installation
sudo rm -rf /opt/prism
cd /workspaces/cs702-ci
rm -f prism.tar.gz

# Download and Install
URL="https://github.com/prismmodelchecker/prism/releases/download/v$VERSION/prism-$VERSION-$PRISM_ARCH.tar.gz"

echo "Downloading PRISM for $ARCH..."
curl -L -o prism.tar.gz "$URL"
tar -xzf prism.tar.gz

SOURCE_FOLDER=$(ls -d prism-$VERSION-*)
sudo mv "$SOURCE_FOLDER" /opt/prism

# Run PRISM's internal installer
echo "Finalizing PRISM installation..."
cd /opt/prism
sudo ./install.sh

# Set up the Environment Path permanently for the user
if ! grep -q "/opt/prism/bin" ~/.bashrc; then
    echo 'export PATH="/opt/prism/bin:$PATH"' >> ~/.bashrc
    echo "Added PRISM to PATH in ~/.bashrc"
fi

# Clean up
rm prism.tar.gz

echo "----------------------------------------"
echo "Installation Complete!"
echo "Please run: source ~/.bashrc"
echo "Then verify with: prism -version"
echo "----------------------------------------"