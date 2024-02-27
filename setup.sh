#!/bin/bash

# Update and Upgrade the system
echo "Updating and upgrading your system..."
apt-get update && sudo apt-get upgrade -y

# Install Python3 and Python3-pip if they are not installed
echo "Checking for Python3 installation..."
if ! command -v python3 &> /dev/null
then
    echo "Python3 could not be found, installing now..."
    apt-get install python3 -y
else
    echo "Python3 is already installed."
fi

echo "Checking for pip installation..."
if ! command -v pip3 &> /dev/null
then
    echo "pip for Python3 could not be found, installing now..."
    apt-get install python3-pip -y
else
    echo "pip for Python3 is already installed."
fi

# Install virtualenv via pip if not installed
echo "Installing virtualenv..."
pip3 install virtualenv

# Create a Python virtual environment
echo "Creating a virtual environment..."
virtualenv myenv

# Activate the virtual environment
echo "Activating the virtual environment..."
source myenv/bin/activate

# Install necessary Python packages within the virtual environment
echo "Installing necessary Python packages..."
pip install scikit-learn pandas numpy shap lime

# Install Rapids - Note: this requires NVIDIA CUDA, check compatibility
echo "Attempting to install RAPIDS AI..."
# Note: RAPIDS installation commands go here. Due to the complexity and specific requirements (like CUDA version),
# it's recommended to follow the official RAPIDS installation guide for this part.
echo "Setup is complete. Don't forget to activate the virtual environment with 'source myenv/bin/activate' before using it."


