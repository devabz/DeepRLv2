#!/bin/bash

# Activate the Conda environment
echo 'conda activate td3-v1' >> ~/.bashrc

# Restart terminal
source ~/.bashrc

# Install additional Python packages
pip install gymnasium-robotics
#pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118