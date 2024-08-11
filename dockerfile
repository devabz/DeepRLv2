# Stage 1: Build stage to install Conda, set up the environment, and install all dependencies
FROM continuumio/miniconda3 AS build-env

# Set the working directory
WORKDIR /app

# Copy environment.yml to the working directory
COPY environment.yml .
COPY requirements.txt .
COPY scripts/setup.sh .

# Create environent
RUN conda create -n td3-v1 -c conda-forge python=3.9 -y

RUN chmod +x ./setup.sh
RUN ./setup.sh

# Set the entry environment and restart terminal 
#RUN /bin/bash -c "echo 'conda activate td3-v1' >> ~/.bashrc && source ~/.bashrc && pip install gymnasium-robotics"

#RUN /bin/bash -c "pip install gymnasium-robotics"
#RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

#SHELL ["conda", "run", "-n", "your_env_name", "/bin/bash", "-c"]
