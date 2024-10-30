# Use the NVIDIA Triton Inference Server base image with Python support
FROM nvcr.io/nvidia/tritonserver:23.10-py3

# Set the working directory to the Triton model repository
WORKDIR /models

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip to the latest version
RUN pip install --upgrade pip
RUN pip install Cython
# Install necessary Python packages
RUN pip install nemo_toolkit\[all\] pydub
RUN apt-get update && \
    apt-get install -y ffmpeg
RUN pip uninstall huggingface_hub -y
RUN pip install huggingface-hub==0.23.2
