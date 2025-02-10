FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy project files
COPY . /app

# Install Python dependencies
RUN pip3 install --upgrade pip && pip3 install -r requirements.txt

# Expose any ports if needed (for example, for notebooks)
EXPOSE 8888

CMD ["python3", "triton/multi_gpu_flash_attention_triton.py"]
