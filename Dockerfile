FROM python:3.8-slim

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# CUDA 12.1 PyTorch build first (verified combination: torch==2.4.1+cu121, cudnn 90100)
RUN pip install --no-cache-dir torch==2.4.1 torchvision --index-url https://download.pytorch.org/whl/cu121
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Run with: docker run --gpus all -it -v /path/to/datasets:/data <image> bash
# Set DIPOISON_DATA_ROOT to wherever the NQ/HotpotQA/MS MARCO corpora are mounted, e.g.:
#   docker run --gpus all -e DIPOISON_DATA_ROOT=/data -v /path/to/datasets:/data -it <image> bash
CMD ["bash"]
