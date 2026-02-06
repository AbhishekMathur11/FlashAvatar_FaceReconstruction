# SurfAvatars – pip/venv, no conda. Persistent container + docker_ws mount at runtime.
FROM nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    ca-certificates \
    python3.8 \
    python3.8-venv \
    python3.8-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    x11-xserver-utils \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1

RUN python3.8 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --upgrade pip setuptools wheel

RUN set -e; \
    mkdir -p /tmp/submodules; \
    git clone --depth 1 https://github.com/camenduru/simple-knn /tmp/submodules/simple-knn; \
    git clone https://github.com/graphdeco-inria/diff-gaussian-rasterization /tmp/submodules/diff-gaussian-rasterization; \
    (cd /tmp/submodules/diff-gaussian-rasterization && git submodule update --init --recursive); \
    true

RUN pip install --no-cache-dir torch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 \
    --extra-index-url https://download.pytorch.org/whl/cu116

ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6"
RUN pip install --no-cache-dir ninja \
    && pip install --no-cache-dir /tmp/submodules/diff-gaussian-rasterization \
    && pip install --no-cache-dir /tmp/submodules/simple-knn \
    && rm -rf /tmp/submodules

RUN pip install --no-cache-dir \
    scipy "chumpy==0.70" scikit-image opencv-python lpips loguru \
    plyfile tqdm yacs "absl-py~=2.3" mediapipe

RUN pip install --no-cache-dir "numpy>=1.21,<1.24"

# PyTorch3D: use CUB from CUDA toolkit to avoid Thrust/CUB version clash (do not use external CUB)
ENV CUB_HOME=/usr/local/cuda/include
ENV MAX_JOBS=2
ENV FORCE_CUDA=1
RUN pip install --no-cache-dir fvcore iopath \
    && pip install --no-cache-dir "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.1"

ENV DISPLAY=:0
RUN mkdir -p /root/persistent /workspace
WORKDIR /workspace

RUN echo 'source /opt/venv/bin/activate 2>/dev/null || true' >> /root/.bashrc

ENTRYPOINT ["/bin/bash"]
CMD ["-l"]
