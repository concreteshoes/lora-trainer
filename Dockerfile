# Copyright (C) 2026 <ByteSizeLife>
# Licensed under AGPL-3.0 with additional terms — see LICENSE for details.
# Commercial redistribution of this image or derivative works is prohibited
# without explicit written permission from the author.

# Use CUDA base image (Single stage to keep build-essential for runtime compilation)
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

# Consolidated environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_PREFER_BINARY=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    PYTHONUNBUFFERED=1 \
    CMAKE_BUILD_PARALLEL_LEVEL=8 \
    PIP_TIMEOUT=100

# 1. System Dependencies & SSH Setup
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv python3-dev python3-tk tk-dev libx11-6 libxext6 \
        curl zip unzip git git-lfs wget vim libgl1 libglib2.0-0 libgoogle-perftools4 \
        libjpeg-dev libpng-dev libwebp-dev libtiff-dev liblcms2-dev ffmpeg \
        build-essential gcc rsync openssh-server aria2 tmux && \
    \
    # Python defaults
    ln -sf /usr/bin/python3 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip && \
    \
    # SSH Config
    mkdir -p /root/.ssh /var/run/sshd && \
    chmod 700 /root/.ssh && \
    sed -i 's/^#\?PasswordAuthentication .*/PasswordAuthentication no/' /etc/ssh/sshd_config && \
    sed -i 's/^#\?PermitRootLogin .*/PermitRootLogin prohibit-password/' /etc/ssh/sshd_config && \
    \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# 2. Stable PyTorch Stack
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install \
        torch==2.9.1+cu128 \
        torchvision==0.24.1+cu128 \
        torchaudio==2.9.1+cu128 \
        --index-url https://download.pytorch.org/whl/cu128

# 3. Core Build Tooling & Specified Version Requirements
# Consolidated list including torch-optimi AND pytorch-optimizer
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install \
        setuptools wheel ninja packaging triton==3.5.1 \
        jupyterlab jupyter-server ipykernel \
        deepspeed==0.18.4 \
        "diffusers>=0.35.1" \
        torch-optimi \
        pytorch-optimizer \
        transformers \
        peft \
        accelerate \
        onnxruntime-gpu \
        bitsandbytes \
        safetensors \
        sentencepiece \
        protobuf \
        toml datasets pillow tqdm tensorboard \
        imageio[ffmpeg] av einops loguru omegaconf \
        iopath termcolor hydra-core easydict ftfy \
        wandb optimum-quanto scipy \
        comfy-kitchen comfy-aimdo

# 4. Install Rclone & Filebrowser
RUN curl -fsSL https://rclone.org/install.sh -o /tmp/rclone_install.sh && \
    bash /tmp/rclone_install.sh && \
    rm /tmp/rclone_install.sh && \
    \
    # Install Filebrowser binary (The script auto-installs to /usr/local/bin/)
    curl -fsSL https://raw.githubusercontent.com/filebrowser/get/master/get.sh | bash

# 5. Clone Repositories
RUN git config --global advice.detachedHead false && \
    git clone --depth 1 --recurse-submodules https://github.com/tdrussell/diffusion-pipe /diffusion_pipe && \
    git clone --depth 1 --recursive https://github.com/kohya-ss/musubi-tuner.git /musubi-tuner

# 6. diffusion-pipe setup
RUN --mount=type=cache,target=/root/.cache/pip \
    cd /diffusion_pipe && \
    grep -viE "flash[-_]?attn|flash[-_]?attention" requirements.txt > /tmp/req.txt && \
    pip install --progress-bar off -v -r /tmp/req.txt && \
    rm /tmp/req.txt

# 7. Musubi-Tuner Finalization
RUN --mount=type=cache,target=/root/.cache/pip \
    cd /musubi-tuner && \
    pip install \
        voluptuous==0.16.0 \
        opencv-python-headless==4.11.0.86 \
        six \
        "huggingface_hub[cli,hf_transfer]" \
        hf_xet \
        prodigyopt \
        timm \
        pydantic && \
    pip install -e . --no-deps

# 8. OneTrainer Setup (The Lean Hybrid Venv)
ENV OT_PREFER_VENV="true" \
    OT_PYTHON_VENV="venv" \
    OT_PYTHON_CMD="python3"

RUN --mount=type=cache,target=/root/.cache/pip \
    git clone --depth 1 --recursive https://github.com/Nerogar/OneTrainer.git /OneTrainer && \
    cd /OneTrainer && \
    \
    # 1. Create venv with access to our high-perf Torch/BitsAndBytes
    python3 -m venv venv --system-site-packages && \
    \
    # 2. Refined Global Requirements Patching
    # Remove editable flags (-e) so git packages install cleanly into site-packages
    sed -i 's/^-e //' requirements-global.txt && \
    \
    # Remove Diffusers git branch (forces use of your global diffusers>=0.35.1)
    sed -i '/github.com\/huggingface\/diffusers/d' requirements-global.txt && \
    \
    # Force headless OpenCV to prevent X11 dependencies from bloating the venv
    sed -i 's/^opencv-python==.*/opencv-contrib-python-headless/' requirements-global.txt && \
    \
    # Targeted unpinning: Only unpin packages that we WANT to inherit from the system.
    # This prevents venv bloat while keeping UI/Optimizers strictly pinned.
    sed -i -E '/^(numpy|pillow|tqdm|scipy|av|setuptools|accelerate|safetensors|tensorboard|transformers|sentencepiece|omegaconf|pytorch_optimizer|huggingface-hub)==/s/==.*//' requirements-global.txt && \
    \
    # 3. Refined CUDA Requirements Patching
    sed -i '/^torch==\|^torchvision==\|^torchaudio==/d' requirements-cuda.txt && \
    sed -i '/triton-windows/d' requirements-cuda.txt && \
    sed -i -E 's/^bitsandbytes==[^ ]*/bitsandbytes/' requirements-cuda.txt && \
    sed -i -E 's/^onnxruntime-gpu==[^ ]*/onnxruntime-gpu/' requirements-cuda.txt && \
    sed -i '/^nvidia-nccl/d' requirements-cuda.txt && \
    sed -i '/^--extra-index-url/d' requirements-cuda.txt && \
    \
    # 4. Installation
    ./venv/bin/pip install --upgrade pip && \
    ./venv/bin/pip install -r requirements.txt && \
    \
    chmod +x *.sh scripts/*.py

# 9. Final Assets & Entrypoint
COPY src/start_script.sh /start_script.sh
COPY docker-entrypoint.sh /docker-entrypoint.sh

ENV HF_HUB_ENABLE_HF_TRANSFER=1
ENV HF_HUB_DISABLE_UPDATE_AT_LAUNCH=1

RUN chmod +x /start_script.sh /docker-entrypoint.sh

ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["/start_script.sh"]