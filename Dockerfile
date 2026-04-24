# Use CUDA base image (Single stage to keep build-essential for runtime compilation)
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

# Consolidated environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_PREFER_BINARY=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    PYTHONUNBUFFERED=1 \
    CMAKE_BUILD_PARALLEL_LEVEL=8

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
    pip install --no-cache-dir \
        torch==2.9.1+cu128 \
        torchvision==0.24.1+cu128 \
        torchaudio==2.9.1+cu128 \
        --index-url https://download.pytorch.org/whl/cu128

# 3. Core Build Tooling & Specified Version Requirements
# Consolidated list including torch-optimi AND pytorch-optimizer
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir \
        setuptools wheel ninja packaging triton==3.5.1 \
        jupyterlab jupyter-server ipykernel \
        deepspeed==0.18.4 \
        "diffusers>=0.35.1" \
        torch-optimi \
        pytorch-optimizer \
        transformers \
        peft \
        accelerate \
        bitsandbytes \
        safetensors \
        sentencepiece \
        protobuf \
        toml datasets pillow tqdm tensorboard \
        imageio[ffmpeg] av einops loguru omegaconf \
        iopath termcolor hydra-core easydict ftfy \
        wandb optimum-quanto scipy \
        comfy-kitchen comfy-aimdo

RUN curl -fsSL https://rclone.org/install.sh -o /tmp/rclone_install.sh && \
    bash /tmp/rclone_install.sh && \
    rm /tmp/rclone_install.sh && \
    curl https://getcroc.schollz.com | bash

# 4. Clone Repositories
RUN git config --global advice.detachedHead false && \
    git clone --depth 1 --recurse-submodules https://github.com/tdrussell/diffusion-pipe /diffusion_pipe && \
    git clone --depth 1 --recursive https://github.com/kohya-ss/musubi-tuner.git /musubi-tuner

# 5. diffusion-pipe setup
RUN --mount=type=cache,target=/root/.cache/pip \
    cd /diffusion_pipe && \
    grep -viE "flash[-_]?attn|flash[-_]?attention" requirements.txt > /tmp/req.txt && \
    pip install --progress-bar off -v -r /tmp/req.txt && \
    rm /tmp/req.txt

# 6. Musubi-Tuner Finalization
RUN cd /musubi-tuner && \
    pip install --no-cache-dir \
        voluptuous==0.16.0 \
        opencv-python==4.11.0.86 \
        six \
        "huggingface_hub[cli,hf_transfer]>=1.3.4" \
        hf_xet \
        prodigyopt \
        timm \
        pydantic && \
    pip install -e . --no-deps

# 7. OneTrainer Setup (The Lean Hybrid Venv)
ENV OT_PREFER_VENV="true" \
    OT_PYTHON_VENV="venv" \
    OT_PYTHON_CMD="python3"

RUN git clone --depth 1 --recursive https://github.com/Nerogar/OneTrainer.git /OneTrainer && \
    cd /OneTrainer && \
    # Key Change: Allow access to global torch/torchvision
    python3 -m venv venv --system-site-packages && \
    ./venv/bin/pip install --upgrade pip && \
    ./venv/bin/pip install --no-cache-dir -r requirements.txt && \
    chmod +x *.sh scripts/*.py

# 8. Final Assets & Entrypoint
COPY src/start_script.sh /start_script.sh
COPY docker-entrypoint.sh /docker-entrypoint.sh

ENV HF_HUB_ENABLE_HF_TRANSFER=1
RUN chmod +x /start_script.sh /docker-entrypoint.sh

ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["/start_script.sh"]