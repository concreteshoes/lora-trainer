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
        python3 python3-pip python3-venv python3-dev \
        curl zip unzip git git-lfs wget vim libgl1 libglib2.0-0 libgoogle-perftools4 \
        libjpeg-dev libpng-dev libwebp-dev libtiff-dev liblcms2-dev ffmpeg \
        build-essential gcc rsync openssh-server aria2 tmux && \
    \
    # Python defaults
    ln -sf /usr/bin/python3 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip && \
    \
    # Surgical SSH Config
    mkdir -p /root/.ssh /var/run/sshd && \
    chmod 700 /root/.ssh && \
    sed -i 's/^#\?PasswordAuthentication .*/PasswordAuthentication no/' /etc/ssh/sshd_config && \
    sed -i 's/^#\?PermitRootLogin .*/PermitRootLogin prohibit-password/' /etc/ssh/sshd_config && \
    \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# 2. Stable PyTorch 2.9 Stack
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir \
        torch==2.9.0+cu128 \
        torchvision==0.24.0+cu128 \
        torchaudio==2.9.0+cu128 \
        --index-url https://download.pytorch.org/whl/cu128

# 3. Core Build Tooling & Heavy Runtime Libs
# DeepSpeed is installed here to ensure it uses the correct Torch/CUDA 12.8 versions
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir \
        setuptools wheel ninja packaging \
        jupyterlab jupyterlab-lsp jupyter-server \
        jupyter-server-terminals ipykernel Pillow jupyterlab_code_formatter \
        tensorboard \
        "peft>=0.17.0" "deepspeed>=0.17.6"

RUN curl -fsSL https://rclone.org/install.sh -o /tmp/rclone_install.sh && \
    bash /tmp/rclone_install.sh && \
    rm /tmp/rclone_install.sh
    
# Install croc for emergency transfers (punches through NAT/Firewalls)
RUN curl https://getcroc.schollz.com | bash
        
# 4. diffusion-pipe Setup (Optimized for speed/size)
RUN git config --global advice.detachedHead false && \
    # Using --depth 1 to skip gigabytes of git history
    git clone --depth 1 --recurse-submodules https://github.com/tdrussell/diffusion-pipe /diffusion_pipe

# Install requirements but skip flash-attn
RUN --mount=type=cache,target=/root/.cache/pip \
    cd /diffusion_pipe && \
    grep -viE "flash[-_]?attn|flash[-_]?attention" requirements.txt > /tmp/req.txt && \
    pip install --progress-bar off -v -r /tmp/req.txt && \
    rm /tmp/req.txt

# 5. Musubi-Tuner Pre-build & Dependencies
RUN git config --global advice.detachedHead false && \
    git clone --depth 1 --recursive https://github.com/kohya-ss/musubi-tuner.git /musubi-tuner && \
    cd /musubi-tuner && \
    # Install the specific versions that musubi-tuner "Dependency Hell" warned about
    pip install --no-cache-dir \
        voluptuous==0.15.2 \
        opencv-python==4.10.0.84 \
        toml \
        einops==0.7.0 \
        protobuf \
        six \
        "huggingface_hub[cli,hf_transfer]==0.34.0" \
        hf_xet \
        prodigyopt \
        bitsandbytes \
        accelerate==1.6.0 \
        sentencepiece \
        timm \
        pydantic \
        av==14.0.1 \
        # These are heavy, best to bake them into the image
        transformers==4.56.1 \
        diffusers==0.32.1 \
        safetensors==0.4.5 && \
    # Install the repo itself in editable mode without re-checking deps
    pip install -e . --no-deps

# 6. Final Assets & Entrypoint
COPY src/start_script.sh /start_script.sh
COPY docker-entrypoint.sh /docker-entrypoint.sh

# Set HF Transfer as default global ENV
ENV HF_HUB_ENABLE_HF_TRANSFER=1

RUN chmod +x /start_script.sh /docker-entrypoint.sh

ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["/start_script.sh"]