#!/usr/bin/env bash

# Function to check if a directory exists and is writable
can_write_to() {
    local target="$1"
    [ -z "$target" ] && return 1

    if [ -d "$target" ]; then
        touch "$target/.write_test" 2> /dev/null || return 1
        rm -f "$target/.write_test"
    else
        mkdir -p "$target" 2> /dev/null || return 1
        touch "$target/.write_test" 2> /dev/null || return 1
        rm -f "$target/.write_test"
    fi

    return 0
}

# Determine NETWORK_VOLUME
if [ -n "${NETWORK_VOLUME-}" ] && can_write_to "$NETWORK_VOLUME"; then
    echo "Using provided NETWORK_VOLUME: $NETWORK_VOLUME"

elif can_write_to "/workspace"; then
    NETWORK_VOLUME="/workspace"
    echo "Defaulting to /workspace"

elif can_write_to "/runpod-volume"; then
    NETWORK_VOLUME="/runpod-volume"
    echo "Defaulting to /runpod-volume"

else
    NETWORK_VOLUME="$(pwd)"
    echo "Fallback to current dir: $NETWORK_VOLUME"
fi

mkdir -p "$NETWORK_VOLUME"
export NETWORK_VOLUME

# Auto cd on shell login
if [ -n "$NETWORK_VOLUME" ] && [ -d "$NETWORK_VOLUME" ]; then
    grep -qxF "cd \"$NETWORK_VOLUME\"" /root/.bashrc \
        || echo "cd \"$NETWORK_VOLUME\"" >> /root/.bashrc
fi

mkdir -p "$NETWORK_VOLUME/logs"
STARTUP_LOG="$NETWORK_VOLUME/logs/startup.log"
echo "--- Startup log $(date) ---" >> "$STARTUP_LOG"

# Keep-alive loop to prevent connection timeout and monitor DNS
(
    echo "Starting network keep-alive service..."
    while true; do
        # Re-enforce DNS just in case the host overrode it
        echo -e "nameserver 8.8.8.8\nnameserver 1.1.1.1" > /etc/resolv.conf
        TIMESTAMP=$(date "+%Y-%m-%d %H:%M:%S")

        # 1. Try to ping Google Drive's API endpoint
        if curl -Is --connect-timeout 5 https://www.google.com > /dev/null 2>&1; then
            echo "[$TIMESTAMP] Internet: REACHABLE (HTTPS)"
        else
            echo "[$TIMESTAMP] Internet: UNREACHABLE"
            # Fallback to check raw DNS resolution via a simple tool like 'host' or 'nslookup'
            if nslookup google.com > /dev/null 2>&1; then
                echo "[$TIMESTAMP] Alert: DNS works, but HTTPS traffic is failing."
            else
                echo "[$TIMESTAMP] Alert: Total network/DNS failure."
            fi
        fi

        # Wait 15 minutes (900 seconds)
        sleep 900
    done
) > "$NETWORK_VOLUME/logs/network_keepalive.log" 2>&1 &

# Run a command quietly, logging output to STARTUP_LOG.
# Shows "Still working..." every 10 seconds.
# On failure, prints a warning with the log path.
run_quiet() {
    local label="$1"
    shift

    # 1. Log a header so you know which command is starting
    echo "====================================================" >> "$STARTUP_LOG"
    echo "BEGIN: $label ($(date))" >> "$STARTUP_LOG"
    echo "COMMAND: $@" >> "$STARTUP_LOG"
    echo "====================================================" >> "$STARTUP_LOG"

    (
        while true; do
            sleep 10
            echo "       Still working on $label..."
        done
    ) &
    local heartbeat_pid=$!

    # 2. Run command. Adding --progress-bar off for pip specifically
    "$@" >> "$STARTUP_LOG" 2>&1
    local exit_code=$?

    kill "$heartbeat_pid" 2> /dev/null
    wait "$heartbeat_pid" 2> /dev/null

    if [ $exit_code -ne 0 ]; then
        echo "       ❌ Warning: $label failed (Exit Code: $exit_code)."
        echo "       Check the end of $STARTUP_LOG for details."
        echo "END: $label (FAILED)" >> "$STARTUP_LOG"
    else
        echo "END: $label (SUCCESS)" >> "$STARTUP_LOG"
    fi

    echo -e "\n" >> "$STARTUP_LOG" # Add spacing between log entries
    return $exit_code
}

# Helper functions for cleaner output
status_msg() { echo -e "\n---> $1"; }

# Force-enable high-speed downloads for this session
export HF_HUB_ENABLE_HF_TRANSFER=1

# ============================================================
# Try to find full tcmalloc first, fallback to minimal
# ============================================================

TCMALLOC_PATH=$(ldconfig -p 2> /dev/null | grep -E 'libtcmalloc\.so' | head -n1 | awk '{print $NF}')

if [ -z "$TCMALLOC_PATH" ]; then
    TCMALLOC_PATH=$(ldconfig -p 2> /dev/null | grep -E 'libtcmalloc_minimal\.so' | head -n1 | awk '{print $NF}')
fi

# Apply if found
if [ -n "$TCMALLOC_PATH" ]; then
    export LD_PRELOAD="$TCMALLOC_PATH"
    echo "Using tcmalloc: $TCMALLOC_PATH"
else
    echo "tcmalloc not found, skipping LD_PRELOAD"
fi

# ============================================================
# GPU detection
# ============================================================

if command -v nvidia-smi > /dev/null 2>&1; then

    readarray -t GPU_INFO < <(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader 2> /dev/null)

    DETECTED_GPU=$(echo "${GPU_INFO[0]}" | cut -d',' -f1 | xargs)

    CUDA_ARCH=$(printf "%s\n" "${GPU_INFO[@]}" \
        | cut -d',' -f2 \
        | sed 's/\.//g' \
        | sort -u \
        | xargs \
        | tr ' ' ';')

else
    DETECTED_GPU="Unknown GPU"
    CUDA_ARCH="80;86;89;90"
fi

# Final fallback
[ -z "$CUDA_ARCH" ] && CUDA_ARCH="80;86;89;90"

echo "$DETECTED_GPU" > /tmp/detected_gpu

# ============================================================
# Startup banner
# ============================================================
echo ""
echo "================================================"
echo "  Starting up..."
status_msg "Detected GPU: $DETECTED_GPU (Compute Capability: $CUDA_ARCH)"
echo "================================================"

# ---------------------------------------------------------
# [1/5] TRITON & BASE LIBRARIES (Install correct version)
# ---------------------------------------------------------
status_msg "[1/5] Installing Triton..."
run_quiet "Triton Install" pip install -U --no-cache-dir --progress-bar off triton==3.5.0

# ---------------------------------------------------------
# [2/5] FLASH ATTENTION LOGIC
# ---------------------------------------------------------
# Flash Attention 2 supports Ampere (8.0) and newer
status_msg "[2/5] Installing Flash Attention"

if echo "$CUDA_ARCH" | grep -Eq '(^|;)(80|86|89|90|100|120)($|;)'; then

    status_msg "Supported architecture detected ($CUDA_ARCH). Installing Flash Attention..."

    PYTHON_VER=$(python -c 'import sys; print(f"{sys.version_info.major}{sys.version_info.minor}")')
    TORCH_VER=$(python -c 'import torch; print(".".join(torch.__version__.split("+")[0].split(".")[:2]))')
    CUDA_VER="128"
    FLASH_ATTENTION_VER="2.8.3"

    FLASH_ATTN_WHEEL_URL="https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.5.4/flash_attn-${FLASH_ATTENTION_VER}+cu${CUDA_VER}torch${TORCH_VER}-cp${PYTHON_VER}-cp${PYTHON_VER}-linux_x86_64.whl"

    if pip install "$FLASH_ATTN_WHEEL_URL" --no-build-isolation >> "$STARTUP_LOG" 2>&1; then

        touch /tmp/flash_attn_wheel_success
        echo "FlashAttention installed via wheel" >> "$STARTUP_LOG"

    else

        echo "       -> Wheel install failed. Building from source in background..."

        (
            set -e
            cd /tmp
            rm -rf flash-attention

            git clone --depth 1 https://github.com/Dao-AILab/flash-attention.git
            cd flash-attention

            export FLASH_ATTN_CUDA_ARCHS="$CUDA_ARCH"
            export MAX_JOBS=$(nproc)
            export NVCC_THREADS=2

            pip install ninja packaging -q
            python setup.py install

            cd /tmp
            rm -rf flash-attention

        ) > "$NETWORK_VOLUME/logs/flash_attn_install.log" 2>&1 &

        FLASH_ATTN_PID=$!
        echo "$FLASH_ATTN_PID" > /tmp/flash_attn_pid

        echo "       -> Background build started (PID: $FLASH_ATTN_PID)"
        echo "       -> Check $NETWORK_VOLUME/logs/flash_attn_install.log for progress."

    fi

else

    status_msg "Unsupported architecture ($CUDA_ARCH). Skipping Flash Attention."
    echo "       -> Flash Attention requires Ampere (Compute 8.0) or newer."
    echo "       -> Falling back to PyTorch SDPA or xFormers for $DETECTED_GPU."

fi

# ---------------------------------------------------------
# [3/5] SAGE ATTENTION LOGIC
# ---------------------------------------------------------
status_msg "[3/5] Installing SageAttention"

# SageAttention requires Ampere (8.0) or newer, just like Flash Attention
if echo "$CUDA_ARCH" | grep -Eq '(^|;)(80|86|89|90|100|120)($|;)'; then

    status_msg "Supported architecture detected ($CUDA_ARCH). Installing SageAttention..."

    # SageAttention is lightweight to build because it relies on Triton JIT kernels
    # rather than heavy C++ CUDA compilations, so a direct pip install is fast and safe here.
    run_quiet "SageAttention Install" pip install -U --no-cache-dir sageattention

    echo "       -> SageAttention installed successfully."

else

    status_msg "Unsupported architecture ($CUDA_ARCH). Skipping SageAttention."
    echo "       -> SageAttention requires Ampere (Compute 8.0) or newer."

fi

# ============================================================
# [4/5] Setting up workspace
# ============================================================
status_msg "[4/5] Setting up workspace..."

# 1. Sync the RunPod helper repo from /tmp to Volume
if [ -d "/tmp/lora-trainer" ]; then
    # If it already exists on volume, remove old training configs to ensure we use latest from Git
    # but keep the directory structure clean.
    if [ -d "$NETWORK_VOLUME/lora-trainer" ]; then
        rm -rf "$NETWORK_VOLUME/lora-trainer"
    fi
    mv /tmp/lora-trainer "$NETWORK_VOLUME/"

    # Move specific training subfolders to the Volume Root for easier access
    for dir in Captioning wan2.2_musubi_training qwen_image_musubi_training z_image_musubi_training z_image_turbo_musubi_training flux2_musubi_training; do
        if [ -d "$NETWORK_VOLUME/lora-trainer/$dir" ]; then
            rm -rf "$NETWORK_VOLUME/$dir" # Remove old version
            mv "$NETWORK_VOLUME/lora-trainer/$dir" "$NETWORK_VOLUME/"
        fi
    done

    # Move and fix script permissions
    for script in interactive_start_training.sh resume_diffusion_pipe_training.sh rclone_configuration.sh; do
        if [ -f "$NETWORK_VOLUME/lora-trainer/$script" ]; then
            mv "$NETWORK_VOLUME/lora-trainer/$script" "$NETWORK_VOLUME/"
            chmod +x "$NETWORK_VOLUME/$script"
        fi
    done

    # Move utility files
    for utility in resume_dp_training_readme.txt; do
        if [ -f "$NETWORK_VOLUME/lora-trainer/$utility" ]; then
            mv "$NETWORK_VOLUME/lora-trainer/$utility" "$NETWORK_VOLUME/"
        fi
    done

    # Handle the send_lora utility
    if [ -f "$NETWORK_VOLUME/lora-trainer/send_lora.sh" ]; then
        chmod +x "$NETWORK_VOLUME/lora-trainer/send_lora.sh"
        cp "$NETWORK_VOLUME/lora-trainer/send_lora.sh" /usr/local/bin/
    fi
fi

# 2. Handle main diffusion-pipe repository
if [ -d "/diffusion_pipe" ]; then
    if [ ! -d "$NETWORK_VOLUME/diffusion_pipe" ]; then
        status_msg "Deploying diffusion-pipe to volume..."
        mv /diffusion_pipe "$NETWORK_VOLUME/"
    else
        status_msg "diffusion-pipe already on volume. Cleaning ephemeral copy..."
        rm -rf /diffusion_pipe
    fi
fi

# 3. Git Update & Dataset Setup
DIFF_PIPE_DIR="$NETWORK_VOLUME/diffusion_pipe"
if [ -d "$DIFF_PIPE_DIR/.git" ]; then
    (cd "$DIFF_PIPE_DIR" && git pull --ff-only) >> "$STARTUP_LOG" 2>&1 || echo "Warning: git pull failed."
fi

# Sync dataset.toml
if [ -f "$NETWORK_VOLUME/lora-trainer/dataset.toml" ]; then
    mkdir -p "$DIFF_PIPE_DIR/examples"
    mv "$NETWORK_VOLUME/lora-trainer/dataset.toml" "$DIFF_PIPE_DIR/examples/"
fi

# 4. Path Patching (TOML Files)
# We check if the backup exists; if it does, the paths are likely already patched.
TOML_DIR="$NETWORK_VOLUME/lora-trainer/toml_files"

if [ -d "$TOML_DIR" ]; then
    status_msg "Patching TOML configurations..."

    for toml_file in "$TOML_DIR"/*.toml; do
        if [ -f "$toml_file" ]; then
            # Prevention check: only patch if the volume path isn't already there
            if ! grep -q "$NETWORK_VOLUME" "$toml_file"; then

                # 1. Standardize paths to use the Network Volume
                # Use [[:space:]]* to handle 'key=/path', 'key = /path', or 'key  =  /path'
                sed -i "s|[[:space:]]*=[[:space:]]*'/models/| = '$NETWORK_VOLUME/models/|g" "$toml_file"
                sed -i "s|[[:space:]]*=[[:space:]]*'/Wan/| = '$NETWORK_VOLUME/models/Wan/|g" "$toml_file"

                # 2. Redirect output folder
                sed -i "s|[[:space:]]*=[[:space:]]*'/data/| = '$NETWORK_VOLUME/output_folder/|g" "$toml_file"
                sed -i "s|[[:space:]]*=[[:space:]]*'/output_folder/| = '$NETWORK_VOLUME/output_folder/|g" "$toml_file"

                # 3. Handle nested/quoted/commented complex paths
                sed -i "s|{path = '/models/|{path = '$NETWORK_VOLUME/models/|g" "$toml_file"
                sed -i "s|merge_adapters = \['/models/|merge_adapters = ['$NETWORK_VOLUME/models/|g" "$toml_file"
                sed -i "s|#transformer_path = '/models/|#transformer_path = '$NETWORK_VOLUME/models/|g" "$toml_file"

                echo "    ✅ Patched: $(basename "$toml_file")"
            else
                echo "    ℹ️ Already Patched: $(basename "$toml_file")"
            fi
        fi
    done
fi

mkdir -p "$NETWORK_VOLUME/image_dataset_here" "$NETWORK_VOLUME/video_dataset_here"

# Final dataset path correction
if [ -f "$DIFF_PIPE_DIR/examples/dataset.toml" ]; then
    sed -i "s|path = '.*grayscale'|path = '$NETWORK_VOLUME/image_dataset_here'|" "$DIFF_PIPE_DIR/examples/dataset.toml"
fi

# Safely handle Musubi-Tuner move/sync
if [ -d "/musubi-tuner" ]; then
    if [ ! -d "$NETWORK_VOLUME/musubi-tuner" ]; then
        status_msg "First run: Moving Musubi-Tuner to $NETWORK_VOLUME..."
        mv /musubi-tuner "$NETWORK_VOLUME/"
    else
        status_msg "Restart detected: Musubi-Tuner already exists on volume."
        # Clean up the image's internal copy to keep the container thin
        rm -rf /musubi-tuner
    fi

    # ALWAYS run this on restart:
    # This re-registers the version on the Network Volume with the fresh OS
    status_msg "Re-linking Musubi-Tuner to Python environment..."
    run_quiet "Musubi Link" pip install -e "$NETWORK_VOLUME/musubi-tuner" --no-deps
fi

# ============================================================
# [5/5] Starting JupyterLab
# ============================================================
status_msg "[5/5] Starting JupyterLab..."

jupyter-lab --ip=0.0.0.0 --allow-root --no-browser \
    --ServerApp.token='' --ServerApp.password='' \
    --ServerApp.allow_origin='*' --ServerApp.allow_credentials=True \
    --notebook-dir="$NETWORK_VOLUME" >> "$STARTUP_LOG" 2>&1 &

echo ""
echo "================================================"
echo ""
echo "  Template ready!"
echo ""
echo "  To access JupyterLab and TensorBoard from your local machine:"
echo ""
echo "  1) Use the SSH command provided by your host and add port forwarding like this:"
echo ""
echo "     Jupyter:"
echo "     ssh -p <SSH_PORT> hostname@<SERVER_IP> -L 8888:localhost:8888"
echo "     Then open your local browser:"
echo "     http://localhost:8888/lab"
echo ""
echo "     TensorBoard:"
echo "     ssh -p <SSH_PORT> hostname@<SERVER_IP> -L 6006:localhost:6006"
echo "  2) Then open your local browser:"
echo "     http://localhost:6006"
echo ""
echo "  You can also access it via the RunPod web interface if deployed there"
echo ""
echo "================================================"
echo ""

# ================================
# SSH Startup
# ================================

echo "🔐 Starting SSH server..."

mkdir -p /var/run/sshd
chmod 700 /root/.ssh

# If SSH_PUBLIC_KEY provided via env, append safely
if [ -n "${SSH_PUBLIC_KEY:-}" ]; then
    echo "Adding SSH_PUBLIC_KEY from environment..."
    touch /root/.ssh/authorized_keys
    chmod 600 /root/.ssh/authorized_keys

    # Avoid duplicates
    grep -qxF "$SSH_PUBLIC_KEY" /root/.ssh/authorized_keys 2> /dev/null \
        || echo "$SSH_PUBLIC_KEY" >> /root/.ssh/authorized_keys
fi

/usr/sbin/sshd

echo "✅ SSH ready."

status_msg "Initialization complete"

sleep infinity
