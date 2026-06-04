#!/usr/bin/env bash
# --- COLORS & UI ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'
print_header() {
    echo -e "\n${BOLD}${PURPLE}================================================================${NC}"
    echo -e "${BOLD}${CYAN}  $1 ${NC}"
    echo -e "${BOLD}${PURPLE}================================================================${NC}"
}
print_status() { echo -e "${BLUE}[WAIT]${NC} $1"; }
print_success() { echo -e "${GREEN}[OK]  ${NC} $1"; }
print_error() { echo -e "${RED}[FAIL]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARN]${NC} $1"; }
echo -e "${BOLD}${CYAN}WAN 2.2 DUAL-FLOW VIDEO / IMAGE TRAINER${NC}"
echo -e "---------------------------------------"

########################################
# GPU Detection
########################################
print_header "STAGE 1: HARDWARE CHECK"

gpu_count() {
    # 1. First check if Accelerate or the system has explicitly masked visible devices
    if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
        # Count the number of comma-separated IDs (e.g., "0,1" -> 2)
        echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l | awk '{print $1}'
    # 2. Fall back to raw hardware detection if no mask is present
    elif command -v nvidia-smi > /dev/null 2>&1; then
        nvidia-smi -L 2> /dev/null | wc -l | awk '{print $1}'
    else
        echo 0
    fi
}

GPU_COUNT=$(gpu_count)

if [ "${GPU_COUNT}" -lt 1 ]; then
    print_error "No CUDA GPUs detected or allowed in this session. Aborting."
    exit 1
fi

print_success "Detected/Allocated GPUs for this run: ${BOLD}${GPU_COUNT}${NC}"

########################################
# Config, Paths & Task Selection
########################################
print_header "STAGE 2: CONFIGURATION & TASK"
CONFIG_FILE="${CONFIG_FILE:-wan_musubi_config.sh}"
if [ -f "$CONFIG_FILE" ]; then
    source "$CONFIG_FILE"
    print_success "Loaded config: ${BOLD}$CONFIG_FILE${NC}"
else
    print_error "Config file $CONFIG_FILE not found!"
    exit 1
fi

# --- Unified Variable Mapping ---
TITLE_HIGH="${TITLE_HIGH:-wan2.2_lora_high}"
TITLE_LOW="${TITLE_LOW:-wan2.2_lora_low}"
CAPTION_EXT="${CAPTION_EXT:-.txt}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-2}"
NUM_REPEATS="${NUM_REPEATS:-5}"
MAX_TRAIN_EPOCHS="${MAX_TRAIN_EPOCHS:-100}"
SAVE_EVERY_N_EPOCHS="${SAVE_EVERY_N_EPOCHS:-20}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
SEED_HIGH="${SEED_HIGH:-41}"
SEED_LOW="${SEED_LOW:-42}"
OPTIMIZER_TYPE="${OPTIMIZER_TYPE:-adamw8bit}"
LR_SCHEDULER="${LR_SCHEDULER:-cosine}"
TIMESTEP_SAMPLING="${TIMESTEP_SAMPLING:-shift}"
NETWORK_DROPOUT="${NETWORK_DROPOUT:-0}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-1}"
NUM_CPU_THREADS_PER_PROCESS="${NUM_CPU_THREADS_PER_PROCESS:-1}"
MAX_DATA_LOADER_N_WORKERS="${MAX_DATA_LOADER_N_WORKERS:-2}"
DISCRETE_FLOW_SHIFT="${DISCRETE_FLOW_SHIFT:-2.0}"
BUCKET_NO_UPSCALE="$(echo "${BUCKET_NO_UPSCALE:-true}" | tr '[:upper:]' '[:lower:]')"
KEEP_DATASET="${KEEP_DATASET:-0}"
SKIP_CACHE="${SKIP_CACHE:-0}"
# LoRA Specifics
LORA_RANK="${LORA_RANK:-32}"
LORA_ALPHA="${LORA_ALPHA:-32}"
# Video Specifics
TARGET_FRAMES="${TARGET_FRAMES:-1, 40, 80}"
FRAME_EXTRACTION="${FRAME_EXTRACTION:-head}"
# Derived Paths
DATASET_DIR="${DATASET_DIR:-$NETWORK_VOLUME/video_dataset_here}"
REPO_DIR="$NETWORK_VOLUME/musubi-tuner"
WAN_CACHE_DIR="$NETWORK_VOLUME/cache/wan"
MODELS_DIR="$NETWORK_VOLUME/models/wan"
# Weight Variables (T2V & I2V)
WAN_VAE="$MODELS_DIR/Wan2_1_VAE_bf16.safetensors"
WAN_T5="$MODELS_DIR/models_t5_umt5-xxl-enc-bf16.pth"
WAN_DIT_HIGH="$MODELS_DIR/Wan-2.2-T2V-High-Noise-BF16.safetensors"
WAN_DIT_LOW="$MODELS_DIR/Wan-2.2-T2V-Low-Noise-BF16.safetensors"
WAN_DIT_I2V_HIGH="$MODELS_DIR/Wan-2.2-I2V-High-Noise-BF16.safetensors"
WAN_DIT_I2V_LOW="$MODELS_DIR/Wan-2.2-I2V-Low-Noise-BF16.safetensors"

export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
export PYTORCH_ALLOC_CONF="expandable_segments:True"

# --- TASK SELECTION (T2V vs I2V) ---
echo -e "\n${CYAN}Select Base Model / Task Type:${NC}"
echo "1) Text-to-Video (t2v-A14B)"
echo "2) Image-to-Video (i2v-A14B)"
read -rp "Selection (1/2, default 1): " TASK_CHOICE
TASK_CHOICE="${TASK_CHOICE:-1}"
if [ "$TASK_CHOICE" = "2" ]; then
    WAN_TASK="i2v-A14B"
    ACTIVE_DIT_HIGH="$WAN_DIT_I2V_HIGH"
    ACTIVE_DIT_LOW="$WAN_DIT_I2V_LOW"
    print_status "Task set to: ${BOLD}Image-to-Video (I2V)${NC}"
else
    WAN_TASK="t2v-A14B"
    ACTIVE_DIT_HIGH="$WAN_DIT_HIGH"
    ACTIVE_DIT_LOW="$WAN_DIT_LOW"
    print_status "Task set to: ${BOLD}Text-to-Video (T2V)${NC}"
fi

# After TASK_CHOICE is resolved, set the correct boundary:
if [ "$WAN_TASK" = "i2v-A14B" ]; then
    TS_BOUNDARY=900
else
    TS_BOUNDARY=875
fi

# --- SINGLE GPU: WEIGHT SELECTION ---
# On dual GPU both weights are always downloaded and trained in parallel.
# On single GPU, ask which DiT to download and train (default: LOW).
if [ "${GPU_COUNT}" -lt 2 ]; then
    echo -e "\n${CYAN}Select DiT weight to download and train:${NC}"
    echo "1) HIGH-Noise DiT"
    echo "2) LOW-Noise DiT"
    read -rp "Selection (1/2, default 2): " WEIGHT_CHOICE
    WEIGHT_CHOICE="${WEIGHT_CHOICE:-2}"
    if [ "$WEIGHT_CHOICE" = "1" ]; then
        SINGLE_DIT_PATH="$ACTIVE_DIT_HIGH"
        SINGLE_TS_MIN="$TS_BOUNDARY"
        SINGLE_TS_MAX="1000"
        SINGLE_NAME="$TITLE_HIGH"
        SINGLE_OUT_SUBDIR="HIGH"
        print_status "Single GPU weight: ${BOLD}HIGH-Noise DiT${NC}"
    else
        SINGLE_DIT_PATH="$ACTIVE_DIT_LOW"
        SINGLE_TS_MIN="0"
        SINGLE_TS_MAX="$TS_BOUNDARY"
        SINGLE_NAME="$TITLE_LOW"
        SINGLE_OUT_SUBDIR="LOW"
        print_status "Single GPU weight: ${BOLD}LOW-Noise DiT${NC}"
    fi
fi

# Dynamic output paths
OUT_HIGH="$NETWORK_VOLUME/output_folder_musubi/wan2.2/$WAN_TASK/$TITLE_HIGH"
OUT_LOW="$NETWORK_VOLUME/output_folder_musubi/wan2.2/$WAN_TASK/$TITLE_LOW"

# Resolve single-GPU output dir now that OUT_HIGH/OUT_LOW are set
if [ "${GPU_COUNT}" -lt 2 ]; then
    if [ "$SINGLE_OUT_SUBDIR" = "HIGH" ]; then
        SINGLE_OUT="$OUT_HIGH"
        SINGLE_SEED="$SEED_HIGH"
    else
        SINGLE_OUT="$OUT_LOW"
        SINGLE_SEED="$SEED_LOW"
    fi
fi

# Remove sub directories for the video dataset
find "$NETWORK_VOLUME/video_dataset_here" -mindepth 1 -maxdepth 1 -type d -exec rm -rf {} +

# --- DATASET AUTO-DETECTION ---
shopt -s nocasematch
if [[ "$DATASET_DIR" == *"image"* ]]; then
    DATASET_TYPE="image"
    print_status "Dataset Type: ${BOLD}IMAGE${NC} (Dual-flow enabled)"
else
    DATASET_TYPE="video"
    print_status "Dataset Type: ${BOLD}VIDEO${NC} (Dual-flow enabled)"
fi
shopt -u nocasematch

mkdir -p "$DATASET_DIR" "$OUT_HIGH" "$OUT_LOW" "$MODELS_DIR" "$WAN_CACHE_DIR"
print_status "Created HIGH and LOW output directories."

########################################
# Total steps calculation
########################################
if [ "$DATASET_TYPE" = "video" ]; then
    IMG_COUNT=$(find "$DATASET_DIR" -maxdepth 1 -type f \( -iname "*.mp4" -o -iname "*.mkv" -o -iname "*.mov" \) | wc -l)
else
    IMG_COUNT=$(find "$DATASET_DIR" -maxdepth 1 -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.webp" \) | wc -l)
fi
if [ "$IMG_COUNT" -le 0 ]; then
    print_error "No media files found in $DATASET_DIR! Check your path or extensions."
    exit 1
fi
EFFECTIVE_BATCH=$((BATCH_SIZE * GRAD_ACCUM_STEPS))
if [ "$EFFECTIVE_BATCH" -eq 0 ]; then
    print_error "Effective batch size is 0. Check BATCH_SIZE and GRAD_ACCUM_STEPS."
    exit 1
fi
SAMPLES_PER_EPOCH=$((IMG_COUNT * NUM_REPEATS))
STEPS_PER_EPOCH=$(((SAMPLES_PER_EPOCH + EFFECTIVE_BATCH - 1) / EFFECTIVE_BATCH))
TOTAL_STEPS=$((STEPS_PER_EPOCH * MAX_TRAIN_EPOCHS))
if [ "$TOTAL_STEPS" -le 0 ]; then
    print_error "TOTAL_STEPS calculated as 0. Check your config."
    exit 1
fi

########################################
# Normalize CSV Helper
########################################
normalize_numeric_csv() {
    local s="$1"
    s="$(echo "$s" | tr -d '[]"')"
    s="$(echo "$s" | sed -E 's/[[:space:]]*,[[:space:]]*/, /g; s/^[[:space:]]+|[[:space:]]+$//g')"
    echo "$s"
}
RESOLUTION_LIST_NORM="$(normalize_numeric_csv "${RESOLUTION_LIST:-"720, 1280"}")"
TARGET_FRAMES_NORM="$(normalize_numeric_csv "${TARGET_FRAMES:-"1, 40, 80"}")"

########################################
# Weights Management (Wan 2.2)
########################################
print_header "STAGE 3: MODEL WEIGHTS (WAN 2.2)"
HF_DL="hf download"
HF_FLAGS="--local-dir $MODELS_DIR"
find "$MODELS_DIR/.cache/huggingface" -name "*.lock" -type f -delete 2> /dev/null || true

########################################
# Retry Download Function
########################################
retry_file_download() {
    local repo="$1"
    local remote_file="$2"
    local expected_path="$3"
    local max_retries=5
    local attempt=1
    local delay=5
    while [[ $attempt -le $max_retries ]]; do
        echo "[INFO] Attempt $attempt → Fetching $(basename "$remote_file")..."
        $HF_DL "$repo" "$remote_file" $HF_FLAGS
        local actual_download_path="$MODELS_DIR/$remote_file"
        if [[ -f "$actual_download_path" ]]; then
            print_status "Moving $(basename "$remote_file") to root models directory..."
            mv "$actual_download_path" "$expected_path"
        fi
        if [[ -f "$expected_path" && -s "$expected_path" ]]; then
            print_success "Verified: $(basename "$expected_path")"
            return 0
        fi
        print_warning "Download failed or path mismatch. Retrying in ${delay}s..."
        sleep $delay
        ((attempt++))
        delay=$((delay * 2))
    done
    print_error "Failed to download $(basename "$remote_file") after $max_retries attempts"
    return 1
}

download_if_missing() {
    local repo="$1"
    local target_path="$2"
    local remote_file="$3"
    if [[ ! -f "$target_path" ]]; then
        print_status "Missing: $(basename "$target_path")"
        retry_file_download "$repo" "$remote_file" "$target_path" || exit 1
    else
        print_success "Found: $(basename "$target_path")"
    fi
}

########################################
# 1. Base Shared Weights (always needed)
########################################
download_if_missing \
    "MonsterMMORPG/Wan_GGUF" \
    "$WAN_T5" \
    "models_t5_umt5-xxl-enc-bf16.pth"
download_if_missing \
    "MonsterMMORPG/Wan_GGUF" \
    "$WAN_VAE" \
    "Wan2_1_VAE_bf16.safetensors"

########################################
# 2. Task-Specific DiT Downloads
#    Always downloads both HIGH + LOW
########################################
if [ "$WAN_TASK" = "t2v-A14B" ]; then
    download_if_missing "MonsterMMORPG/Wan_GGUF" "$WAN_DIT_HIGH" "Wan-2.2-T2V-High-Noise-BF16.safetensors"
    download_if_missing "MonsterMMORPG/Wan_GGUF" "$WAN_DIT_LOW" "Wan-2.2-T2V-Low-Noise-BF16.safetensors"
elif [ "$WAN_TASK" = "i2v-A14B" ]; then
    download_if_missing "MonsterMMORPG/Wan_GGUF" "$WAN_DIT_I2V_HIGH" "Wan-2.2-I2V-High-Noise-BF16.safetensors"
    download_if_missing "MonsterMMORPG/Wan_GGUF" "$WAN_DIT_I2V_LOW" "Wan-2.2-I2V-Low-Noise-BF16.safetensors"
fi

########################################
# Final Validation
########################################
MISSING_WEIGHTS=false

# Check shared weights
if [[ ! -f "$WAN_T5" || ! -f "$WAN_VAE" ]]; then
    MISSING_WEIGHTS=true
fi

# Check task-specific dual weights
if [ "$WAN_TASK" = "t2v-A14B" ]; then
    [[ ! -f "$WAN_DIT_HIGH" || ! -f "$WAN_DIT_LOW" ]] && MISSING_WEIGHTS=true
elif [ "$WAN_TASK" = "i2v-A14B" ]; then
    [[ ! -f "$WAN_DIT_I2V_HIGH" || ! -f "$WAN_DIT_I2V_LOW" ]] && MISSING_WEIGHTS=true
fi

if [ "$MISSING_WEIGHTS" = true ]; then
    print_error "Weight validation failed for task: $WAN_TASK."
    echo "[DEBUG] Current contents of $MODELS_DIR:"
    find "$MODELS_DIR" -maxdepth 3
    exit 1
fi
print_success "Wan 2.2 weights ready."

########################################
# Dataset Setup
########################################
print_header "STAGE 4: DATASET PREP"

for ACTIVE_OUT in "$OUT_HIGH" "$OUT_LOW"; do
    DATASET_TOML="$ACTIVE_OUT/dataset.toml"

    if [ "${KEEP_DATASET:-0}" = "1" ] && [ -f "$DATASET_TOML" ]; then
        print_status "Keeping existing dataset.toml in $(basename "$ACTIVE_OUT")"
    else
        print_status "Writing dataset.toml for $(basename "$ACTIVE_OUT") (Type: $DATASET_TYPE)"

        # 1. Global Settings Block (Ending with a clean empty line buffer)
        cat > "$DATASET_TOML" << TOML
[general]
resolution = [${RESOLUTION_LIST_NORM}]
caption_extension = "${CAPTION_EXT:-.txt}"
batch_size = ${BATCH_SIZE:-1}
enable_bucket = true
bucket_no_upscale = ${BUCKET_NO_UPSCALE:-true}

TOML

        # 2. Append the Dataset Segment cleanly based on Data Modality
        if [ "$DATASET_TYPE" = "video" ]; then
            cat >> "$DATASET_TOML" << TOML
[[datasets]]
video_directory = "$DATASET_DIR"
cache_directory = "${WAN_CACHE_DIR}"
num_repeats = ${NUM_REPEATS:-1}
target_frames = [${TARGET_FRAMES_NORM}]
frame_extraction = "${FRAME_EXTRACTION:-full}"
TOML
            # Safely inject optional strategy-dependent keys
            case "$FRAME_EXTRACTION" in
                "slide") echo "frame_stride = ${FRAME_STRIDE:-1}" >> "$DATASET_TOML" ;;
                "uniform") echo "frame_sample = ${FRAME_SAMPLE:-4}" >> "$DATASET_TOML" ;;
                "head") ;; # no extra key needed
                *) echo "max_frames = ${MAX_FRAMES:-100}" >> "$DATASET_TOML" ;;
            esac
        else
            # Explicitly maps image datasets while honoring your loops/repeats
            cat >> "$DATASET_TOML" << TOML
[[datasets]]
image_directory = "${DATASET_DIR}"
cache_directory = "${WAN_CACHE_DIR}"
num_repeats = ${NUM_REPEATS:-1}
TOML
        fi

        print_success "dataset.toml cleanly created in $(basename "$ACTIVE_OUT")."
    fi
done

########################################
# Caching
########################################
print_header "STAGE 5: PRE-CACHING"
if [ "$SKIP_CACHE" = "1" ]; then
    print_warning "Skipping caching."
else
    print_status "Caching Latents (VAE)..."
    I2V_FLAG=""
    [ "$WAN_TASK" = "i2v-A14B" ] && I2V_FLAG="--i2v"
    python3 "$REPO_DIR/wan_cache_latents.py" --dataset_config "$OUT_HIGH/dataset.toml" --vae "$WAN_VAE" $I2V_FLAG
    print_status "Caching Text (T5)..."
    python3 "$REPO_DIR/wan_cache_text_encoder_outputs.py" --dataset_config "$OUT_HIGH/dataset.toml" --t5 "$WAN_T5" --batch_size 4
fi

########################################
# Dynamic Save Frequency
########################################
DYNAMIC_SAVE_STEPS=$STEPS_PER_EPOCH
if [ "$DYNAMIC_SAVE_STEPS" -lt 20 ]; then
    DYNAMIC_SAVE_STEPS=20
fi
if [ "${USE_EMA:-0}" = "1" ]; then
    print_success "Save Frequency: Every $DYNAMIC_SAVE_STEPS steps."
fi

########################################
# Training Launch
########################################
print_header "STAGE 6: TRAINING LAUNCH"
TENSORBOARD_FOLDER="$NETWORK_VOLUME/output_folder_musubi"
print_status "TensorBoard logs for this run are located at:\n$TENSORBOARD_FOLDER\n"
echo -e "\n${BOLD}${YELLOW}View progress at:${NC} http://localhost:6006"
echo -e ""
echo -e "------------------------------------"
echo -e "${CYAN}Output (High):${NC}         $TITLE_HIGH"
echo -e "${CYAN}Output (Low):${NC}          $TITLE_LOW"
echo -e "${CYAN}Detected Type:${NC}         ${BOLD}$DATASET_TYPE${NC} ($IMG_COUNT files)"
echo -e "${CYAN}Task Mode:${NC}             $WAN_TASK"
echo -e "${CYAN}Frames:${NC}                $TARGET_FRAMES"
echo -e "------------------------------------"
echo -e "${CYAN}Rank / Alpha:${NC}          $LORA_RANK / $LORA_ALPHA"
echo -e "${CYAN}Timestep sampling:${NC}     $TIMESTEP_SAMPLING"
echo -e "${CYAN}Flow shift:${NC}            $DISCRETE_FLOW_SHIFT"
echo -e "${CYAN}Optimizer:${NC}             $OPTIMIZER_TYPE (LR: $LEARNING_RATE)"
echo -e "${CYAN}Scheduler:${NC}             $LR_SCHEDULER"
echo -e "${CYAN}Attention:${NC}             $ATTN"
echo -e "${CYAN}Network dropout:${NC}       $NETWORK_DROPOUT"
if [ -n "$BLOCKS_TO_SWAP" ]; then
    echo -e "${YELLOW}Blocks to Swap:${NC}        ${BOLD}$BLOCKS_TO_SWAP (CPU Offloading Active)${NC}"
fi
echo -e "${CYAN}Grad Accum:${NC}            $GRAD_ACCUM_STEPS (Effective Batch: $EFFECTIVE_BATCH)"
echo -e "${CYAN}Estimated Steps:${NC}       $TOTAL_STEPS"
echo -e "------------------------------------"
sleep 5

########################################
# DYNAMIC SCHEDULER & WARMUP
########################################
LR_WARMUP_STEPS=0
LR_SCHEDULER_POWER=1.0
if [ "$LR_SCHEDULER" == "constant" ]; then
    LR_WARMUP_STEPS=0
elif [ "$OPTIMIZER_TYPE" == "prodigyopt.Prodigy" ]; then
    if [ "$TOTAL_STEPS" -lt 400 ]; then
        LR_WARMUP_STEPS=30
    elif [ "$TOTAL_STEPS" -lt 1500 ]; then
        LR_WARMUP_STEPS=$((TOTAL_STEPS * 10 / 100))
    else
        LR_WARMUP_STEPS=$((TOTAL_STEPS * 5 / 100))
    fi
elif [ "$OPTIMIZER_TYPE" == "adamw" ] || [ "$OPTIMIZER_TYPE" == "adamw8bit" ] || [ "$OPTIMIZER_TYPE" == "adafactor" ]; then
    LR_WARMUP_STEPS=$((TOTAL_STEPS * 5 / 100))
fi
if [ "$LR_SCHEDULER" != "constant" ]; then
    MIN_WARMUP=$(((TOTAL_STEPS * 5 + 99) / 100))
    [ "$MIN_WARMUP" -lt 20 ] && MIN_WARMUP=20
    MAX_WARMUP=$(((TOTAL_STEPS * 12 + 99) / 100))
    [ "$LR_WARMUP_STEPS" -lt "$MIN_WARMUP" ] && LR_WARMUP_STEPS=$MIN_WARMUP
    [ "$LR_WARMUP_STEPS" -gt "$MAX_WARMUP" ] && LR_WARMUP_STEPS=$MAX_WARMUP
fi
print_success "LR Scheduler: ${BOLD}$LR_SCHEDULER${NC}"
print_success "Warmup Steps: ${BOLD}$LR_WARMUP_STEPS${NC}"

STATE_FILE="$REPO_DIR/training_state.tmp"
cat << EOF > "$STATE_FILE"
LR_SCHEDULER_POWER="$LR_SCHEDULER_POWER"
DYNAMIC_SAVE_STEPS="$DYNAMIC_SAVE_STEPS"
EOF
print_success "Training state exported to $STATE_FILE"

COMMON_FLAGS=(
    --task "$WAN_TASK"
    --vae "$WAN_VAE"
    --t5 "$WAN_T5"
    --optimizer_type "$OPTIMIZER_TYPE"
    --lr_warmup_steps "$LR_WARMUP_STEPS"
    --lr_scheduler "$LR_SCHEDULER"
    --lr_scheduler_power "$LR_SCHEDULER_POWER"
    --learning_rate "$LEARNING_RATE"
    --gradient_accumulation_steps "$GRAD_ACCUM_STEPS"
    --max_data_loader_n_workers "$MAX_DATA_LOADER_N_WORKERS"
    --persistent_data_loader_workers
    --network_module networks.lora_wan
    --network_dim "$LORA_RANK"
    --network_alpha "$LORA_ALPHA"
    --timestep_sampling "$TIMESTEP_SAMPLING"
    --weighting_scheme none
    --discrete_flow_shift "$DISCRETE_FLOW_SHIFT"
    --network_dropout "$NETWORK_DROPOUT"
    --save_state
    --max_train_epochs "$MAX_TRAIN_EPOCHS"
    --save_every_n_epochs "$SAVE_EVERY_N_EPOCHS"
)
if [ "$OPTIMIZER_TYPE" == "adafactor" ]; then COMMON_FLAGS+=("--max_grad_norm" "0"); fi
if [ "${FP8_BASE:-0}" = "1" ]; then COMMON_FLAGS+=("--fp8_base"); fi
if [ "${FP8_SCALED:-0}" = "1" ]; then COMMON_FLAGS+=("--fp8_scaled"); fi
if [ "${FP8_T5:-0}" = "1" ]; then COMMON_FLAGS+=("--fp8_t5"); fi
if [ "${USE_EMA:-0}" = "1" ]; then COMMON_FLAGS+=("--save_every_n_steps" "$DYNAMIC_SAVE_STEPS"); fi
if [ -n "$BLOCKS_TO_SWAP" ]; then COMMON_FLAGS+=("--blocks_to_swap" "$BLOCKS_TO_SWAP"); fi
if [ "${GRADIENT_CHECKPOINTING:-1}" = "1" ]; then COMMON_FLAGS+=("--gradient_checkpointing"); fi
if [ "${ATTN:-flash}" = "flash" ]; then
    COMMON_FLAGS+=(--flash_attn --mixed_precision bf16)
elif [ "$ATTN" = "sdpa" ]; then
    COMMON_FLAGS+=(--sdpa --mixed_precision bf16)
fi
if [ ${#OPTIMIZER_ARGS[@]} -gt 0 ]; then
    COMMON_FLAGS+=("--optimizer_args" "${OPTIMIZER_ARGS[@]}")
fi

# --- EXECUTION ---
if [ "${GPU_COUNT}" -ge 2 ]; then
    print_success "Multi-GPU Training! Running parallel HIGH/LOW noise flows."

    # 1. Launch HIGH-Noise training on GPU 0 explicitly forcing a single local process
    env CUDA_VISIBLE_DEVICES=0 accelerate launch \
        --num_processes 1 \
        --num_cpu_threads_per_process "$NUM_CPU_THREADS_PER_PROCESS" \
        --main_process_port 29500 --mixed_precision bf16 \
        "$REPO_DIR/wan_train_network.py" --dit "$ACTIVE_DIT_HIGH" \
        --preserve_distribution_shape \
        --min_timestep "$TS_BOUNDARY" --max_timestep 1000 --seed "$SEED_HIGH" \
        --output_dir "$OUT_HIGH" --output_name "$TITLE_HIGH" \
        --logging_dir "$OUT_HIGH/logs" \
        --dataset_config "$OUT_HIGH/dataset.toml" \
        --log_with tensorboard "${COMMON_FLAGS[@]}" &

    # Give the first background process a 3-second head start to create its tracking files
    # and prevent NCCL or cache lockups with the second instance
    sleep 3

    # 2. Launch LOW-Noise training on GPU 1 explicitly forcing a single local process
    env CUDA_VISIBLE_DEVICES=1 accelerate launch \
        --num_processes 1 \
        --num_cpu_threads_per_process "$NUM_CPU_THREADS_PER_PROCESS" \
        --main_process_port 29501 --mixed_precision bf16 \
        "$REPO_DIR/wan_train_network.py" --dit "$ACTIVE_DIT_LOW" \
        --preserve_distribution_shape \
        --min_timestep 0 --max_timestep "$TS_BOUNDARY" --seed "$SEED_LOW" \
        --output_dir "$OUT_LOW" --output_name "$TITLE_LOW" \
        --logging_dir "$OUT_LOW/logs" \
        --dataset_config "$OUT_LOW/dataset.toml" \
        --log_with tensorboard "${COMMON_FLAGS[@]}" &

    # Wait for both background processes to finish training safely
    wait
    print_success "Dual-GPU Training Complete."
else
    # Single GPU: use the weight selected at the top of Stage 2
    print_success "Single GPU Training: ${BOLD}$SINGLE_NAME${NC} ($([ "$WEIGHT_CHOICE" = "1" ] && echo "HIGH-Noise" || echo "LOW-Noise"))"
    accelerate launch \
        --num_processes 1 \
        --num_cpu_threads_per_process "$NUM_CPU_THREADS_PER_PROCESS" \
        --mixed_precision bf16 \
        "$REPO_DIR/wan_train_network.py" --dit "$SINGLE_DIT_PATH" \
        --preserve_distribution_shape \
        --min_timestep "$SINGLE_TS_MIN" --max_timestep "$SINGLE_TS_MAX" \
        --seed "$SINGLE_SEED" \
        --output_dir "$SINGLE_OUT" --output_name "$SINGLE_NAME" \
        --logging_dir "$SINGLE_OUT/logs" \
        --dataset_config "$SINGLE_OUT/dataset.toml" \
        --log_with tensorboard "${COMMON_FLAGS[@]}"
fi

########################################
# Auto-Convert (Full Batch Mode)
########################################
print_header "STAGE 7: POST-PROCESSING"
CONVERT_SCRIPT="$REPO_DIR/convert_lora.py"
if [ -f "$CONVERT_SCRIPT" ]; then
    DIRS_TO_SCAN=("$OUT_HIGH" "$OUT_LOW")
    CONVERT_COUNT=0
    for TARGET_DIR in "${DIRS_TO_SCAN[@]}"; do
        if [ ! -d "$TARGET_DIR" ]; then continue; fi
        print_status "Scanning $TARGET_DIR for LoRAs to convert..."
        shopt -s nullglob
        for lora in "$TARGET_DIR"/*.safetensors; do
            [[ "$lora" == *"_comfy.safetensors" ]] && continue
            [[ "$lora" == *"model_states"* ]] && continue
            [[ "$lora" == *"-step"* ]] && continue
            COMFY_LORA_PATH="${lora%.safetensors}_comfy.safetensors"
            print_status "Converting $(basename "$lora")...."
            if python3 "$CONVERT_SCRIPT" --input "$lora" --output "$COMFY_LORA_PATH" --target other > /dev/null 2>&1; then
                if python3 -c "from safetensors import safe_open; f = safe_open('$COMFY_LORA_PATH', framework='pt'); f.metadata(); f.keys()" > /dev/null 2>&1; then
                    print_success "Verified: $(basename "$COMFY_LORA_PATH")"
                    ((CONVERT_COUNT++))
                else
                    print_error "CORRUPT: $(basename "$COMFY_LORA_PATH") verification failed."
                    rm -f "$COMFY_LORA_PATH"
                fi
            else
                print_error "FAILED: Conversion error on $(basename "$lora")"
            fi
        done
        shopt -u nullglob
    done
    if [ "$CONVERT_COUNT" -eq 0 ]; then
        print_warning "No new LoRA files found to convert."
    else
        print_success "Batch conversion complete. Total converted: $CONVERT_COUNT"
    fi
else
    print_error "Conversion script not found at $CONVERT_SCRIPT"
fi

print_header "ALL TASKS COMPLETE"
