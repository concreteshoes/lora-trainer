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

# Specific Wan 2.2 UI Colors
C_HIGH='\033[38;5;40m' # Greenish for High Noise
C_LOW='\033[38;5;214m' # Orange for Low Noise

print_header() {
    echo -e "\n${BOLD}${PURPLE}================================================================${NC}"
    echo -e "${BOLD}${CYAN}  $1 ${NC}"
    echo -e "${BOLD}${PURPLE}================================================================${NC}"
}

print_status() { echo -e "${BLUE}[WAIT]${NC} $1"; }
print_success() { echo -e "${GREEN}[OK]  ${NC} $1"; }
print_error() { echo -e "${RED}[FAIL]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARN]${NC} $1"; }

clear
echo -e "${BOLD}${CYAN}WAN 2.2 DUAL-FLOW VIDEO / IMAGE TRAINER${NC}"
echo -e "---------------------------------------"

########################################
# GPU & Blackwell Detection
########################################
print_header "STAGE 1: HARDWARE CHECK"

gpu_count() {
    if command -v nvidia-smi > /dev/null 2>&1; then
        nvidia-smi -L 2> /dev/null | wc -l | awk '{print $1}'
    else
        echo 0
    fi
}

GPU_COUNT=$(gpu_count)
if [ "${GPU_COUNT}" -lt 1 ]; then
    print_error "No CUDA GPUs detected. Aborting."
    exit 1
fi
print_success "Detected GPUs: ${BOLD}${GPU_COUNT}${NC}"

# Blackwell Check
if [ -f /tmp/gpu_arch_type ] && [ "$(cat /tmp/gpu_arch_type)" = "blackwell" ]; then
    echo -e "${RED}${BOLD}!!! WARNING: BLACKWELL GPU DETECTED (B100/B200/5090) !!!${NC}"
    print_warning "Compatibility may be limited. H100/H200 recommended."
    for i in {5..1}; do
        echo -n "$i.."
        sleep 1
    done
    echo ""
fi

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

# --- Unified Variable Mapping (The "Bridge") ---
TITLE_HIGH="${TITLE_HIGH:-wan2.2_lora_high}"
TITLE_LOW="${TITLE_LOW:-wan2.2_lora_low}"
CAPTION_EXT="${CAPTION_EXT:-.txt}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-2}"
NUM_REPEATS="${NUM_REPEATS:-1}"
MAX_TRAIN_EPOCHS="${MAX_TRAIN_EPOCHS:-100}"
SAVE_EVERY_N_EPOCHS="${SAVE_EVERY_N_EPOCHS:-20}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
SEED_HIGH="${SEED_HIGH:-41}"
SEED_LOW="${SEED_LOW:-42}"
OPTIMIZER_TYPE="${OPTIMIZER_TYPE:-adamw8bit}"
LR_SCHEDULER="${LR_SCHEDULER:-cosine}"
TIMESTEP_SAMPLING="${TIMESTEP_SAMPLING:-shift}"
NETWORK_DROPOUT="${NETWORK_DROPOUT:-0.01}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-1}"
NUM_CPU_THREADS_PER_PROCESS="${NUM_CPU_THREADS_PER_PROCESS:-1}"
MAX_DATA_LOADER_N_WORKERS="${MAX_DATA_LOADER_N_WORKERS:-2}"
DISCRETE_FLOW_SHIFT="${DISCRETE_FLOW_SHIFT:-2.0}"
BUCKET_NO_UPSCALE="$(echo "${BUCKET_NO_UPSCALE:-true}" | tr '[:upper:]' '[:lower:]')"
KEEP_DATASET="${KEEP_DATASET:-0}"
SKIP_CACHE="${SKIP_CACHE:-0}"
FP8_BASE="${FP8_BASE:-0}"
FP8_T5="${FP8_T5:-0}"

# LoRA Specifics
LORA_RANK="${LORA_RANK:-64}"
LORA_ALPHA="${LORA_ALPHA:-32}"

# Video Specifics
TARGET_FRAMES="${TARGET_FRAMES:-1, 57, 117}"
FRAME_EXTRACTION="${FRAME_EXTRACTION:-head}"

# Derived Paths
OUT_HIGH="${OUT_HIGH:-$NETWORK_VOLUME/output_folder_musubi/wan22/$TITLE_HIGH}"
OUT_LOW="${OUT_LOW:-$NETWORK_VOLUME/output_folder_musubi/wan22/$TITLE_LOW}"
DATASET_DIR="${DATASET_DIR:-$NETWORK_VOLUME/video_dataset_here}"
REPO_DIR="$NETWORK_VOLUME/musubi-tuner"
WAN_CACHE_DIR="$NETWORK_VOLUME/cache/wan"
MODELS_DIR="$NETWORK_VOLUME/models/Wan"

# Weight Variables (T2V & I2V)
WAN_VAE="$MODELS_DIR/wan_2.1_vae.safetensors"
WAN_T5="$MODELS_DIR/models_t5_umt5-xxl-enc-bf16.pth"
WAN_DIT_HIGH="$MODELS_DIR/wan2.2_t2v_high_noise_14B_fp16.safetensors"
WAN_DIT_LOW="$MODELS_DIR/wan2.2_t2v_low_noise_14B_fp16.safetensors"
WAN_DIT_I2V_HIGH="$MODELS_DIR/wan2.2_i2v_high_noise_14B_fp16.safetensors"
WAN_DIT_I2V_LOW="$MODELS_DIR/wan2.2_i2v_low_noise_14B_fp16.safetensors"

export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
export PYTORCH_ALLOC_CONF=expandable_segments:True

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

# --- DATASET AUTO-DETECTION ---
shopt -s nocasematch
if [[ "$DATASET_DIR" == *"image"* ]]; then
    TRAIN_MODE="IMAGE"
    DATASET_TYPE="image"
    print_status "Dataset Type: ${BOLD}IMAGE${NC} (Will enforce HIGH-noise only)"
else
    TRAIN_MODE="VIDEO"
    DATASET_TYPE="video"
    print_status "Dataset Type: ${BOLD}VIDEO${NC} (Dual-flow enabled)"
fi
shopt -u nocasematch

mkdir -p "$DATASET_DIR" "$OUT_HIGH" "$MODELS_DIR" "$WAN_CACHE_DIR"

if [ "$TRAIN_MODE" == "VIDEO" ]; then
    mkdir -p "$OUT_LOW"
    print_status "Video mode: Created HIGH and LOW output directories."
else
    print_status "Image mode: Created HIGH output directory only."
fi

########################################
# Total steps calculation
########################################
# --- MEDIA COUNTING ---
if [ "$DATASET_TYPE" = "video" ]; then
    # Catches .mp4, .MP4, .mkv, etc if you add them
    IMG_COUNT=$(find "$DATASET_DIR" -maxdepth 1 -type f \( -iname "*.mp4" -o -iname "*.mkv" -o -iname "*.mov" \) | wc -l)
else
    # -iname is standard and case-insensitive
    IMG_COUNT=$(find "$DATASET_DIR" -maxdepth 1 -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.webp" \) | wc -l)
fi

if [ "$IMG_COUNT" -le 0 ]; then
    print_error "No media files found in $DATASET_DIR! Check your path or extensions."
    exit 1
fi

# Calculate Effective Batch
EFFECTIVE_BATCH=$((BATCH_SIZE * GRAD_ACCUM_STEPS))
if [ "$EFFECTIVE_BATCH" -eq 0 ]; then
    print_error "Effective batch size is 0. Check BATCH_SIZE and GRAD_ACCUM_STEPS."
    exit 1
fi

# 1. Total samples seen per epoch
SAMPLES_PER_EPOCH=$((IMG_COUNT * NUM_REPEATS))

# 2. Steps per epoch (using ceiling math to match accelerate/Musubi padding)
STEPS_PER_EPOCH=$(((SAMPLES_PER_EPOCH + EFFECTIVE_BATCH - 1) / EFFECTIVE_BATCH))

# 3. Final total steps
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

RESOLUTION_LIST_NORM="$(normalize_numeric_csv "${RESOLUTION_LIST:-"1024, 1024"}")"
TARGET_FRAMES_NORM="$(normalize_numeric_csv "${TARGET_FRAMES:-"1, 57, 117"}")"

########################################
# Weights Management (Wan 2.2)
########################################
print_header "STAGE 3: MODEL WEIGHTS (WAN 2.2)"

HF_DL="hf download"
HF_FLAGS="--local-dir $MODELS_DIR"

# Optional: clear stale locks
find "$MODELS_DIR/.cache/huggingface" -name "*.lock" -type f -delete 2> /dev/null || true

########################################
# Retry Download Function (File आधारित)
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

        # --- VALIDATION ---
        if [[ -f "$expected_path" && -s "$expected_path" ]]; then
            echo "[OK] Verified: $(basename "$expected_path")"
            return 0
        fi

        echo "[WARN] Download failed or incomplete. Retrying in ${delay}s..."
        sleep $delay

        ((attempt++))
        delay=$((delay * 2))
    done

    print_error "Failed to download $(basename "$remote_file") after $max_retries attempts"
    return 1
}

########################################
# Expected Paths
########################################
# Base
# (these should already point to $MODELS_DIR/... in your env)
# WAN_T5
# WAN_VAE

# T2V
# WAN_DIT_HIGH
# WAN_DIT_LOW

# I2V
# WAN_DIT_I2V_HIGH
# WAN_DIT_I2V_LOW

########################################
# Download Wrapper
########################################
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
# 1. Base Shared Weights
########################################
download_if_missing \
    "Wan-AI/Wan2.1-I2V-14B-720P" \
    "$WAN_T5" \
    "models_t5_umt5-xxl-enc-bf16.pth"

download_if_missing \
    "Comfy-Org/Wan_2.1_ComfyUI_repackaged" \
    "$WAN_VAE" \
    "split_files/vae/wan_2.1_vae.safetensors"

########################################
# 2. T2V (Text-to-Video)
########################################
download_if_missing \
    "Comfy-Org/Wan_2.2_ComfyUI_Repackaged" \
    "$WAN_DIT_HIGH" \
    "split_files/diffusion_models/wan2.2_t2v_high_noise_14B_fp16.safetensors"

download_if_missing \
    "Comfy-Org/Wan_2.2_ComfyUI_Repackaged" \
    "$WAN_DIT_LOW" \
    "split_files/diffusion_models/wan2.2_t2v_low_noise_14B_fp16.safetensors"

########################################
# 3. I2V (Image-to-Video)
########################################
download_if_missing \
    "Comfy-Org/Wan_2.2_ComfyUI_Repackaged" \
    "$WAN_DIT_I2V_HIGH" \
    "split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp16.safetensors"

download_if_missing \
    "Comfy-Org/Wan_2.2_ComfyUI_Repackaged" \
    "$WAN_DIT_I2V_LOW" \
    "split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp16.safetensors"

########################################
# 4. Flatten Structure (ONLY if needed)
########################################
if [[ -d "$MODELS_DIR/split_files" ]]; then
    print_status "Flattening directory structure..."

    find "$MODELS_DIR" -mindepth 2 -type f \( -name "*.safetensors" -o -name "*.pth" \) -exec mv -t "$MODELS_DIR" {} +

    rm -rf "$MODELS_DIR/split_files"
fi

########################################
# Final Validation (critical)
########################################
if [[ ! -f "$WAN_T5" || ! -f "$WAN_VAE" ||
    ! -f "$WAN_DIT_HIGH" || ! -f "$WAN_DIT_LOW" ||
    ! -f "$WAN_DIT_I2V_HIGH" || ! -f "$WAN_DIT_I2V_LOW" ]]; then

    print_error "Wan 2.2 weights validation failed."
    echo "[DEBUG] Current contents:"
    find "$MODELS_DIR" -maxdepth 3
    exit 1
fi

print_success "Wan 2.2 weights ready."

########################################
# Dataset Setup
########################################
print_header "STAGE 4: DATASET PREP"

DATASET_TOML="$OUT_HIGH/dataset.toml"
if [ "${KEEP_DATASET:-0}" = "1" ] && [ -f "$DATASET_TOML" ]; then
    print_status "Keeping existing dataset.toml"
else
    print_status "Writing dataset.toml (Type: $DATASET_TYPE)"
    cat > "$DATASET_TOML" << TOML
[general]
resolution = [${RESOLUTION_LIST_NORM}]
caption_extension = "${CAPTION_EXT:-.txt}"
batch_size = ${BATCH_SIZE:-1}
enable_bucket = true
bucket_no_upscale = ${BUCKET_NO_UPSCALE}
num_repeats = ${NUM_REPEATS}

[[datasets]]
TOML
    if [ "$DATASET_TYPE" = "video" ]; then
        cat >> "$DATASET_TOML" << TOML
video_directory = "$DATASET_DIR"
target_frames = [${TARGET_FRAMES_NORM}]
frame_extraction = "${FRAME_EXTRACTION:-head}"
frame_stride = ${FRAME_STRIDE:-1}
frame_sample = ${FRAME_SAMPLE:-1}
max_frames = ${MAX_FRAMES:-129}
fp_latent_window_size = ${FP_LATENT_WINDOW_SIZE:-9}
TOML
    else
        cat >> "$DATASET_TOML" << TOML
image_directory = "${DATASET_DIR}"
cache_directory = "${WAN_CACHE_DIR}"
TOML
    fi
    print_success "dataset.toml created."
fi

########################################
# Caching
########################################
print_header "STAGE 5: PRE-CACHING"

if [ "$SKIP_CACHE" = "1" ]; then
    print_warning "Skipping caching."
else
    print_status "Caching Latents (VAE)..."
    python3 "$REPO_DIR/wan_cache_latents.py" --dataset_config "$DATASET_TOML" --vae "$WAN_VAE"
    print_status "Caching Text (T5)..."
    python3 "$REPO_DIR/wan_cache_text_encoder_outputs.py" --dataset_config "$DATASET_TOML" --t5 "$WAN_T5"
fi

########################################
# Dynamic Save Frequency
########################################
# For optimal Post-Hoc EMA quality, we save at the end of every epoch.
# This ensures each snapshot represents a full cycle of the dataset.
DYNAMIC_SAVE_STEPS=$STEPS_PER_EPOCH

# Safety floor to prevent disk thrashing on extremely small datasets/high effective batches
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
echo -e "${CYAN}Grad Accum:${NC}            $GRAD_ACCUM_STEPS (Effective Batch: $EFFECTIVE_BATCH)"
echo -e "${CYAN}Estimated Steps:${NC}       $TOTAL_STEPS"
echo -e "------------------------------------"

sleep 5

########################################
# DYNAMIC SCHEDULER & WARMUP
########################################

LR_WARMUP_STEPS=0
LR_SCHEDULER_POWER=1.0

# --- BASE WARMUP ---
if [ "$LR_SCHEDULER" == "constant" ] || [ "$OPTIMIZER_TYPE" == "adafactor" ]; then
    LR_WARMUP_STEPS=0

elif [ "$OPTIMIZER_TYPE" == "prodigyopt.Prodigy" ]; then
    if [ "$TOTAL_STEPS" -lt 400 ]; then
        LR_WARMUP_STEPS=30
    elif [ "$TOTAL_STEPS" -lt 1500 ]; then
        LR_WARMUP_STEPS=$((TOTAL_STEPS * 10 / 100))
    else
        LR_WARMUP_STEPS=$((TOTAL_STEPS * 5 / 100))
    fi

elif [ "$OPTIMIZER_TYPE" == "adamw" ] || [ "$OPTIMIZER_TYPE" == "adamw8bit" ]; then
    LR_WARMUP_STEPS=$((TOTAL_STEPS * 5 / 100))
fi

# --- SAFETY BOUNDS ---
# Only 'constant' should skip this, as it actually requires 0 warmup.
if [ "$LR_SCHEDULER" != "constant" ]; then
    # Using ceiling math for percentage bounds
    MIN_WARMUP=$(((TOTAL_STEPS * 5 + 99) / 100))
    [ "$MIN_WARMUP" -lt 20 ] && MIN_WARMUP=20

    MAX_WARMUP=$(((TOTAL_STEPS * 12 + 99) / 100))

    # clamp
    if [ "$LR_WARMUP_STEPS" -lt "$MIN_WARMUP" ]; then
        LR_WARMUP_STEPS=$MIN_WARMUP
    fi

    if [ "$LR_WARMUP_STEPS" -gt "$MAX_WARMUP" ]; then
        LR_WARMUP_STEPS=$MAX_WARMUP
    fi
fi

print_success "LR Scheduler: ${BOLD}$LR_SCHEDULER${NC}"
print_success "Warmup Steps: ${BOLD}$LR_WARMUP_STEPS${NC}"

# --- DUMP CALCULATED STATE FOR RESUME ---
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
    --dataset_config "$DATASET_TOML"
    --optimizer_type "$OPTIMIZER_TYPE"
    --lr_warmup_steps "$LR_WARMUP_STEPS"
    --lr_scheduler "$LR_SCHEDULER"
    --lr_scheduler_power "$LR_SCHEDULER_POWER"
    --learning_rate "$LEARNING_RATE"
    --gradient_accumulation_steps "$GRAD_ACCUM_STEPS"
    --max_data_loader_n_workers "$MAX_DATA_LOADER_N_WORKERS"
    --persistent_data_loader_workers
    --network_module networks.lora_wan
    --network_dim "${LORA_RANK:-64}"
    --network_alpha "${LORA_ALPHA:-32}"
    --timestep_sampling "$TIMESTEP_SAMPLING"
    --weighting_scheme none
    --discrete_flow_shift "$DISCRETE_FLOW_SHIFT"
    --max_grad_norm 1.0
    --network_dropout "$NETWORK_DROPOUT"
    --save_state
    --max_train_epochs "$MAX_TRAIN_EPOCHS"
    --save_every_n_epochs "$SAVE_EVERY_N_EPOCHS"
)

# 2. Dynamic Memory Management
if [ "${FP8_BASE:-0}" = "1" ]; then COMMON_FLAGS+=("--fp8_base"); fi
if [ "${FP8_T5:-0}" = "1" ]; then COMMON_FLAGS+=("--fp8_t5"); fi

# EMA and DYNAMIC_SAVE_STEPS
if [ "${USE_EMA:-0}" = "1" ]; then COMMON_FLAGS+=("--save_every_n_steps" "$DYNAMIC_SAVE_STEPS"); fi

# Gradient Checkpointing
if [ "${GRADIENT_CHECKPOINTING:-1}" = "1" ]; then COMMON_FLAGS+=("--gradient_checkpointing"); fi

# Attention
if [ "${ATTN:-flash}" = "flash" ]; then
    COMMON_FLAGS+=(--flash_attn --mixed_precision bf16)
elif [ "$ATTN" = "sdpa" ]; then
    COMMON_FLAGS+=(--sdpa --mixed_precision bf16)
fi

# 3. Inject Optimizer Args Array
if [ -n "${OPTIMIZER_ARGS+x}" ]; then
    for arg in "${OPTIMIZER_ARGS[@]}"; do
        COMMON_FLAGS+=("--optimizer_args" "$arg")
    done
fi

# 4. EXECUTION FLOW
if [ "$TRAIN_MODE" == "IMAGE" ]; then
    print_status "[HIGH-NOISE ONLY] Image Dataset detected. Using GPU 0."

    env CUDA_VISIBLE_DEVICES=0 accelerate launch --num_cpu_threads_per_process "$NUM_CPU_THREADS_PER_PROCESS" --mixed_precision bf16 \
        "$REPO_DIR/wan_train_network.py" --dit "$ACTIVE_DIT_HIGH" --preserve_distribution_shape \
        --min_timestep 875 --max_timestep 1000 --seed "$SEED_HIGH" \
        --output_dir "$OUT_HIGH" --output_name "$TITLE_HIGH" --logging_dir "$OUT_HIGH/logs" \
        --log_with tensorboard "${COMMON_FLAGS[@]}"

elif [ "${GPU_COUNT}" -ge 2 ]; then
    print_success "Multi-GPU Video Training! Running parallel HIGH/LOW noise flows."

    # GPU 0: HIGH NOISE
    env CUDA_VISIBLE_DEVICES=0 accelerate launch --num_cpu_threads_per_process "$NUM_CPU_THREADS_PER_PROCESS" --main_process_port 29500 --mixed_precision bf16 \
        "$REPO_DIR/wan_train_network.py" --dit "$ACTIVE_DIT_HIGH" --preserve_distribution_shape \
        --min_timestep 875 --max_timestep 1000 --seed "$SEED_HIGH" \
        --output_dir "$OUT_HIGH" --output_name "$TITLE_HIGH" --logging_dir "$OUT_HIGH/logs" \
        --log_with tensorboard "${COMMON_FLAGS[@]}" &

    # GPU 1: LOW NOISE
    env CUDA_VISIBLE_DEVICES=1 accelerate launch --num_cpu_threads_per_process "$NUM_CPU_THREADS_PER_PROCESS" --main_process_port 29501 --mixed_precision bf16 \
        "$REPO_DIR/wan_train_network.py" --dit "$ACTIVE_DIT_LOW" --preserve_distribution_shape \
        --min_timestep 0 --max_timestep 875 --seed "$SEED_LOW" \
        --output_dir "$OUT_LOW" --output_name "$TITLE_LOW" --logging_dir "$OUT_LOW/logs" \
        --log_with tensorboard "${COMMON_FLAGS[@]}" &

    wait
    print_success "Dual-GPU Training Complete."
else
    # VIDEO MODE & SINGLE GPU: Prompt User
    print_warning "Single GPU detected for Video Training. You must choose one mode."
    echo "1) HIGH-noise (GPU 0)"
    echo "2) LOW-noise (GPU 0)"
    read -rp "Selection: " choice

    DIT_PATH=$([ "$choice" = "1" ] && echo "$ACTIVE_DIT_HIGH" || echo "$ACTIVE_DIT_LOW")
    TS_MIN=$([ "$choice" = "1" ] && echo "875" || echo "0")
    TS_MAX=$([ "$choice" = "1" ] && echo "1000" || echo "875")
    NAME=$([ "$choice" = "1" ] && echo "$TITLE_HIGH" || echo "$TITLE_LOW")
    OUT=$([ "$choice" = "1" ] && echo "$OUT_HIGH" || echo "$OUT_LOW")
    SEED=$([ "$choice" = "1" ] && echo "$SEED_HIGH" || echo "$SEED_LOW")

    accelerate launch --num_cpu_threads_per_process "$NUM_CPU_THREADS_PER_PROCESS" --mixed_precision bf16 \
        "$REPO_DIR/wan_train_network.py" --dit "$DIT_PATH" --preserve_distribution_shape \
        --min_timestep "$TS_MIN" --max_timestep "$TS_MAX" --seed "$SEED" \
        --output_dir "$OUT" --output_name "$NAME" --logging_dir "$OUT/logs" \
        --log_with tensorboard "${COMMON_FLAGS[@]}"
fi

########################################
# Auto-Convert (Full Batch Mode)
########################################
print_header "STAGE 6: POST-PROCESSING"
CONVERT_SCRIPT="$REPO_DIR/convert_lora.py"

if [ -f "$CONVERT_SCRIPT" ]; then
    print_status "Scanning $OUTPUT_DIR for LoRAs to convert..."

    CONVERT_COUNT=0
    shopt -s nullglob
    for lora in "$OUTPUT_DIR"/*.safetensors; do

        # 1. Skip files that are already converted
        [[ "$lora" == *"_comfy.safetensors" ]] && continue

        # 2. Skip model_states
        [[ "$lora" == *"model_states"* ]] && continue

        # 3. NEW: Skip intermediate 'step' snapshots used for EMA
        # We keep the Epochs (-000001) and the Final/EMA models
        if [[ "$lora" == *"-step"* ]]; then
            continue
        fi

        # Define the output name
        COMFY_LORA_PATH="${lora%.safetensors}_comfy.safetensors"

        print_status "Converting $(basename "$lora")..."

        if python3 "$CONVERT_SCRIPT" --input "$lora" --output "$COMFY_LORA_PATH" --target other > /dev/null 2>&1; then
            # Deep Verify Header Integrity
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

    if [ "$CONVERT_COUNT" -eq 0 ]; then
        print_warning "No new LoRA files found to convert."
    else
        print_success "Batch conversion complete. Total converted: $CONVERT_COUNT"
    fi
else
    print_error "Conversion script not found at $CONVERT_SCRIPT"
fi

print_header "ALL WAN TASKS COMPLETE"
