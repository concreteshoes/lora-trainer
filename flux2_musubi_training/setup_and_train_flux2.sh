#!/usr/bin/env bash

# --- COLORS & UI ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Helper for section headers
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
echo -e "${BOLD}${CYAN}FLUX.2-KLEIN-BASE-9B TRAINER${NC}"
echo -e "------------------------------------------------"

########################################
# GPU detection
########################################
print_header "STAGE 1: HARDWARE CHECK"

gpu_count() {
    if command -v nvidia-smi > /dev/null 2>&1; then
        nvidia-smi -L 2> /dev/null | wc -l | awk '{print $1}'
    elif [ -n "${CUDA_VISIBLE_DEVICES-}" ] && [ "${CUDA_VISIBLE_DEVICES}" != "" ]; then
        echo "${CUDA_VISIBLE_DEVICES}" | awk -F',' '{print NF}'
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

########################################
# CUDA compatibility check
########################################
print_status "Checking CUDA Kernel Compatibility..."

python3 << PYTHON_EOF
import sys
import torch
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'
try:
    if torch.cuda.is_available():
        x = torch.randn(1, device='cuda')
        y = x * 2
        print(f"{GREEN}✅ CUDA compatibility check passed{NC}")
    else:
        print(f"\n{RED}" + "="*70)
        print("CUDA NOT AVAILABLE")
        print("="*70 + f"{NC}")
        print("\nThis script requires CUDA to run.")
        print("SOLUTION: Please deploy with CUDA 12.8.")
        sys.exit(1)
except Exception as e:
    print(f"\n{RED}" + "="*70)
    print("CUDA KERNEL/COMPATIBILITY ERROR")
    print("="*70 + f"{NC}")
    print(f"Error details: {e}")
    sys.exit(1)
PYTHON_EOF

if [ $? -ne 0 ]; then exit 1; fi

########################################
# Load user config
########################################
print_header "STAGE 2: CONFIGURATION"

CONFIG_FILE="${CONFIG_FILE:-flux2_musubi_config.sh}"

if [ -f "$CONFIG_FILE" ]; then
    source "$CONFIG_FILE"
    print_success "Loaded config: ${BOLD}$CONFIG_FILE${NC}"
else
    print_warning "Config file $CONFIG_FILE not found! Using defaults."
fi

# Essential Variable Mapping
OUTPUT_NAME="${OUTPUT_NAME:-my_flux2_lora}"
CAPTION_EXT="${CAPTION_EXT:-.txt}"
LORA_RANK="${LORA_RANK:-32}"
LORA_ALPHA="${LORA_ALPHA:-16}"
SAVE_EVERY_N_EPOCHS="${SAVE_EVERY_N_EPOCHS:-2}"
MAX_TRAIN_EPOCHS="${MAX_TRAIN_EPOCHS:-16}"
LEARNING_RATE="${LEARNING_RATE:-1.0}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_REPEATS="${NUM_REPEATS:-10}"
NETWORK_DROPOUT="${NETWORK_DROPOUT:-0.01}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-4}"
TE_CACHE_BATCH_SIZE="${TE_CACHE_BATCH_SIZE:-4}"
OPTIMIZER_TYPE="${OPTIMIZER_TYPE:-prodigyopt.Prodigy}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-1}"
NUM_CPU_THREADS_PER_PROCESS="${NUM_CPU_THREADS_PER_PROCESS:-1}"
MAX_DATA_LOADER_N_WORKERS="${MAX_DATA_LOADER_N_WORKERS:-2}"
DISCRETE_FLOW_SHIFT="${DISCRETE_FLOW_SHIFT:-3.0}"
BUCKET_NO_UPSCALE="$(echo "${BUCKET_NO_UPSCALE:-true}" | tr '[:upper:]' '[:lower:]')"
KEEP_DATASET="${KEEP_DATASET:-0}"
SKIP_CACHE="${SKIP_CACHE:-0}"

# Base paths for Flux 2
REPO_DIR="$NETWORK_VOLUME/musubi-tuner"
DATASET_DIR="${DATASET_DIR:-$NETWORK_VOLUME/image_dataset_here}"
FLUX_CACHE_DIR="$NETWORK_VOLUME/cache/flux2"
MODELS_DIR="$NETWORK_VOLUME/models/flux2"
OUTPUT_DIR="$NETWORK_VOLUME/output_folder_musubi/flux2/$OUTPUT_NAME"

########################################
# Total steps calculation
########################################
# --- MEDIA COUNTING ---
IMG_COUNT=$(find "$DATASET_DIR" -maxdepth 1 -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.webp" \) | wc -l)

if [ "$IMG_COUNT" -eq 0 ]; then
    print_error "No images found in $DATASET_DIR! Please check your dataset path."
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

export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
export PYTORCH_ALLOC_CONF=expandable_segments:True

mkdir -p "$DATASET_DIR" "$OUTPUT_DIR" "$MODELS_DIR" "$FLUX_CACHE_DIR"
cd "$REPO_DIR"

########################################
# Numeric Normalization
########################################

normalize_numeric_csv() {
    local s="$1"
    s="$(echo "$s" | tr -d '[]"')"
    s="$(echo "$s" | sed -E 's/[[:space:]]*,[[:space:]]*/, /g; s/^[[:space:]]+|[[:space:]]+$//g')"
    echo "$s"
}

RESOLUTION_LIST_NORM="$(normalize_numeric_csv "${RESOLUTION_LIST:-"1024, 1024"}")"

########################################
# Hugging Face Login & Model Weights
########################################
print_header "STAGE 3: MODEL WEIGHTS & AUTHENTICATION"

# Specific model paths
FLUX2_MODEL="$MODELS_DIR/flux2-klein-base-9b.safetensors"
FLUX2_VAE="$MODELS_DIR/ae.safetensors"
FLUX2_TEXT_ENCODER="$MODELS_DIR/text_encoder/model-00004-of-00004.safetensors"

# Modern HF CLI syntax (2026)
HF_DL="hf download"
HF_FLAGS="--local-dir $MODELS_DIR"

# Check if essential files exist before starting
if [[ ! -f "$FLUX2_MODEL" || ! -f "$FLUX2_VAE" || ! -f "$FLUX2_TEXT_ENCODER" ]]; then
    print_warning "Core weights missing. Preparing for gated download..."

    # --- 1. Authentication Check ---
    if [ -z "${HF_TOKEN:-}" ] && [ -z "${HUGGING_FACE_TOKEN:-}" ]; then
        echo -e "${YELLOW}Hugging Face Token not found in environment.${NC}"
        echo -e "FLUX.2 (Black Forest Labs) requires gated access approval."
        read -s -p "Enter your Hugging Face Token (starts with hf_): " USER_HF_TOKEN
        echo ""
        export HF_TOKEN="$USER_HF_TOKEN"
    else
        # Standardize on HF_TOKEN for the CLI
        export HF_TOKEN="${HF_TOKEN:-$HUGGING_FACE_TOKEN}"
    fi

    # Log in using the modern CLI command
    hf auth login --token "$HF_TOKEN"

    # --- 2. Download Sequence ---

    print_status "Downloading FLUX.2-klein-base-9B DiT..."
    $HF_DL black-forest-labs/FLUX.2-klein-base-9B \
        flux2-klein-base-9b.safetensors \
        $HF_FLAGS

    print_status "Downloading FLUX.2 Root AE..."
    $HF_DL black-forest-labs/FLUX.2-dev \
        ae.safetensors \
        $HF_FLAGS

    print_status "Downloading Qwen Text Encoder (Multi-file)..."
    # Note: Modern 'hf' uses --include with space-separated patterns or repeated flags
    $HF_DL black-forest-labs/FLUX.2-klein-9B \
        --include "text_encoder/*" \
        $HF_FLAGS

    print_success "Flux.2 weights downloaded and verified."
else
    print_success "Weights already present in ${BOLD}$MODELS_DIR${NC}"
fi

FLUX2_TEXT_ENCODER=$(find "$MODELS_DIR/text_encoder" -name "*00001-of-*.safetensors" | head -n 1)

########################################
# Create/keep dataset.toml
########################################
DATASET_TOML="$OUTPUT_DIR/dataset.toml"
mkdir -p "$(dirname "$DATASET_TOML")"

if [ "${KEEP_DATASET:-0}" = "1" ] && [ -f "$DATASET_TOML" ]; then
    print_status "KEEP_DATASET=1: Using existing dataset.toml"
else
    print_status "Writing dataset.toml to $OUTPUT_DIR"
    cat > "$DATASET_TOML" << TOML
[general]
resolution = [${RESOLUTION_LIST_NORM}]
caption_extension = "${CAPTION_EXT:-.txt}"
batch_size = ${BATCH_SIZE:-1}
enable_bucket = true
bucket_no_upscale = ${BUCKET_NO_UPSCALE}

[[datasets]]
image_directory = "${DATASET_DIR}"
cache_directory = "${FLUX_CACHE_DIR}"
num_repeats = ${NUM_REPEATS:-10}
TOML
    print_success "dataset.toml written."
fi

########################################
# Pre-caching
########################################
print_header "STAGE 4: PRE-CACHING"

if [ "${SKIP_CACHE:-0}" = "1" ]; then
    print_warning "SKIP_CACHE=1: Skipping latent & Text Encoder caching."
else
    print_status "Caching latents (VAE)..."
    python3 "$REPO_DIR/src/musubi_tuner/flux_2_cache_latents.py" \
        --dataset_config "$DATASET_TOML" \
        --vae "$FLUX2_VAE" \
        --model_version klein-base-9b

    print_status "Caching Text Encoder outputs (Qwen)..."
    python3 "$REPO_DIR/src/musubi_tuner/flux_2_cache_text_encoder_outputs.py" \
        --dataset_config "$DATASET_TOML" \
        --text_encoder "$FLUX2_TEXT_ENCODER" \
        --batch_size "${TE_CACHE_BATCH_SIZE:-4}" \
        --model_version klein-base-9b \
        --fp8_text_encoder
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
# Launch training
########################################
print_header "STAGE 5: TRAINING LAUNCH"

# TensorBoard Logic
TENSORBOARD_ROOT="$NETWORK_VOLUME/output_folder_musubi"
if pgrep -f "tensorboard.*6006" > /dev/null; then
    print_success "TensorBoard already running."
else
    print_status "Starting TensorBoard on port 6006..."
    tensorboard --logdir "$TENSORBOARD_ROOT" --port 6006 --bind_all > /dev/null 2>&1 &
    print_success "TensorBoard started."
fi

echo -e "\n${BOLD}${YELLOW}View progress at:${NC} http://localhost:6006"
echo -e ""
echo -e "------------------------------------"
echo -e "${CYAN}Output Name:${NC}        $OUTPUT_NAME"
echo -e "${CYAN}Images Found:${NC}       $IMG_COUNT (Repeats: $NUM_REPEATS)"
echo -e "${CYAN}Epochs:${NC}             $MAX_TRAIN_EPOCHS"
echo -e "${CYAN}Rank / Alpha:${NC}       ${BOLD}$LORA_RANK / $LORA_ALPHA${NC}"
echo -e "${CYAN}Optimizer:${NC}          $OPTIMIZER_TYPE (LR: $LEARNING_RATE)"
echo -e "${CYAN}Grad Accum:${NC}         $GRAD_ACCUM_STEPS (Effective Batch: $EFFECTIVE_BATCH)"
echo -e "${CYAN}Estimated Steps:${NC}    $TOTAL_STEPS"
echo -e "------------------------------------"

sleep 5

########################################
# DYNAMIC SCHEDULER & WARMUP
########################################

LR_WARMUP_STEPS=0
LR_SCHEDULER_POWER=1.0

# --- BASE WARMUP ---
if [ "$OPTIMIZER_TYPE" == "prodigyopt.Prodigy" ]; then
    LR_SCHEDULER="cosine"

    if [ "$TOTAL_STEPS" -lt 400 ]; then
        LR_WARMUP_STEPS=30
    elif [ "$TOTAL_STEPS" -lt 1500 ]; then
        LR_WARMUP_STEPS=$((TOTAL_STEPS * 10 / 100))
    else
        LR_WARMUP_STEPS=$((TOTAL_STEPS * 5 / 100))
    fi

elif [ "$OPTIMIZER_TYPE" == "adafactor" ]; then
    LR_SCHEDULER="constant"
    LR_WARMUP_STEPS=0

elif [ "$OPTIMIZER_TYPE" == "adamw8bit" ]; then
    LR_SCHEDULER="cosine"
    LR_WARMUP_STEPS=$((TOTAL_STEPS * 5 / 100))

else
    LR_SCHEDULER="constant"
    LR_WARMUP_STEPS=0
fi

# --- SAFETY BOUNDS (adjusted for small dataset stability) ---
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

print_success "LR Scheduler: ${BOLD}$LR_SCHEDULER${NC}"
print_success "Warmup Steps: ${BOLD}$LR_WARMUP_STEPS${NC}"

# --- DUMP CALCULATED STATE FOR RESUME ---
STATE_FILE="$REPO_DIR/training_state.tmp"

cat << EOF > "$STATE_FILE"
LR_SCHEDULER="$LR_SCHEDULER"
LR_SCHEDULER_POWER="$LR_SCHEDULER_POWER"
DYNAMIC_SAVE_STEPS="$DYNAMIC_SAVE_STEPS"
EOF

print_success "Training state exported to $STATE_FILE"

COMMON_FLAGS=(
    --model_version klein-base-9b
    --dit "$FLUX2_MODEL"
    --vae "$FLUX2_VAE"
    --text_encoder "$FLUX2_TEXT_ENCODER"
    --dataset_config "$DATASET_TOML"
    --output_dir "$OUTPUT_DIR"
    --logging_dir "$OUTPUT_DIR/logs"
    --log_with tensorboard
    --output_name "$OUTPUT_NAME"
    --save_every_n_epochs "$SAVE_EVERY_N_EPOCHS"
    --max_train_epochs "$MAX_TRAIN_EPOCHS"
    --flash_attn --mixed_precision bf16
    --network_module networks.lora_flux_2
    --network_dim "$LORA_RANK"
    --network_alpha "$LORA_ALPHA"
    --gradient_accumulation_steps "$GRAD_ACCUM_STEPS"
    --max_data_loader_n_workers "$MAX_DATA_LOADER_N_WORKERS"
    --persistent_data_loader_workers
    --timestep_sampling flux2_shift
    --discrete_flow_shift "$DISCRETE_FLOW_SHIFT"
    --weighting_scheme none
    --fp8_text_encoder
    --network_dropout "$NETWORK_DROPOUT"
    --save_state
    --optimizer_type "$OPTIMIZER_TYPE"
    --lr_warmup_steps "$LR_WARMUP_STEPS"
    --lr_scheduler "$LR_SCHEDULER"
    --lr_scheduler_power "$LR_SCHEDULER_POWER"
    --learning_rate "$LEARNING_RATE"
    --seed 42
)

# Handle FP8 Toggles from Config
if [ "${FP8_BASE:-0}" = "1" ]; then COMMON_FLAGS+=("--fp8_base"); fi
if [ "${FP8_SCALED:-0}" = "1" ]; then COMMON_FLAGS+=("--fp8_scaled"); fi

# EMA and DYNAMIC_SAVE_STEPS
if [ "${USE_EMA:-0}" = "1" ]; then COMMON_FLAGS+=("--save_every_n_steps" "$DYNAMIC_SAVE_STEPS"); fi

# Gradient Checkpointing
if [ "${GRADIENT_CHECKPOINTING:-1}" = "1" ]; then COMMON_FLAGS+=("--gradient_checkpointing"); fi

# Inject Optimizer Args Array
if [ -n "${OPTIMIZER_ARGS+x}" ]; then
    for arg in "${OPTIMIZER_ARGS[@]}"; do
        COMMON_FLAGS+=("--optimizer_args" "$arg")
    done
fi

accelerate launch --num_cpu_threads_per_process "$NUM_CPU_THREADS_PER_PROCESS" --mixed_precision bf16 \
    "$REPO_DIR/flux_train_network.py" \
    "${COMMON_FLAGS[@]}"

########################################
# 9) Auto-Convert (Full Batch Mode)
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

print_header "ALL TASKS COMPLETE"
