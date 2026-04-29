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
echo -e "${BOLD}${CYAN}Z-IMAGE TURBO TRAINER${NC}"
echo -e "------------------------------------"

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

CONFIG_FILE="${CONFIG_FILE:-z_image_musubi_config.sh}"

if [ -f "$CONFIG_FILE" ]; then
    source "$CONFIG_FILE"
    print_success "Loaded config: ${BOLD}$CONFIG_FILE${NC}"
else
    print_error "Config file $CONFIG_FILE not found!"
    exit 1
fi

# --- Unified Variable Mapping (The "Bridge") ---
OUTPUT_NAME="${OUTPUT_NAME:-my_zimage_lora}"
CAPTION_EXT="${CAPTION_EXT:-.txt}"
MAX_TRAIN_EPOCHS="${MAX_TRAIN_EPOCHS:-60}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-4}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_REPEATS="${NUM_REPEATS:-15}"
OPTIMIZER_TYPE="${OPTIMIZER_TYPE:-prodigyopt.Prodigy}"
LR_SCHEDULER="${LR_SCHEDULER:-cosine}"
TIMESTEP_SAMPLING="${TIMESTEP_SAMPLING:-shift}"
LEARNING_RATE="${LEARNING_RATE:-1.0}"
SAVE_EVERY_N_EPOCHS="${SAVE_EVERY_N_EPOCHS:-1}"
TE_CACHE_BATCH_SIZE="${TE_CACHE_BATCH_SIZE:-8}"
NETWORK_DROPOUT="${NETWORK_DROPOUT:-0.01}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-1}"
SPLIT_ATTN="${SPLIT_ATTN:-1}"
NUM_CPU_THREADS_PER_PROCESS="${NUM_CPU_THREADS_PER_PROCESS:-1}"
MAX_DATA_LOADER_N_WORKERS="${MAX_DATA_LOADER_N_WORKERS:-2}"
DISCRETE_FLOW_SHIFT="${DISCRETE_FLOW_SHIFT:-2.5}"
BUCKET_NO_UPSCALE="$(echo "${BUCKET_NO_UPSCALE:-true}" | tr '[:upper:]' '[:lower:]')"
KEEP_DATASET="${KEEP_DATASET:-0}"
SKIP_CACHE="${SKIP_CACHE:-0}"

# LoRA Specifics
LORA_RANK="${LORA_RANK:-${NETWORK_DIM:-64}}"
LORA_ALPHA="${LORA_ALPHA:-${NETWORK_ALPHA:-64}}"

# Derived Paths
OUTPUT_DIR="$NETWORK_VOLUME/output_folder_musubi/z_image_turbo/$OUTPUT_NAME"
DATASET_DIR="${DATASET_DIR:-$NETWORK_VOLUME/image_dataset_here}"
REPO_DIR="$NETWORK_VOLUME/musubi-tuner"
MODELS_DIR="$NETWORK_VOLUME/models/z_image"
ZIMAGE_CACHE_DIR="$NETWORK_VOLUME/cache/z_image_turbo"
ZIMAGE_MODEL="$MODELS_DIR/z_image_de_turbo_v1_bf16.safetensors"
ZIMAGE_VAE="$MODELS_DIR/ae.safetensors"
ZIMAGE_TEXT_ENCODER="$MODELS_DIR/qwen_3_4b.safetensors"

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

mkdir -p "$DATASET_DIR" "$OUTPUT_DIR" "$MODELS_DIR" "$ZIMAGE_CACHE_DIR"
cd "$REPO_DIR"

# --- ROBUST HOTFIX BLOCK ---
TARGET_FILE="$REPO_DIR/src/musubi_tuner/zimage_generate_image.py"

if [ -f "$TARGET_FILE" ]; then
    # Check if the file still contains the 'mask' bug
    if grep -q "dtype=torch.bfloat16" "$TARGET_FILE"; then
        echo -e "${YELLOW}🛠️ Patching Mask Dtype Bug...${NC}"
        sed -i 's/\["mask"\]\.to(device, dtype=torch\.bfloat16)/["mask"].to(device)/g' "$TARGET_FILE"
    fi

    # Check if the file still contains the 'multiplier' bug
    if grep -q "default=1.0, help=\"lora multiplier\"" "$TARGET_FILE"; then
        echo -e "${YELLOW}🛠️ Patching Multiplier Bug...${NC}"
        sed -i 's/default=1\.0,\s*help="lora multiplier"/default=None, help="lora multiplier"/g' "$TARGET_FILE"
    fi
else
    echo -e "${RED}⚠️ Warning: $TARGET_FILE not found, skipping patches.${NC}"
fi

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
# Weight Initialization (Z-Image Turbo)
########################################
print_header "STAGE 3: MODEL WEIGHTS (Z-IMAGE TURBO)"

HF_DL="hf download"
HF_FLAGS="--local-dir $MODELS_DIR"

# Optional: clear stale locks
find "$MODELS_DIR/.cache/huggingface" -name "*.lock" -type f -delete 2> /dev/null || true

########################################
# Retry Download Function (File आधारित)
########################################
retry_file_download() {
    local repo="$1"
    local file="$2"
    local expected_path="$3"

    local max_retries=5
    local attempt=1
    local delay=5

    while [[ $attempt -le $max_retries ]]; do
        echo "[INFO] Attempt $attempt → Fetching $file..."

        $HF_DL "$repo" "$file" $HF_FLAGS

        # --- VALIDATION ---
        if [[ -f "$expected_path" && -s "$expected_path" ]]; then
            echo "[OK] Verified: $(basename "$expected_path")"
            return 0
        fi

        echo "[WARN] Download failed or incomplete for $file. Retrying in ${delay}s..."
        sleep $delay

        ((attempt++))
        delay=$((delay * 2))
    done

    print_error "Failed to download $file after $max_retries attempts"
    return 1
}

########################################
# Expected Paths (after flatten)
########################################
ZIMAGE_MODEL="$MODELS_DIR/z_image_de_turbo_v1_bf16.safetensors"
ZIMAGE_VAE="$MODELS_DIR/ae.safetensors"
ZIMAGE_TEXT_ENCODER="$MODELS_DIR/qwen_3_4b.safetensors"

########################################
# Download if missing
########################################
if [[ ! -f "$ZIMAGE_MODEL" || ! -f "$ZIMAGE_VAE" || ! -f "$ZIMAGE_TEXT_ENCODER" ]]; then
    print_warning "Z-Image Turbo weights missing. Downloading via HF CLI..."

    # 1. DiT
    retry_file_download \
        "ostris/Z-Image-De-Turbo" \
        "z_image_de_turbo_v1_bf16.safetensors" \
        "$ZIMAGE_MODEL" || exit 1

    # 2. VAE (nested path)
    retry_file_download \
        "Comfy-Org/z_image_turbo" \
        "split_files/vae/ae.safetensors" \
        "$MODELS_DIR/split_files/vae/ae.safetensors" || exit 1

    # 3. Text Encoder (nested path)
    retry_file_download \
        "Comfy-Org/z_image_turbo" \
        "split_files/text_encoders/qwen_3_4b.safetensors" \
        "$MODELS_DIR/split_files/text_encoders/qwen_3_4b.safetensors" || exit 1

    ########################################
    # Flatten Structure (ONLY after success)
    ########################################
    print_status "Flattening directory structure..."

    find "$MODELS_DIR" -mindepth 2 -type f -name "*.safetensors" -exec mv -t "$MODELS_DIR" {} +

    # Cleanup
    rm -rf "$MODELS_DIR/split_files"

    ########################################
    # Final Validation
    ########################################
    if [[ ! -f "$ZIMAGE_MODEL" || ! -f "$ZIMAGE_VAE" || ! -f "$ZIMAGE_TEXT_ENCODER" ]]; then
        print_error "Final validation failed after flattening."
        find "$MODELS_DIR" -maxdepth 3
        exit 1
    fi

    print_success "Z-Image Turbo weights downloaded and verified."

else
    print_success "Weights already present in ${BOLD}$MODELS_DIR${NC}"
fi
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
caption_extension = "${CAPTION_EXT}"
batch_size = ${BATCH_SIZE}
enable_bucket = true
bucket_no_upscale = ${BUCKET_NO_UPSCALE}

[[datasets]]
image_directory = "${DATASET_DIR}"
cache_directory = "${ZIMAGE_CACHE_DIR}"
num_repeats = ${NUM_REPEATS}
TOML
    print_success "dataset.toml written."
fi

########################################
# Pre-caching (Z-Image Specialization)
########################################
print_header "STAGE 4: PRE-CACHING"

if [ "${SKIP_CACHE:-0}" = "1" ]; then
    print_warning "SKIP_CACHE=1: Skipping caching."
else
    print_status "Caching latents..."
    # Z-Image uses the standard latent cache
    python3 "$REPO_DIR/zimage_cache_latents.py" \
        --dataset_config "$DATASET_TOML" --vae "$ZIMAGE_VAE"

    print_status "Caching Text Encoder (Qwen-3.4B)..."
    # Note: Qwen-3.4B is large. TE_CACHE_BATCH_SIZE matters here.
    python3 "$REPO_DIR/zimage_cache_text_encoder_outputs.py" \
        --dataset_config "$DATASET_TOML" \
        --text_encoder "$ZIMAGE_TEXT_ENCODER" \
        --batch_size "$TE_CACHE_BATCH_SIZE" \
        --fp8_llm
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

TENSORBOARD_FOLDER="$NETWORK_VOLUME/output_folder_musubi"
print_status "TensorBoard logs for this run are located at:\n$TENSORBOARD_FOLDER\n"

echo -e "\n${BOLD}${YELLOW}View progress at:${NC} http://localhost:6006"
echo -e ""
echo -e "------------------------------------"
echo -e "${CYAN}Output Name:${NC}         $OUTPUT_NAME"
echo -e "${CYAN}Images Found:${NC}        $IMG_COUNT (Repeats: $NUM_REPEATS)"
echo -e "${CYAN}Epochs:${NC}              $MAX_TRAIN_EPOCHS"
echo -e "${CYAN}Rank / Alpha:${NC}        $LORA_RANK / $LORA_ALPHA"
echo -e "${CYAN}Timestep sampling:${NC}   $TIMESTEP_SAMPLING"
echo -e "${CYAN}Flow shift:${NC}          $DISCRETE_FLOW_SHIFT"
echo -e "${CYAN}Optimizer:${NC}           $OPTIMIZER_TYPE (LR: $LEARNING_RATE)"
echo -e "${CYAN}Scheduler:${NC}           $LR_SCHEDULER"
echo -e "${CYAN}Attention:${NC}           $ATTN"
echo -e "${CYAN}Network dropout:${NC}     $NETWORK_DROPOUT"
echo -e "${CYAN}Grad Accum:${NC}          $GRAD_ACCUM_STEPS (Effective Batch: $EFFECTIVE_BATCH)"
echo -e "${CYAN}Estimated Steps:${NC}     $TOTAL_STEPS"
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
    --dit "$ZIMAGE_MODEL"
    --vae "$ZIMAGE_VAE"
    --text_encoder "$ZIMAGE_TEXT_ENCODER"
    --dataset_config "$DATASET_TOML"
    --output_dir "$OUTPUT_DIR"
    --logging_dir "$OUTPUT_DIR/logs"
    --log_with tensorboard
    --output_name "$OUTPUT_NAME"
    --save_every_n_epochs "$SAVE_EVERY_N_EPOCHS"
    --max_train_epochs "$MAX_TRAIN_EPOCHS"
    --network_module networks.lora_zimage
    --network_dim "$LORA_RANK"
    --network_alpha "$LORA_ALPHA"
    --gradient_accumulation_steps "$GRAD_ACCUM_STEPS"
    --max_data_loader_n_workers "$MAX_DATA_LOADER_N_WORKERS"
    --persistent_data_loader_workers
    --timestep_sampling "$TIMESTEP_SAMPLING"
    --weighting_scheme none
    --discrete_flow_shift "$DISCRETE_FLOW_SHIFT"
    --learning_rate "$LEARNING_RATE"
    --optimizer_type "$OPTIMIZER_TYPE"
    --lr_warmup_steps "$LR_WARMUP_STEPS"
    --lr_scheduler "$LR_SCHEDULER"
    --lr_scheduler_power "$LR_SCHEDULER_POWER"
    --network_dropout "$NETWORK_DROPOUT"
    --fp8_llm
    --save_state
    --seed 42
)

# Dynamic FP8 Toggles for Z-Image
if [ "${FP8_BASE:-0}" = "1" ]; then COMMON_FLAGS+=("--fp8_base"); fi
if [ "${FP8_SCALED:-0}" = "1" ]; then COMMON_FLAGS+=("--fp8_scaled"); fi

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

# Split Attn
if [ "${SPLIT_ATTN:-0}" = "1" ]; then COMMON_FLAGS+=("--split_attn"); fi

# Inject Optimizer Args Array
if [ ${#OPTIMIZER_ARGS[@]} -gt 0 ]; then
    COMMON_FLAGS+=("--optimizer_args" "${OPTIMIZER_ARGS[@]}")
fi

accelerate launch --num_cpu_threads_per_process "$NUM_CPU_THREADS_PER_PROCESS" --mixed_precision bf16 \
    "$REPO_DIR/zimage_train_network.py" \
    "${COMMON_FLAGS[@]}"

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

print_header "ALL TASKS COMPLETE"
