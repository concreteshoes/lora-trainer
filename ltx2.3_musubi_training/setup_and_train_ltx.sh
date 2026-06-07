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
echo -e "${BOLD}${CYAN}LTX 2.3 VIDEO / IMAGE TRAINER${NC}"
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
CONFIG_FILE="${CONFIG_FILE:-ltx_musubi_config.sh}"
if [ -f "$CONFIG_FILE" ]; then
    source "$CONFIG_FILE"
    print_success "Loaded config: ${BOLD}$CONFIG_FILE${NC}"
else
    print_error "Config file $CONFIG_FILE not found!"
    exit 1
fi

# --- Unified Variable Mapping ---
OUTPUT_NAME="${OUTPUT_NAME:-my_ltx23_lora}"
CAPTION_EXT="${CAPTION_EXT:-.txt}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-2}"
NUM_REPEATS="${NUM_REPEATS:-5}"
MAX_TRAIN_EPOCHS="${MAX_TRAIN_EPOCHS:-100}"
SAVE_EVERY_N_EPOCHS="${SAVE_EVERY_N_EPOCHS:-20}"
ACCUMULATION_GROUP_REMAINDER="${ACCUMULATION_GROUP_REMAINDER:-drop}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
OPTIMIZER_TYPE="${OPTIMIZER_TYPE:-adamw8bit}"
LR_SCHEDULER="${LR_SCHEDULER:-cosine}"
TIMESTEP_SAMPLING="${TIMESTEP_SAMPLING:-shifted_logit_normal}"
NETWORK_DROPOUT="${NETWORK_DROPOUT:-0}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-1}"
NUM_CPU_THREADS_PER_PROCESS="${NUM_CPU_THREADS_PER_PROCESS:-1}"
MAX_DATA_LOADER_N_WORKERS="${MAX_DATA_LOADER_N_WORKERS:-2}"
BUCKET_NO_UPSCALE="$(echo "${BUCKET_NO_UPSCALE:-true}" | tr '[:upper:]' '[:lower:]')"
KEEP_DATASET="${KEEP_DATASET:-0}"
SKIP_CACHE="${SKIP_CACHE:-0}"
# LoRA Specifics
LORA_RANK="${LORA_RANK:-32}"
LORA_ALPHA="${LORA_ALPHA:-32}"
# Video Specifics
TARGET_FRAMES="${TARGET_FRAMES:-1, 40, 80}"
FRAME_EXTRACTION="${FRAME_EXTRACTION:-full}"
# Derived Paths
DATASET_DIR="${DATASET_DIR:-$NETWORK_VOLUME/video_dataset_here}"
REPO_DIR="$NETWORK_VOLUME/musubi-tuner"
LTX_CACHE_DIR="$NETWORK_VOLUME/cache/ltx"
MODELS_DIR="$NETWORK_VOLUME/models/ltx"
LTX_DIT="$MODELS_DIR/ltx-2.3-22b-dev-fp8.safetensors"
LTX_TE="$MODELS_DIR/gemma_3_12B_it_fp8_e4m3fn.safetensors"
OUTPUT_DIR="$NETWORK_VOLUME/output_folder_musubi/ltx23/$OUTPUT_NAME"

export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
export PYTORCH_ALLOC_CONF="expandable_segments:True"

# Remove sub directories for the video dataset
find "$NETWORK_VOLUME/video_dataset_here" -mindepth 1 -maxdepth 1 -type d -exec rm -rf {} +

# --- DATASET AUTO-DETECTION ---
shopt -s nocasematch
if [[ "$DATASET_DIR" == *"image"* ]]; then
    DATASET_TYPE="image"
    print_status "Dataset Type: ${BOLD}IMAGE${NC}"
else
    DATASET_TYPE="video"
    print_status "Dataset Type: ${BOLD}VIDEO${NC}"
fi
shopt -u nocasematch

mkdir -p "$DATASET_DIR" "$OUTPUT_DIR" "$MODELS_DIR" "$LTX_CACHE_DIR"
cd "$REPO_DIR"

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
# Model Download Section (Robust HF CLI)
########################################
print_header "STAGE 3: MODEL ACQUISITION (LTX-2.3)"

# Ensure huggingface-cli is used (standardizes "hf download" alias)
HF_DL="hf download"
HF_FLAGS="--local-dir $MODELS_DIR"

# Clean up stale locks from interrupted transfers
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
        echo -e "${BLUE}[INFO]${NC} Attempt $attempt → Fetching $(basename "$remote_file")..."

        # Execute Hugging Face CLI
        $HF_DL "$repo" "$remote_file" $HF_FLAGS

        local actual_download_path="$MODELS_DIR/$remote_file"

        # If downloaded path differs from expected path, move it
        if [[ -f "$actual_download_path" && "$actual_download_path" != "$expected_path" ]]; then
            print_status "Moving $(basename "$remote_file") to root models directory..."
            mv "$actual_download_path" "$expected_path"
        fi

        # Verification check
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
# 1. Download LTX Weights
########################################
download_if_missing \
    "Lightricks/LTX-2.3-fp8" \
    "$LTX_DIT" \
    "ltx-2.3-22b-dev-fp8.safetensors"

download_if_missing \
    "GitMylo/LTX-2-comfy_gemma_fp8_e4m3fn" \
    "$LTX_TE" \
    "gemma_3_12B_it_fp8_e4m3fn.safetensors"

########################################
# Final Validation
########################################
MISSING_WEIGHTS=false

if [[ ! -f "$LTX_DIT" || ! -f "$LTX_TE" ]]; then
    MISSING_WEIGHTS=true
fi

if [ "$MISSING_WEIGHTS" = true ]; then
    print_error "Weight validation failed. Missing LTX-2.3 core files."
    echo "[DEBUG] Current contents of $MODELS_DIR:"
    find "$MODELS_DIR" -maxdepth 3
    exit 1
fi

print_success "LTX 2.3 weights verified and ready."

########################################
# Dataset Setup
########################################
print_header "STAGE 4: DATASET PREP"

DATASET_TOML="$OUTPUT_DIR/dataset.toml"

if [ "${KEEP_DATASET:-0}" = "1" ] && [ -f "$DATASET_TOML" ]; then
    print_status "Keeping existing dataset.toml in $(basename "$OUTPUT_DIR")"
else
    print_status "Writing dataset.toml for $(basename "$OUTPUT_DIR") (Type: ${LTX_MODE:-video})"

    # 1. Global settings block
    cat > "$DATASET_TOML" << TOML
[general]
resolution = [${RESOLUTION_LIST_NORM}]
caption_extension = "${CAPTION_EXT:-.txt}"
batch_size = ${BATCH_SIZE:-1}
enable_bucket = true
bucket_no_upscale = ${BUCKET_NO_UPSCALE:-true}

TOML

    # 2. Individual dataset specification block
    if [ "${LTX_MODE:-video}" = "video" ] || [ "${LTX_MODE:-video}" = "av" ]; then
        cat >> "$DATASET_TOML" << TOML
[[datasets]]
video_directory = "$DATASET_DIR"
cache_directory = "${LTX_CACHE_DIR:-$CACHE_DIR}"
num_repeats = ${NUM_REPEATS:-5}
target_frames = [${TARGET_FRAMES_NORM}]
frame_extraction = "${FRAME_EXTRACTION:-full}"
TOML
        # Append specific parameters cleanly based on extraction strategy
        case "$FRAME_EXTRACTION" in
            "slide") echo "frame_stride = ${FRAME_STRIDE:-1}" >> "$DATASET_TOML" ;;
            "uniform") echo "frame_sample = ${FRAME_SAMPLE:-4}" >> "$DATASET_TOML" ;;
            "head") ;; # no extra key needed
            *) echo "max_frames = ${MAX_FRAMES:-100}" >> "$DATASET_TOML" ;;
        esac
    else
        # Image dataset fallback configuration
        cat >> "$DATASET_TOML" << TOML
[[datasets]]
image_directory = "${DATASET_DIR}"
cache_directory = "${LTX_CACHE_DIR:-$CACHE_DIR}"
num_repeats = ${NUM_REPEATS:-5}
TOML
    fi

    print_success "dataset.toml successfully created in $(basename "$OUTPUT_DIR")."
fi

########################################
# Caching
########################################
print_header "STAGE 5: CACHING PIPELINE"
if [ "${SKIP_CACHE:-0}" -eq 1 ]; then
    print_warning "SKIP_CACHE is set to 1. Skipping all caching scripts and reusing existing cache."
else
    print_status "SKIP_CACHE is 0. Flushing old cache to ensure a clean, absolute rebuild..."
    if [ -d "$LTX_CACHE_DIR" ]; then
        rm -rf "${LTX_CACHE_DIR:?}/"*
    fi

    print_status "Running LTX Latent Caching..."
    python3 ltx2_cache_latents.py \
        --dataset_config "$DATASET_TOML" \
        --ltx2_checkpoint "$LTX_DIT"
    if [ $? -ne 0 ]; then
        print_error "Latent caching failed!"
        exit 1
    fi
    print_success "Latent caching complete."

    print_status "Running Text Encoder Caching..."
    python3 ltx2_cache_text_encoder_outputs.py \
        --dataset_config "$DATASET_TOML" \
        --ltx2_checkpoint "$LTX_DIT" \
        --gemma_safetensors "$LTX_TE" \
        --batch_size "${BATCH_SIZE:-1}"
    if [ $? -ne 0 ]; then
        print_error "Text caching failed!"
        exit 1
    fi
    print_success "Text caching complete."
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
echo -e "${CYAN}Output Name:${NC}           $OUTPUT_NAME"
echo -e "${CYAN}Detected Type:${NC}         ${BOLD}$DATASET_TYPE${NC} ($IMG_COUNT files)"
echo -e "${CYAN}Frames:${NC}                $TARGET_FRAMES"
echo -e "------------------------------------"
echo -e "${CYAN}Rank / Alpha:${NC}          $LORA_RANK / $LORA_ALPHA"
echo -e "${CYAN}Timestep sampling:${NC}     $TIMESTEP_SAMPLING"
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
    --ltx2_checkpoint "$LTX_DIT"
    --ltx_version "${LTX_VERSION:-2.3}"
    --ltx2_mode "${LTX_MODE:-video}"
    --gemma_safetensors "$LTX_TE"
    --dataset_config "$DATASET_TOML"
    --output_dir "$OUTPUT_DIR"
    --output_name "$OUTPUT_NAME"
    --network_module networks.lora_ltx2
    --network_dim "$LORA_RANK"
    --network_alpha "$LORA_ALPHA"
    --learning_rate "$LEARNING_RATE"
    --optimizer_type "$OPTIMIZER_TYPE"
    --lr_warmup_steps "$LR_WARMUP_STEPS"
    --lr_scheduler "$LR_SCHEDULER"
    --lr_scheduler_power "$LR_SCHEDULER_POWER"
    --max_train_epochs "$MAX_TRAIN_EPOCHS"
    --save_every_n_epochs "$SAVE_EVERY_N_EPOCHS"
    --gradient_accumulation_steps "$GRAD_ACCUM_STEPS"
    --accumulation_group_remainder "${ACCUMULATION_GROUP_REMAINDER:-drop}"
    --caption_dropout_rate "${CAPTION_DROPOUT_RATE:-0}"
    --network_dropout "$NETWORK_DROPOUT"
    --save_state
    --fp8_base
    --weighting_scheme none
    --timestep_sampling "$TIMESTEP_SAMPLING"
    --seed 42
)
if [ "$OPTIMIZER_TYPE" == "adafactor" ]; then COMMON_FLAGS+=("--max_grad_norm" "0"); fi
if [ "$TIMESTEP_SAMPLING" != "shifted_logit_normal" ]; then COMMON_FLAGS+=("--discrete_flow_shift" "$DISCRETE_FLOW_SHIFT"); fi
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
print_header "STAGE 7: TRAINING EXECUTION"

if [ "${GPU_COUNT}" -ge 2 ]; then
    print_success "Multi-GPU Training detected! Launching DDP across ${GPU_COUNT} GPUs."
else
    print_success "Single GPU Training detected."
fi

# A single, clean accelerate launch command handles both Single and Multi-GPU scaling automatically
accelerate launch \
    --num_processes "$GPU_COUNT" \
    --num_cpu_threads_per_process "$NUM_CPU_THREADS_PER_PROCESS" \
    --mixed_precision bf16 \
    "$REPO_DIR/ltx2_train_network.py" \
    --logging_dir "$OUTPUT_DIR/logs" \
    --log_with tensorboard \
    "${COMMON_FLAGS[@]}"

# Catch any fatal Python execution errors
if [ $? -ne 0 ]; then
    print_error "Training failed during execution!"
    exit 1
fi

print_success "Training process finished successfully."

########################################
# Auto-Convert (Full Batch Mode)
########################################
print_header "STAGE 8: POST-PROCESSING"
CONVERT_SCRIPT="$REPO_DIR/convert_lora.py"

if [ -f "$CONVERT_SCRIPT" ]; then
    DIRS_TO_SCAN=("$OUTPUT_DIR")
    CONVERT_COUNT=0
    for TARGET_DIR in "${DIRS_TO_SCAN[@]}"; do
        if [ ! -d "$TARGET_DIR" ]; then continue; fi
        print_status "Scanning $TARGET_DIR for LoRAs to convert..."
        shopt -s nullglob
        for lora in "$TARGET_DIR"/*.safetensors; do
            [[ "$lora" == *"_comfy.safetensors" ]] && continue
            [[ "$lora" == *"model_states"* ]] && continue
            [[ "$lora" == *"-step"* ]] && continue

            # Skip Musubi-Tuner intermediate epoch saves (e.g., my_lora-000001.safetensors)
            if [[ "$lora" =~ -[0-9]{6}\.safetensors$ ]]; then continue; fi

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
