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

# ==============================================================================
# STARTUP BANNER
# ==============================================================================
echo -e "${PURPLE}################################################################${NC}"
echo -e "${CYAN}#                Z-IMAGE TURBO - RESUME TRAINING SCRIPT            #${NC}"
echo -e "${PURPLE}################################################################${NC}"
echo ""

########################################
# Utility functions
########################################
print_info() { echo -e "${BLUE}[INFO]${NC} $*"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $*"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $*"; }
print_error() { echo -e "${RED}[ERROR]${NC} $*"; }
print_status() { echo -e "${CYAN}[STATUS]${NC} $*"; }

########################################
# Detect checkpoints
########################################
list_checkpoints() {
    local output_dir="$1"
    local checkpoints=()

    [ ! -d "$output_dir" ] && return

    shopt -s nullglob
    # We now look for:
    # 1. Standard 'checkpoint-*' or 'epoch-*'
    # 2. Musubi-style folders ending in '-state'
    local matches=("${output_dir}"/checkpoint-* "${output_dir}"/epoch-* "${output_dir}"/*-state)
    shopt -u nullglob

    for d in "${matches[@]}"; do
        # A folder is valid if it contains ANY of these core "resume" files
        if [ -d "$d" ]; then
            if [ -f "$d/optimizer.bin" ] || [ -f "$d/optimizer.pt" ] \
                || [ -f "$d/model_state.pt" ] || [ -d "$d/pytorch_model" ] \
                || [ -f "$d/random_states.pkl" ]; then
                checkpoints+=("$d")
            fi
        fi
    done

    # Sort numerically so the highest epoch is suggested/picked correctly
    echo "${checkpoints[@]}" | tr ' ' '\n' | sort -V | tr '\n' ' '
}

########################################
# 1. Load the config
########################################
CONFIG_FILE="${1:-z_image_musubi_config.sh}"
if [ ! -f "$CONFIG_FILE" ]; then
    print_error "Config file '$CONFIG_FILE' not found."
    exit 1
fi

# Load variables (OUTPUT_NAME, MAX_TRAIN_EPOCHS, etc.)
source "$CONFIG_FILE"

# 2. Reconstruct Path logic to match Main Script
REPO_DIR="$NETWORK_VOLUME/musubi-tuner"
MODELS_DIR="$NETWORK_VOLUME/models/z_image"
OUTPUT_DIR="$NETWORK_VOLUME/output_folder_musubi/z_image_turbo/$OUTPUT_NAME""
DATASET_TOML="$OUTPUT_DIR/dataset.toml"
ZIMAGE_MODEL="$MODELS_DIR/z_image_de_turbo_v1_bf16.safetensors"
ZIMAGE_VAE="$MODELS_DIR/ae.safetensors"
ZIMAGE_TEXT_ENCODER="$MODELS_DIR/qwen_3_4b.safetensors"

export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
export PYTORCH_ALLOC_CONF=expandable_segments:True

########################################
# 3. Detect and Select Checkpoint
########################################
if [ ! -d "$OUTPUT_DIR" ]; then
    print_error "Directory $OUTPUT_DIR does not exist. Check your OUTPUT_NAME."
    exit 1
fi

CHECKPOINTS=($(list_checkpoints "$OUTPUT_DIR"))
RESUME_CHECKPOINT=""

if [ ${#CHECKPOINTS[@]} -eq 0 ]; then
    print_error "No valid checkpoints found in $OUTPUT_DIR"
    exit 1
elif [ ${#CHECKPOINTS[@]} -eq 1 ]; then
    RESUME_CHECKPOINT="${CHECKPOINTS[0]}"
    print_info "Only one checkpoint found. Auto-selecting: $(basename "$RESUME_CHECKPOINT")"
else
    print_info "Multiple checkpoints found:"
    for i in "${!CHECKPOINTS[@]}"; do
        echo "  [$i] $(basename "${CHECKPOINTS[$i]}")"
    done
    read -p "Select checkpoint index to resume: " IDX
    if [[ "$IDX" =~ ^[0-9]+$ ]] && [ "$IDX" -lt "${#CHECKPOINTS[@]}" ]; then
        RESUME_CHECKPOINT="${CHECKPOINTS[$IDX]}"
    else
        print_error "Invalid selection."
        exit 1
    fi
fi

########################################
# 4. Epoch Extension Logic (Smart Automation)
########################################
if [ -n "$RESUME_CHECKPOINT" ]; then
    # Extract completed epochs from folder name
    FOLDER_NAME=$(basename "$RESUME_CHECKPOINT")
    LAST_VAL=$(echo "$FOLDER_NAME" | grep -o "[0-9]\+" | tail -n 1)
    LAST_VAL=$((10#$LAST_VAL))

    echo -e "${PURPLE}================================================================${NC}"
    echo -e "${BOLD}${CYAN} RESUME CONFIGURATION ${NC}"
    echo -e "${PURPLE}================================================================${NC}"
    echo -e "${YELLOW}Resumed State: $FOLDER_NAME (Completed Epochs: $LAST_VAL)${NC}"
    echo -e "Global Target in Config: ${BOLD}$MAX_TRAIN_EPOCHS${NC}"

    REMAINING=$((MAX_TRAIN_EPOCHS - LAST_VAL))
    DEFAULT_EP=$((REMAINING * 2))

    echo -e "${GREEN}Detected $LAST_VAL done. Remaining: $REMAINING.${NC}"

    read -p "Musubi needs to stabalize the learning rate, recommended to run x2 the amount - ($DEFAULT_EP), or enter custom: " USER_EP

    if [[ "$USER_EP" =~ ^[0-9]+$ ]]; then
        MAX_TRAIN_EPOCHS="$USER_EP"
    else
        MAX_TRAIN_EPOCHS="$DEFAULT_EP"
    fi

    print_info "Running $MAX_TRAIN_EPOCHS additional epochs from checkpoint."

    # 3. Load State Variables
    STATE_FILE="$REPO_DIR/training_state.tmp"
    if [ -f "$STATE_FILE" ]; then
        source "$STATE_FILE"
        echo -e "${GREEN}✅ Resumed State Loaded${NC}"
    else
        echo -e "${RED}❌ Error: State file not found. Cannot resume with consistent math!${NC}"
        exit 1
    fi

    # --- LEARNING RATE OVERRIDE ---
    echo -e "${YELLOW}Scheduler Reset Detected. Original LR was: $LEARNING_RATE${NC}"
    read -p "Enter a 'tail' learning rate (e.g., 4e-6) or press enter to keep original: " RESCUE_LR

    if [ -n "$RESCUE_LR" ]; then
        ACTIVE_LR="$RESCUE_LR"
        ACTIVE_SCHEDULER="constant"
        print_info "Forcing Constant Scheduler at $ACTIVE_LR"
    else
        ACTIVE_LR="$LEARNING_RATE"
        ACTIVE_SCHEDULER="$LR_SCHEDULER"
    fi

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
        --flash_attn --mixed_precision bf16
        --network_module networks.lora_zimage
        --network_dim "$LORA_RANK"
        --network_alpha "$LORA_ALPHA"
        --gradient_accumulation_steps "$GRAD_ACCUM_STEPS"
        --max_data_loader_n_workers "$MAX_DATA_LOADER_N_WORKERS"
        --persistent_data_loader_workers
        --resume "$RESUME_CHECKPOINT"
        --timestep_sampling "shift"
        --weighting_scheme none
        --discrete_flow_shift "$DISCRETE_FLOW_SHIFT"
        --learning_rate "$ACTIVE_LR"
        --optimizer_type "$OPTIMIZER_TYPE"
        --lr_warmup_steps "$LR_WARMUP_STEPS"
        --lr_scheduler "$ACTIVE_SCHEDULER"
        --lr_scheduler_power "$LR_SCHEDULER_POWER"
        --network_dropout "$NETWORK_DROPOUT"
        --fp8_llm
        --save_state
        --seed 42
    )

    # Dynamic FP8 Toggles
    if [ "${FP8_BASE:-0}" = "1" ]; then COMMON_FLAGS+=("--fp8_base"); fi
    if [ "${FP8_SCALED:-0}" = "1" ]; then COMMON_FLAGS+=("--fp8_scaled"); fi

    # EMA and DYNAMIC_SAVE_STEPS
    if [ "${USE_EMA:-0}" = "1" ]; then COMMON_FLAGS+=("--save_every_n_steps" "$DYNAMIC_SAVE_STEPS"); fi

    # Gradient Checkpointing
    if [ "${GRADIENT_CHECKPOINTING:-1}" = "1" ]; then COMMON_FLAGS+=("--gradient_checkpointing"); fi

    # Split Attn
    if [ "${SPLIT_ATTN:-1}" = "1" ]; then COMMON_FLAGS+=("--split_attn"); fi

    # Inject Optimizer Args
    if [ -n "${OPTIMIZER_ARGS+x}" ]; then
        for arg in "${OPTIMIZER_ARGS[@]}"; do
            COMMON_FLAGS+=("--optimizer_args" "$arg")
        done
    fi

    print_info "Launching Accelerate Resume..."
    sleep 2

    # --- PRE-FLIGHT DISK CLEANUP ---
    print_warning "Reclaiming space in $OUTPUT_DIR (Preserving: $(basename "$RESUME_CHECKPOINT"))..."
    find "$OUTPUT_DIR" -maxdepth 1 -type d -name "*-state" ! -path "$RESUME_CHECKPOINT" -exec rm -rf {} + > /dev/null 2>&1

    accelerate launch --num_cpu_threads_per_process "$NUM_CPU_THREADS_PER_PROCESS" --mixed_precision bf16 \
        "$REPO_DIR/zimage_train_network.py" \
        "${COMMON_FLAGS[@]}"
fi

########################################
# 6. Incremental Post-Resume Conversion
########################################
print_info "STAGE: POST-RESUME CONVERSION"
CONVERT_SCRIPT="$REPO_DIR/convert_lora.py"

if [ -f "$CONVERT_SCRIPT" ]; then
    CONVERT_COUNT=0
    shopt -s nullglob
    for lora in "$OUTPUT_DIR"/*.safetensors; do
        # 1. Standard Skips
        [[ "$lora" == *"_comfy.safetensors" ]] && continue
        [[ "$lora" == *"model_states"* ]] && continue

        # 2. EMA STEP SKIP: Prevent conversion of intermediate snapshots
        # These are reserved for the EMA explorer script.
        [[ "$lora" == *"-step"* ]] && continue

        COMFY_PATH="${lora%.safetensors}_comfy.safetensors"

        # Conversion Logic
        DO_CONVERT=0
        if [ ! -f "$COMFY_PATH" ]; then
            DO_CONVERT=1
        elif [ "$lora" -nt "$COMFY_PATH" ]; then
            print_warning "Detected updated source: $(basename "$lora"). Re-converting..."
            DO_CONVERT=1
        elif ! python3 -c "from safetensors import safe_open; f = safe_open('$COMFY_PATH', framework='pt'); f.metadata(); f.keys()" > /dev/null 2>&1; then
            print_warning "Corrupted output detected: $(basename "$COMFY_PATH"). Re-converting..."
            DO_CONVERT=1
        fi

        if [ "$DO_CONVERT" -eq 1 ]; then
            [ -f "$COMFY_PATH" ] && rm -f "$COMFY_PATH"
            print_status "Converting: $(basename "$lora")"

            if python3 "$CONVERT_SCRIPT" --input "$lora" --output "$COMFY_PATH" --target other > /dev/null 2>&1; then
                # Sanity check the result
                if python3 -c "from safetensors import safe_open; f = safe_open('$COMFY_PATH', framework='pt'); f.metadata(); f.keys()" > /dev/null 2>&1; then
                    print_success "Converted & Verified: $(basename "$COMFY_PATH")"
                    ((CONVERT_COUNT++))
                else
                    print_error "CRITICAL: $(basename "$COMFY_PATH") is still corrupt."
                    rm -f "$COMFY_PATH"
                fi
            else
                print_error "Failed to convert $(basename "$lora")"
            fi
        fi
    done
    shopt -u nullglob
    [ "$CONVERT_COUNT" -eq 0 ] && print_success "All checkpoints up to date." || print_success "Done ($CONVERT_COUNT new/updated files)."
else
    print_error "Conversion script not found at $CONVERT_SCRIPT"
fi

print_info "ALL TASKS COMPLETE"
