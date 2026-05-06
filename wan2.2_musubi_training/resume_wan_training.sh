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
echo -e "${CYAN}#                WAN 2.2 - RESUME TRAINING SCRIPT            #${NC}"
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

gpu_count() {
    if command -v nvidia-smi > /dev/null 2>&1; then
        nvidia-smi -L 2> /dev/null | wc -l | awk '{print $1}'
    else
        echo 0
    fi
}

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
# 1. Load Config & Setup Paths
########################################
CONFIG_FILE="${1:-wan_musubi_config.sh}"
if [ ! -f "$CONFIG_FILE" ]; then
    print_error "Config file '$CONFIG_FILE' not found."
    exit 1
fi
source "$CONFIG_FILE"

# Output Dirs
TITLE_HIGH="${TITLE_HIGH:-Wan2.2_lora_high}"
TITLE_LOW="${TITLE_LOW:-Wan2.2_lora_low}"
OUT_HIGH="$NETWORK_VOLUME/output_folder_musubi/wan22/$TITLE_HIGH"
OUT_LOW="$NETWORK_VOLUME/output_folder_musubi/wan22/$TITLE_LOW"
DATASET_TOML="$OUT_HIGH/dataset.toml"
REPO_DIR="$NETWORK_VOLUME/musubi-tuner"
MODELS_DIR="$NETWORK_VOLUME/models/Wan"

export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
export PYTORCH_ALLOC_CONF=expandable_segments:True

# Model Paths
WAN_VAE="$MODELS_DIR/Wan2.1_VAE.pth"
WAN_T5="$MODELS_DIR/models_t5_umt5-xxl-enc-bf16.pth"
WAN_DIT_HIGH="$MODELS_DIR/wan2.2_t2v_high_noise_14B_fp16.safetensors"
WAN_DIT_LOW="$MODELS_DIR/wan2.2_t2v_low_noise_14B_fp16.safetensors"
WAN_DIT_I2V_HIGH="$MODELS_DIR/wan2.2_i2v_high_noise_14B_fp16.safetensors"
WAN_DIT_I2V_LOW="$MODELS_DIR/wan2.2_i2v_low_noise_14B_fp16.safetensors"

# --- TASK SELECTION (T2V vs I2V) ---
echo -e "\n${CYAN}${BOLD}Select Task Type for Resume:${NC}"
echo "1) Text-to-Video (t2v-A14B)"
echo "2) Image-to-Video (i2v-A14B)"
read -rp "Selection (1/2, default 1): " TASK_CHOICE
TASK_CHOICE=${TASK_CHOICE:-1}

if [ "$TASK_CHOICE" = "2" ]; then
    WAN_TASK="i2v-A14B"
    ACTIVE_DIT_HIGH="$WAN_DIT_I2V_HIGH"
    ACTIVE_DIT_LOW="$WAN_DIT_I2V_LOW"
    print_info "Task set to: ${BOLD}Image-to-Video (I2V)${NC}"
else
    WAN_TASK="t2v-A14B"
    ACTIVE_DIT_HIGH="$WAN_DIT_HIGH"
    ACTIVE_DIT_LOW="$WAN_DIT_LOW"
    print_info "Task set to: ${BOLD}Text-to-Video (T2V)${NC}"
fi

# --- AUTO-DETECT MODE (Image vs Video Dataset) ---
shopt -s nocasematch
if [[ "$DATASET_DIR" == *"image"* ]]; then
    TRAIN_MODE="IMAGE"
    print_info "Detection: IMAGE dataset. Enabling High-noise only."
else
    TRAIN_MODE="VIDEO"
    print_info "Detection: VIDEO dataset. Enabling Dual-Flow."
fi
shopt -u nocasematch

GPU_COUNT=$(gpu_count)

########################################
# 2. Logic for Resume & Extension
########################################

STATE_FILE="$REPO_DIR/training_state.tmp"
if [ -f "$STATE_FILE" ]; then
    source "$STATE_FILE"
    print_success "Resumed State Loaded"
else
    print_error "State file not found! Cannot resume with consistent math."
    exit 1
fi

# Detect Current Progress (Using HIGH as the master indicator)
CKPTS=($(list_checkpoints "$OUT_HIGH"))
if [ ${#CKPTS[@]} -gt 0 ]; then
    LATEST="${CKPTS[-1]}"
    LAST_NUM=$(basename "$LATEST" | grep -o "[0-9]\+" | tail -n 1 || echo "0")
    LAST_NUM=$((10#$LAST_NUM))

    echo -e "${PURPLE}================================================================${NC}"
    echo -e "${BOLD}${CYAN} RESUME CONFIGURATION (WAN 2.2) ${NC}"
    echo -e "${PURPLE}================================================================${NC}"
    echo -e "${YELLOW}Resumed State: $(basename "$LATEST") (Completed: $LAST_NUM epochs)${NC}"

    # --- A. EPOCH EXTENSION LOGIC ---
    REMAINING=$((MAX_TRAIN_EPOCHS - LAST_NUM))
    [ $REMAINING -lt 0 ] && REMAINING=0

    # Suggest 2x remaining to allow for stabilization after the Musubi reset
    SUGGESTED_EP=$((REMAINING * 2))
    [ $SUGGESTED_EP -eq 0 ] && SUGGESTED_EP=2

    echo -e "${GREEN}Musubi resets the scheduler on resume.${NC}"
    read -p "Enter ADDITIONAL epochs to run (Suggested for stability: $SUGGESTED_EP): " USER_EP
    MAX_TRAIN_EPOCHS=${USER_EP:-$SUGGESTED_EP}

    # --- B. LR STABILIZATION (The Rescue) ---
    echo -e "\n${YELLOW}[!] Scheduler Reset Protection${NC}"
    echo -e "Original Config LR: ${BOLD}$LEARNING_RATE${NC}"
    read -p "Enter 'Tail' LR for stabilization (e.g., 4e-6) or ENTER to use original: " RESCUE_LR

    if [ -n "$RESCUE_LR" ]; then
        ACTIVE_LR="$RESCUE_LR"
        ACTIVE_SCHEDULER="constant"
        print_info "Forcing ${BOLD}constant${NC} scheduler at ${BOLD}$ACTIVE_LR${NC}"
    else
        ACTIVE_LR="$LEARNING_RATE"
        ACTIVE_SCHEDULER="$LR_SCHEDULER"
        print_warning "Using original scheduler. Watch for spikes!"
    fi
else
    print_error "No checkpoints found to resume in $OUT_HIGH"
    exit 1
fi

# Apply the Rescue values to the Flags
COMMON_FLAGS=(
    --task "$WAN_TASK"
    --vae "$WAN_VAE"
    --t5 "$WAN_T5"
    --dataset_config "$DATASET_TOML"
    --optimizer_type "$OPTIMIZER_TYPE"
    --lr_warmup_steps 0
    --lr_scheduler "$ACTIVE_SCHEDULER"
    --lr_scheduler_power "$LR_SCHEDULER_POWER"
    --learning_rate "$ACTIVE_LR"
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

# Dynamic Memory Management
if [ "${FP8_BASE:-0}" = "1" ]; then COMMON_FLAGS+=("--fp8_base"); fi
if [ "${FP8_SCALED:-0}" = "1" ]; then COMMON_FLAGS+=("--fp8_scaled"); fi
if [ "${FP8_T5:-0}" = "1" ]; then COMMON_FLAGS+=("--fp8_t5"); fi

# EMA and DYNAMIC_SAVE_STEPS
if [ "${USE_EMA:-0}" = "1" ]; then COMMON_FLAGS+=("--save_every_n_steps" "$DYNAMIC_SAVE_STEPS"); fi

# Gradient Checkpointing
if [ "${GRADIENT_CHECKPOINTING:-1}" = "1" ]; then COMMON_FLAGS+=("--gradient_checkpointing"); fi

# Attention
if [ "${ATTN:-flash}" = "flash" ]; then
    COMMON_FLAGS+=(--flash_attn --mixed_precision fp16)
elif [ "$ATTN" = "sdpa" ]; then
    COMMON_FLAGS+=(--sdpa --mixed_precision fp16)
fi

# Inject Optimizer Args Array
if [ ${#OPTIMIZER_ARGS[@]} -gt 0 ]; then
    COMMON_FLAGS+=("--optimizer_args" "${OPTIMIZER_ARGS[@]}")
fi

# Prompt for Epoch Extension (Uses HIGH as the indicator)
CKPTS=($(list_checkpoints "$OUT_HIGH"))
if [ ${#CKPTS[@]} -gt 0 ]; then
    LATEST="${CKPTS[-1]}"
    LAST_NUM=$(basename "$LATEST" | grep -o "[0-9]\+" | head -n 1 || echo "0")
    if [ "$MAX_TRAIN_EPOCHS" -le "$LAST_NUM" ]; then
        print_warning "Current MAX_TRAIN_EPOCHS ($MAX_TRAIN_EPOCHS) reached (Last: $LAST_NUM)."
        read -p "Enter new total number of epochs (or Enter to keep current): " NEW_E
        if [[ "$NEW_E" =~ ^[0-9]+$ ]]; then
            MAX_TRAIN_EPOCHS="$NEW_E"
            print_success "Global target updated to $MAX_TRAIN_EPOCHS epochs."
        fi
    fi
fi

resume_model() {
    local type="$1"
    local dir="$2"
    local gpu="$3"
    local dit="$4"
    local seed="$5"
    local title="$6"
    local min_t="$7"
    local max_t="$8"

    local CKPTS=($(list_checkpoints "$dir"))
    local RESUME_ARGS=()

    if [ ${#CKPTS[@]} -gt 0 ]; then
        local LATEST="${CKPTS[-1]}"
        print_success "Launching $type Resume on GPU $gpu..."
        RESUME_ARGS=(--resume "$LATEST")

        # --- PRE-FLIGHT DISK CLEANUP ---
        # Crucial for Wan 2.2: 14B state files are ~50GB+ each.
        print_warning "Purging old states in $dir (Preserving: $(basename "$LATEST"))..."
        find "$dir" -maxdepth 1 -type d -name "*-state" ! -path "$LATEST" -exec rm -rf {} + > /dev/null 2>&1
    fi

    CUDA_VISIBLE_DEVICES="$gpu" \
        accelerate launch --num_cpu_threads_per_process "$NUM_CPU_THREADS_PER_PROCESS" --mixed_precision fp16 \
        "$REPO_DIR/wan_train_network.py" \
        --dit "$dit" \
        --preserve_distribution_shape --min_timestep "$min_t" --max_timestep "$max_t" \
        --seed "$seed" \
        --output_dir "$dir" \
        --output_name "$title" \
        --logging_dir "$dir/logs" \
        --log_with tensorboard \
        "${RESUME_ARGS[@]}" \
        "${COMMON_FLAGS[@]}"
}

########################################
# 3. Execution Launch
########################################
cd "$REPO_DIR"

if [ "$TRAIN_MODE" == "VIDEO" ]; then
    if [ "$GPU_COUNT" -ge 2 ]; then
        print_info "Launching Dual-GPU VIDEO Resume..."
        resume_model "HIGH" "$OUT_HIGH" 0 "$ACTIVE_DIT_HIGH" "$SEED_HIGH" "$TITLE_HIGH" 875 1000 &
        resume_model "LOW" "$OUT_LOW" 1 "$ACTIVE_DIT_LOW" "$SEED_LOW" "$TITLE_LOW" 0 875 &
        wait
    else
        print_info "Single GPU detected for VIDEO."
        echo "1) Resume HIGH (875-1000)"
        echo "2) Resume LOW (0-875)"
        read -rp "Selection: " choice
        [ "$choice" == "1" ] && resume_model "HIGH" "$OUT_HIGH" 0 "$ACTIVE_DIT_HIGH" "$SEED_HIGH" "$TITLE_HIGH" 875 1000
        [ "$choice" == "2" ] && resume_model "LOW" "$OUT_LOW" 0 "$ACTIVE_DIT_LOW" "$SEED_LOW" "$TITLE_LOW" 0 875
    fi
else
    print_info "Launching HIGH-noise IMAGE Resume..."
    resume_model "HIGH" "$OUT_HIGH" 0 "$ACTIVE_DIT_HIGH" "$SEED_HIGH" "$TITLE_HIGH" 875 1000
fi

########################################
# 4. Smart Incremental Auto-Convert
########################################
print_status "Starting Incremental Post-Resume Conversion..."

CONVERT_SCRIPT="$REPO_DIR/convert_lora.py"

# Verification Helper
verify_lora() {
    local file="$1"
    if [ -s "$file" ] && python3 -c "from safetensors import safe_open; f = safe_open('$file', framework='pt'); f.metadata(); f.keys()" > /dev/null 2>&1; then
        return 0 # Valid
    else
        return 1 # Corrupt
    fi
}

convert_new_wan_checkpoints() {
    local base_dir="$1"
    local dir_label="$2"

    if [ ! -d "$base_dir" ]; then
        print_warning "Directory $base_dir not found. Skipping $dir_label."
        return
    fi

    print_status "Checking for new $dir_label checkpoints..."

    local new_count=0
    shopt -s nullglob
    for lora in "$base_dir"/*.safetensors; do
        # --- FILTERS ---
        [[ "$lora" == *"_comfy.safetensors" ]] && continue
        [[ "$lora" == *"model_states"* ]] && continue

        # SKIP EMA STEPS: Intermediate snapshots are for the explorer script only
        [[ "$lora" == *"-step"* ]] && continue

        local comfy_path="${lora%.safetensors}_comfy.safetensors"

        DO_CONVERT=0
        if [ ! -f "$comfy_path" ]; then
            DO_CONVERT=1
        elif [ "$lora" -nt "$comfy_path" ]; then
            print_warning "Detected updated source: $(basename "$lora"). Re-converting..."
            DO_CONVERT=1
        elif ! verify_lora "$comfy_path"; then
            print_warning "Corrupted output detected: $(basename "$comfy_path"). Re-converting..."
            DO_CONVERT=1
        fi

        if [ "$DO_CONVERT" -eq 1 ]; then
            [ -f "$comfy_path" ] && rm -f "$comfy_path"
            print_status "Converting: $(basename "$lora")"

            if python3 "$CONVERT_SCRIPT" --input "$lora" --output "$comfy_path" --target other > /dev/null 2>&1; then
                # Double-check the NEW file
                if verify_lora "$comfy_path"; then
                    print_success "Converted & Verified: $(basename "$comfy_path")"
                    ((new_count++))
                else
                    print_error "FAILED: New conversion of $(basename "$lora") is corrupt."
                    rm -f "$comfy_path"
                fi
            else
                print_error "FAILED: Script error converting $(basename "$lora")"
            fi
        fi
    done
    shopt -u nullglob

    if [ "$new_count" -gt 0 ]; then
        print_success "Successfully converted $new_count new $dir_label checkpoints."
    fi
}

print_info "ALL TASKS COMPLETE"
