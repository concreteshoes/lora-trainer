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
print_info() { echo -e "${BLUE}[INFO]${NC} $*"; }
print_status() { echo -e "${CYAN}[STATUS]${NC} $*"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $*"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $*"; }
print_error() { echo -e "${RED}[ERROR]${NC} $*"; }

echo -e "${BOLD}${CYAN}LTX 2.3 ADVANCED INTERACTIVE RESUME RUNNER${NC}"
echo -e "---------------------------------------"

########################################
# STAGE 1: HARDWARE CHECK
########################################
print_header "STAGE 1: HARDWARE CHECK"

gpu_count() {
    if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
        echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l | awk '{print $1}'
    elif command -v nvidia-smi > /dev/null 2>&1; then
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
    local matches=("${output_dir}"/checkpoint-* "${output_dir}"/epoch-* "${output_dir}"/*-state)
    shopt -u nullglob

    for d in "${matches[@]}"; do
        if [ -d "$d" ]; then
            if [ -f "$d/optimizer.bin" ] || [ -f "$d/optimizer.pt" ] \
                || [ -f "$d/model_state.pt" ] || [ -d "$d/pytorch_model" ] \
                || [ -f "$d/random_states.pkl" ]; then
                checkpoints+=("$d")
            fi
        fi
    done
    echo "${checkpoints[@]}" | tr ' ' '\n' | sort -V | tr '\n' ' '
}

GPU_COUNT=$(gpu_count)
[ "${GPU_COUNT}" -lt 1 ] && {
    print_error "No CUDA GPUs detected. Aborting."
    exit 1
}
print_success "Detected/Allocated GPUs for this run: ${BOLD}${GPU_COUNT}${NC}"

########################################
# STAGE 2: CONFIGURATION LOADING
########################################
print_header "STAGE 2: CONFIGURATION LOADING"
CONFIG_FILE="${CONFIG_FILE:-ltx_musubi_config.sh}"
if [ -f "$CONFIG_FILE" ]; then
    source "$CONFIG_FILE"
    print_success "Loaded config: ${BOLD}$CONFIG_FILE${NC}"
else
    print_error "Config file $CONFIG_FILE not found!"
    exit 1
fi

# --- Core Setup Variables ---
OUTPUT_NAME="${OUTPUT_NAME:-my_ltx23_lora}"
MAX_TRAIN_EPOCHS="${MAX_TRAIN_EPOCHS:-100}"
SAVE_EVERY_N_EPOCHS="${SAVE_EVERY_N_EPOCHS:-20}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-2}"
ACCUMULATION_GROUP_REMAINDER="${ACCUMULATION_GROUP_REMAINDER:-drop}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
OPTIMIZER_TYPE="${OPTIMIZER_TYPE:-adamw8bit}"
LR_SCHEDULER="${LR_SCHEDULER:-cosine}"
TIMESTEP_SAMPLING="${TIMESTEP_SAMPLING:-shifted_logit_normal}"
NETWORK_DROPOUT="${NETWORK_DROPOUT:-0}"
NUM_CPU_THREADS_PER_PROCESS="${NUM_CPU_THREADS_PER_PROCESS:-1}"

# Paths
REPO_DIR="$NETWORK_VOLUME/musubi-tuner"
MODELS_DIR="$NETWORK_VOLUME/models/ltx"
LTX_DIT="$MODELS_DIR/ltx-2.3-22b-dev-fp8.safetensors"
LTX_TE="$MODELS_DIR/gemma_3_12B_it_fp8_e4m3fn.safetensors"
OUTPUT_DIR="$NETWORK_VOLUME/output_folder_musubi/ltx23/$OUTPUT_NAME"
DATASET_TOML="$OUTPUT_DIR/dataset.toml"
STATE_FILE="$REPO_DIR/training_state.tmp"

export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
export PYTORCH_ALLOC_CONF="expandable_segments:True"

if [ ! -d "$OUTPUT_DIR" ]; then
    print_error "Directory $OUTPUT_DIR does not exist. Check your OUTPUT_NAME."
    exit 1
fi

########################################
# STAGE 3: INTERACTIVE CHECKPOINT SELECTION
########################################
print_header "STAGE 3: DETECT AND SELECT CHECKPOINT"

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
    read -rp "Select checkpoint index to resume: " IDX
    if [[ "$IDX" =~ ^[0-9]+$ ]] && [ "$IDX" -lt "${#CHECKPOINTS[@]}" ]; then
        RESUME_CHECKPOINT="${CHECKPOINTS[$IDX]}"
    else
        print_error "Invalid selection."
        exit 1
    fi
fi

########################################
# STAGE 4: EPOCH EXTENSION & SNAPSHOT PROTECTION
########################################
print_header "STAGE 4: PROTECTION & METRIC MATH MANAGEMENT"

# Extract completed epochs from target directory name
FOLDER_NAME=$(basename "$RESUME_CHECKPOINT")
LAST_VAL=$(echo "$FOLDER_NAME" | grep -o "[0-9]\+" | tail -n 1 || echo "0")
LAST_VAL=$((10#$LAST_VAL))

echo -e "${PURPLE}================================================================${NC}"
echo -e "${BOLD}${CYAN} RESUME CONTROL CENTER ${NC}"
echo -e "${PURPLE}================================================================${NC}"
echo -e "${YELLOW}Resumed Milestone: $FOLDER_NAME (Completed Epochs: $LAST_VAL)${NC}"

REMAINING=$((MAX_TRAIN_EPOCHS - LAST_VAL))
[ $REMAINING -lt 0 ] && REMAINING=0
DEFAULT_EP=$((REMAINING * 2))
[ $DEFAULT_EP -eq 0 ] && DEFAULT_EP=2

echo -e "${GREEN}Detected $LAST_VAL epochs already completed. Room remaining to baseline cap: $REMAINING.${NC}"
read -rp "Enter number of ADDITIONAL epochs to run for this resume block (default $DEFAULT_EP): " USER_EP

ADDITIONAL_EP=${USER_EP:-$DEFAULT_EP}
ACTIVE_MAX_EPOCHS=$((LAST_VAL + ADDITIONAL_EP))

# --- SNAPSHOT OVERWRITE PROTECTION ---
print_info "Checking for overlapping snapshot weight files to preserve..."
PROTECT_COUNT=0
for ((e = 1; e <= ADDITIONAL_EP; e++)); do
    TARGET_EPOCH=$((LAST_VAL + e))
    PADDED_EPOCH=$(printf "%06d" "$TARGET_EPOCH")
    FILE_TO_PROTECT="$OUTPUT_DIR/${OUTPUT_NAME}-${PADDED_EPOCH}.safetensors"

    if [ -f "$FILE_TO_PROTECT" ]; then
        mv "$FILE_TO_PROTECT" "${FILE_TO_PROTECT%.safetensors}_pre_resume.safetensors"
        ((PROTECT_COUNT++))
    fi
done
[ "$PROTECT_COUNT" -gt 0 ] && print_success "Preserved $PROTECT_COUNT historical snapshots with '_pre_resume' extension safely."

print_info "Training trajectory updated. Script will run for $ADDITIONAL_EP more epochs (Target Cap: $ACTIVE_MAX_EPOCHS)."

# --- RUNTIME MATH RESTORE ---
if [ -f "$STATE_FILE" ]; then
    source "$STATE_FILE"
    print_success "Successfully imported training session global math states."
else
    print_error "State file missing at $STATE_FILE! Consistent global tracking math lost."
    exit 1
fi

# --- SCHEDULER RESET PROTECTION ---
echo -e "\n${YELLOW}Scheduler Reset Imminent. Original Baseline Config LR was: $LEARNING_RATE${NC}"
read -rp "Enter a stabilization 'tail' LR (e.g., 5e-5) or press ENTER to use baseline: " RESCUE_LR

if [ -n "$RESCUE_LR" ]; then
    ACTIVE_LR="$RESCUE_LR"
    ACTIVE_SCHEDULER="constant"
    ACTIVE_WARMUP=0
    print_success "Forcing ${BOLD}constant${NC} scheduler at stabilization rate: ${BOLD}$ACTIVE_LR${NC}"
else
    ACTIVE_LR="$LEARNING_RATE"
    ACTIVE_SCHEDULER="$LR_SCHEDULER"
    ACTIVE_WARMUP=0
    print_warning "Proceeding with original scheduler configuration sequence."
fi
echo -e "---------------------------------------\n"

# Extra flags
RESET_OPTIMIZER="${RESET_OPTIMIZER:-0}"
RESET_OPTIMIZER_PARAMS="${RESET_OPTIMIZER_PARAMS:-0}"
RESET_DATALOADER="${RESET_DATALOADER:-0}"

########################################
# STAGE 5: RESUME FLAGS PROCESSING
########################################
print_header "STAGE 5: RESUME FLAGS PROCESSING"

COMMON_FLAGS=(
    --ltx2_checkpoint "$LTX_DIT"
    --ltx_version "${LTX_VERSION:-2.3}"
    --ltx2_mode "${LTX_MODE:-video}"
    --gemma_safetensors "$LTX_TE"
    --dataset_config "$DATASET_TOML"
    --output_dir "$OUTPUT_DIR"
    --output_name "$OUTPUT_NAME"
    --network_module networks.lora_ltx2
    --network_dim "${LORA_RANK:-32}"
    --network_alpha "${LORA_ALPHA:-32}"
    --learning_rate "$ACTIVE_LR"
    --optimizer_type "$OPTIMIZER_TYPE"
    --lr_warmup_steps "$ACTIVE_WARMUP"
    --lr_scheduler "$ACTIVE_SCHEDULER"
    --lr_scheduler_power "${LR_SCHEDULER_POWER:-1.0}"
    --max_train_epochs "$ACTIVE_MAX_EPOCHS"
    --save_every_n_epochs "$SAVE_EVERY_N_EPOCHS"
    --gradient_accumulation_steps "$GRAD_ACCUM_STEPS"
    --accumulation_group_remainder "$ACCUMULATION_GROUP_REMAINDER"
    --caption_dropout_rate "${CAPTION_DROPOUT_RATE:-0}"
    --network_dropout "$NETWORK_DROPOUT"
    --resume "$RESUME_CHECKPOINT"
    --save_state
    --fp8_base
    --weighting_scheme none
    --timestep_sampling "$TIMESTEP_SAMPLING"
    --seed 42
    --attention_mode "${ATTN:-flash}"
)

[ "$RESET_OPTIMIZER" -eq 1 ] && COMMON_FLAGS+=("--reset_optimizer") && print_warning "Active: --reset_optimizer"
[ "$RESET_OPTIMIZER_PARAMS" -eq 1 ] && COMMON_FLAGS+=("--reset_optimizer_params") && print_warning "Active: --reset_optimizer_params"
[ "$RESET_DATALOADER" -eq 1 ] && COMMON_FLAGS+=("--reset_dataloader") && print_warning "Active: --reset_dataloader"

if [ "$OPTIMIZER_TYPE" == "adafactor" ]; then COMMON_FLAGS+=("--max_grad_norm" "0"); fi
if [ "$TIMESTEP_SAMPLING" != "shifted_logit_normal" ]; then COMMON_FLAGS+=("--discrete_flow_shift" "$DISCRETE_FLOW_SHIFT"); fi
if [ "${USE_EMA:-0}" = "1" ]; then COMMON_FLAGS+=("--save_every_n_steps" "$DYNAMIC_SAVE_STEPS"); fi
if [ -n "$BLOCKS_TO_SWAP" ]; then COMMON_FLAGS+=("--blocks_to_swap" "$BLOCKS_TO_SWAP"); fi
if [ "${GRADIENT_CHECKPOINTING:-1}" = "1" ]; then COMMON_FLAGS+=("--gradient_checkpointing"); fi

if [ "${ATTN:-flash}" = "flash" ]; then
    COMMON_FLAGS+=(--flash_attn --mixed_precision bf16)
else
    COMMON_FLAGS+=(--sdpa --mixed_precision bf16)
fi

########################################
# STAGE 6: TRAINING EXECUTION
########################################
print_header "STAGE 6: RESUME TRAINING EXECUTION"

# --- PRE-FLIGHT DISK RECLAIM CLEANUP ---
print_warning "Reclaiming disk space in $OUTPUT_DIR (Preserving target: $(basename "$RESUME_CHECKPOINT"))..."
find "$OUTPUT_DIR" -maxdepth 1 -type d -name "*-state" ! -path "$RESUME_CHECKPOINT" -exec rm -rf {} + > /dev/null 2>&1

cd "$REPO_DIR" || exit 1

accelerate launch \
    --num_processes "$GPU_COUNT" \
    --num_cpu_threads_per_process "$NUM_CPU_THREADS_PER_PROCESS" \
    --mixed_precision bf16 \
    "$REPO_DIR/ltx2_train_network.py" \
    --logging_dir "$OUTPUT_DIR/logs" \
    --log_with tensorboard \
    "${COMMON_FLAGS[@]}"

if [ $? -ne 0 ]; then
    print_error "Resumed LTX training session crashed."
    exit 1
fi

print_success "Training loop concluded successfully."

########################################
# STAGE 7: POST-RESUME VERIFIED EXPORTS
########################################
print_header "STAGE 7: COMFYUI CONVERSION EXPORTS"
CONVERT_SCRIPT="$REPO_DIR/convert_lora.py"

if [ -f "$CONVERT_SCRIPT" ]; then
    CONVERT_COUNT=0
    shopt -s nullglob
    for lora in "$OUTPUT_DIR"/*.safetensors; do
        [[ "$lora" == *"_comfy.safetensors" ]] && continue
        [[ "$lora" == *"model_states"* ]] && continue
        [[ "$lora" == *"-step"* ]] && continue

        COMFY_PATH="${lora%.safetensors}_comfy.safetensors"

        DO_CONVERT=0
        if [ ! -f "$COMFY_PATH" ]; then
            DO_CONVERT=1
        elif [ "$lora" -nt "$COMFY_PATH" ]; then
            print_warning "Detected updated source: $(basename "$lora"). Re-converting..."
            DO_CONVERT=1
        elif ! python3 -c "from safetensors import safe_open; f = safe_open('$COMFY_PATH', framework='pt'); f.metadata(); f.keys()" > /dev/null 2>&1; then
            print_warning "Corrupted conversion wrapper detected: $(basename "$COMFY_PATH"). Re-building..."
            DO_CONVERT=1
        fi

        if [ "$DO_CONVERT" -eq 1 ]; then
            [ -f "$COMFY_PATH" ] && rm -f "$COMFY_PATH"
            print_status "Converting: $(basename "$lora")"

            if python3 "$CONVERT_SCRIPT" --input "$lora" --output "$COMFY_PATH" --target other > /dev/null 2>&1; then
                if python3 -c "from safetensors import safe_open; f = safe_open('$COMFY_PATH', framework='pt'); f.metadata(); f.keys()" > /dev/null 2>&1; then
                    print_success "Converted & Verified: $(basename "$COMFY_PATH")"
                    ((CONVERT_COUNT++))
                else
                    print_error "CRITICAL: $(basename "$COMFY_PATH") headers remain unreadable after script run."
                    rm -f "$COMFY_PATH"
                fi
            else
                print_error "Failed pipeline compilation task on $(basename "$lora")"
            fi
        fi
    done
    shopt -u nullglob
    [ "$CONVERT_COUNT" -eq 0 ] && print_success "All ComfyUI files up to date." || print_success "Done ($CONVERT_COUNT outputs built/refreshed)."
else
    print_error "Conversion engine target not discovered at $CONVERT_SCRIPT"
fi

print_header "RESUME SEQUENCE COMPLETE"
