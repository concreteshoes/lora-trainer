#!/bin/bash

# --- COLORS & UI ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

########################################
# Utility functions
########################################
print_info() { echo -e "${BLUE}[INFO]${NC} $*"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $*"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $*"; }
print_error() { echo -e "${RED}[ERROR]${NC} $*"; }
print_status() { echo -e "${CYAN}[STATUS]${NC} $*"; }

# =================================================================
# INTERACTIVE EMA RESCUE & EXPLORER
# Accurate Epoch-to-Step mapping based on Musubi dynamic logic.
# =================================================================

# 1. Load Config
CONFIG_FILE="${CONFIG_FILE:-z_image_musubi_config.sh}"
if [ -f "$CONFIG_FILE" ]; then
    source "$CONFIG_FILE"
    print_success "Loaded config: ${BOLD}$CONFIG_FILE${NC}"
else
    print_error "Config file $CONFIG_FILE not found!"
    exit 1
fi

REPO_DIR="$NETWORK_VOLUME/musubi-tuner"

# 2. Re-calculate Training Math
IMG_COUNT=$(find "$DATASET_DIR" -maxdepth 1 -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.webp" \) | wc -l)
EFFECTIVE_BATCH=$((BATCH_SIZE * GRAD_ACCUM_STEPS))

# --- REFINED ACCURATE MATH ---
# Samples = Images * Repeats
SAMPLES_PER_EPOCH=$((IMG_COUNT * NUM_REPEATS))

# Steps per Epoch Float (Using awk for portability)
STEPS_PER_EPOCH_FLOAT=$(awk "BEGIN {printf \"%.2f\", $SAMPLES_PER_EPOCH / $EFFECTIVE_BATCH}")

# Steps per Epoch Int = ceil(Samples / Effective Batch)
STEPS_PER_EPOCH_INT=$(((SAMPLES_PER_EPOCH + EFFECTIVE_BATCH - 1) / EFFECTIVE_BATCH))

# Total steps matching Musubi's logic
CALCULATED_TOTAL_STEPS=$((STEPS_PER_EPOCH_INT * MAX_TRAIN_EPOCHS))

# 3. Setup Target Directory
TARGET_DIR="$NETWORK_VOLUME/output_folder_musubi/z_image/$OUTPUT_NAME"

echo -e "\n${BOLD}--- TRAINING STATS ---${NC}"
print_info "Images: ${BOLD}$IMG_COUNT${NC} | Repeats: ${BOLD}$NUM_REPEATS${NC} | Effective Batch: ${BOLD}$EFFECTIVE_BATCH${NC}"
print_info "Musubi Mapping: ${BOLD}$STEPS_PER_EPOCH_INT${NC} steps per Epoch."
print_info "Total Training: ${BOLD}$CALCULATED_TOTAL_STEPS${NC} steps."
echo "------------------------------------------------"

# 4. Scan for snapshots
shopt -s nullglob
FILES=("$TARGET_DIR"/*-step*.safetensors)
shopt -u nullglob

if [ ${#FILES[@]} -eq 0 ]; then
    print_error "No step snapshots found in $TARGET_DIR"
    exit 1
fi

# Sort steps numerically
AVAILABLE_STEPS=()
for f in "${FILES[@]}"; do
    if [[ $(basename "$f") =~ -step([0-9]+) ]]; then
        AVAILABLE_STEPS+=("${BASH_REMATCH[1]}")
    fi
done
IFS=$'\n' AVAILABLE_STEPS=($(sort -n <<< "${AVAILABLE_STEPS[*]}"))
unset IFS

# 5. Display the Map
printf "${BOLD}%-10s | %-12s | %-15s${NC}\n" "STEP" "EPOCH" "STATUS"
echo "------------------------------------------------"

for s in "${AVAILABLE_STEPS[@]}"; do
    # Calculate current epoch as float using awk
    CURRENT_EPOCH=$(awk "BEGIN {printf \"%.1f\", $s / $STEPS_PER_EPOCH_FLOAT}")

    # Visual cues for training stages (using awk for float comparisons)
    STATUS=""
    if awk "BEGIN {exit !($CURRENT_EPOCH < ($MAX_TRAIN_EPOCHS / 3.0))}"; then
        STATUS="\e[2m(Early/Learning)\e[0m"
    elif awk "BEGIN {exit !($CURRENT_EPOCH > ($MAX_TRAIN_EPOCHS * 0.8))}"; then
        STATUS="${YELLOW}(Late/Overcook?)${NC}"
    else
        STATUS="${GREEN}(Sweet Spot?)${NC}"
    fi

    printf "%-10s | %-12s | %b\n" "$s" "$CURRENT_EPOCH" "$STATUS"
done
echo "------------------------------------------------"

# 6. Interaction
echo -e "${CYAN}Please specify the merge parameters (skip leading zeros):${NC}"

# Clean strings for display
DEFAULT_START_UI=$(echo "${AVAILABLE_STEPS[0]}" | sed 's/^0*//')
DEFAULT_END_UI=$(echo "${AVAILABLE_STEPS[-1]}" | sed 's/^0*//')

read -p "Enter START STEP (default $DEFAULT_START_UI): " USER_START_INPUT
USER_START_VAL=${USER_START_INPUT:-$DEFAULT_START_UI}

read -p "Enter END STEP (default $DEFAULT_END_UI): " USER_END_INPUT
USER_END_VAL=${USER_END_INPUT:-$DEFAULT_END_UI}

read -p "Enter EMA Beta (default 0.99): " USER_BETA
USER_BETA=${USER_BETA:-0.99}

# Filter and calculate human-readable labels
EMA_FILES=()
for s in "${AVAILABLE_STEPS[@]}"; do
    # 10# force base-10 to prevent octal errors
    if ((10#$s >= 10#$USER_START_VAL && 10#$s <= 10#$USER_END_VAL)); then
        EMA_FILES+=("$TARGET_DIR/${OUTPUT_NAME}-step${s}.safetensors")
    fi
done

if [ ${#EMA_FILES[@]} -lt 2 ]; then
    print_error "Found ${#EMA_FILES[@]} files. Need at least 2 snapshots to merge."
    exit 1
fi

# 7. Generate Descriptive Filename
# Map steps back to Epoch for the label
START_EPOCH=$(awk "BEGIN {printf \"%.0f\", $USER_START_VAL / $STEPS_PER_EPOCH_FLOAT}")
END_EPOCH=$(awk "BEGIN {printf \"%.0f\", $USER_END_VAL / $STEPS_PER_EPOCH_FLOAT}")
BETA_LABEL=$(echo "$USER_BETA" | tr -d '.')

# Clean Filename: [Model]_ema_s[Step]_to_s[Step]_e[Epoch]_beta[Beta].safetensors
FILE_LABEL="${OUTPUT_NAME}_ema_s${USER_START_VAL}_to_s${USER_END_VAL}_e${START_EPOCH}to${END_EPOCH}_beta${BETA_LABEL}"
FINAL_OUT="$TARGET_DIR/${FILE_LABEL}.safetensors"

echo -e "\n${YELLOW}[WAIT]${NC} Merging ${BOLD}${#EMA_FILES[@]}${NC} snapshots..."
print_info "Range: Epoch ${BOLD}$START_EPOCH${NC} to ${BOLD}$END_EPOCH${NC}"

python3 - "${EMA_FILES[@]}" << PYTHON_EOF
import sys
import torch
from safetensors.torch import load_file, save_file

files = sys.argv[1:]
beta = float("$USER_BETA")
n = len(files)
merged = None
weight_sum = 0.0

for i, path in enumerate(files):
    weight = beta ** (n - i - 1)
    weight_sum += weight
    state = load_file(path)
    if merged is None:
        merged = {k: v.to(torch.float32) * weight for k, v in state.items()}
    else:
        for k, v in state.items():
            if k in merged:
                merged[k].add_(v.to(torch.float32), alpha=weight)

for k in merged:
    merged[k] /= weight_sum

save_file(merged, "$FINAL_OUT")
PYTHON_EOF

# 8. ComfyUI Conversion
CONVERT_SCRIPT="$REPO_DIR/convert_lora.py"
if [ -f "$CONVERT_SCRIPT" ]; then
    COMFY_OUT="${TARGET_DIR}/${FILE_LABEL}_comfy.safetensors"
    print_status "Converting to ComfyUI..."
    python3 "$CONVERT_SCRIPT" --input "$FINAL_OUT" --output "$COMFY_OUT" --target other > /dev/null

    echo -e "\n${GREEN}[SUCCESS]${NC} Merge Complete!"
    echo -e "File: ${BOLD}$(basename "$COMFY_OUT")${NC}"
fi
