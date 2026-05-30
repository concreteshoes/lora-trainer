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
ALL_FILES=("$TARGET_DIR"/*.safetensors)
shopt -u nullglob

declare -A STEP_TO_FILE
AVAILABLE_STEPS=()

print_status "Scanning for snapshots in $TARGET_DIR..."

for f in "${ALL_FILES[@]}"; do
    # Skip utilities and results
    [[ "$f" == *"_comfy"* ]] && continue
    [[ "$f" == *"model_states"* ]] && continue
    [[ "$f" == *"_ema_"* ]] && continue

    base=$(basename "$f" .safetensors)
    step=0

    # 1. Detect Step-based snapshots (e.g., model-step000500.safetensors)
    if [[ "$base" =~ -step([0-9]+) ]]; then
        step=$((10#${BASH_REMATCH[1]}))

    # 2. Detect Pre-Resume snapshots (e.g., model-000005_pre_resume.safetensors)
    elif [[ "$base" =~ _pre_resume$ ]]; then
        # Strip suffix and extract epoch number
        tmp_base="${base%_pre_resume}"
        if [[ "$tmp_base" =~ -([0-9]+)$ ]]; then
            epoch=$((10#${BASH_REMATCH[1]}))
            step=$((epoch * STEPS_PER_EPOCH_INT))
        fi

    # 3. Detect Standard Epoch snapshots (e.g., model-000005.safetensors)
    elif [[ "$base" =~ -([0-9]+)$ ]]; then
        epoch=$((10#${BASH_REMATCH[1]}))
        step=$((epoch * STEPS_PER_EPOCH_INT))
    fi

    # If we found a valid step, map it
    if [ "$step" -gt 0 ]; then
        # If multiple files exist for the same step, prioritize step-based over epoch-based
        if [ -z "${STEP_TO_FILE[$step]}" ] || [[ "$base" == *"-step"* ]]; then
            STEP_TO_FILE[$step]="$f"
            AVAILABLE_STEPS+=("$step")
        fi
    fi
done

if [ ${#AVAILABLE_STEPS[@]} -eq 0 ]; then
    print_error "No snapshots found! Checked for -step, epoch numbers, and _pre_resume files."
    exit 1
fi

# Sort steps numerically and remove duplicates
IFS=$'\n' AVAILABLE_STEPS=($(sort -nu <<< "${AVAILABLE_STEPS[*]}"))
unset IFS

# 5. Display the Unified Map
printf "${BOLD}%-10s | %-12s | %-20s${NC}\n" "STEP" "EPOCH" "SOURCE TYPE"
echo "------------------------------------------------------------"

for s in "${AVAILABLE_STEPS[@]}"; do
    CURRENT_EPOCH=$(awk "BEGIN {printf \"%.1f\", $s / $STEPS_PER_EPOCH_FLOAT}")
    FILE_PATH="${STEP_TO_FILE[$s]}"
    FILE_NAME=$(basename "$FILE_PATH")

    TYPE="Epoch"
    [[ "$FILE_NAME" == *"-step"* ]] && TYPE="Step/EMA"
    [[ "$FILE_NAME" == *"_pre_resume"* ]] && TYPE="${YELLOW}Pre-Resume${NC}"

    printf "%-10s | %-12s | %b\n" "$s" "$CURRENT_EPOCH" "$TYPE"
done
echo "------------------------------------------------------------"

# 6. Interaction & File Gathering
echo -e "${CYAN}Please specify the merge parameters (skip leading zeros):${NC}"

DEFAULT_START_UI=$(echo "${AVAILABLE_STEPS[0]}" | sed 's/^0*//')
DEFAULT_END_UI=$(echo "${AVAILABLE_STEPS[-1]}" | sed 's/^0*//')

read -p "Enter START STEP (default $DEFAULT_START_UI): " USER_START_INPUT
USER_START_VAL=${USER_START_INPUT:-$DEFAULT_START_UI}

read -p "Enter END STEP (default $DEFAULT_END_UI): " USER_END_INPUT
USER_END_VAL=${USER_END_INPUT:-$DEFAULT_END_UI}

read -p "Enter EMA Beta (default 0.99): " USER_BETA
USER_BETA=${USER_BETA:-0.99}

read -p "Enter sigma_rel for Power EMA (or ENTER to use constant beta): " USER_SIGMA_REL

# Gather files based on sorted step range
EMA_FILES=()
for s in "${AVAILABLE_STEPS[@]}"; do
    if ((10#$s >= 10#$USER_START_VAL && 10#$s <= 10#$USER_END_VAL)); then
        EMA_FILES+=("${STEP_TO_FILE[$s]}")
    fi
done

if [ ${#EMA_FILES[@]} -lt 2 ]; then
    print_error "Found ${#EMA_FILES[@]} files in range. Need at least 2 snapshots."
    exit 1
fi

# 7. Generate Descriptive Filename
# Map steps back to Epoch for the label
START_EPOCH=$(awk "BEGIN {printf \"%.0f\", $USER_START_VAL / $STEPS_PER_EPOCH_FLOAT}")
END_EPOCH=$(awk "BEGIN {printf \"%.0f\", $USER_END_VAL / $STEPS_PER_EPOCH_FLOAT}")
BETA_LABEL=$(echo "$USER_BETA" | tr -d '.')

# Clean Filename: [Model]_ema_s[Step]_to_s[Step]_e[Epoch]_beta[Beta].safetensors
FILE_LABEL="${OUTPUT_NAME}_ema_s${USER_START_VAL}_to_s${USER_END_VAL}_e${START_EPOCH}to${END_EPOCH}_beta${BETA_LABEL}"

if [ -n "$USER_SIGMA_REL" ]; then
    SIGMA_LABEL=$(echo "$USER_SIGMA_REL" | tr -d '.')
    FILE_LABEL="${OUTPUT_NAME}_ema_s${USER_START_VAL}_to_s${USER_END_VAL}_e${START_EPOCH}to${END_EPOCH}_sigrel${SIGMA_LABEL}"
else
    FILE_LABEL="${OUTPUT_NAME}_ema_s${USER_START_VAL}_to_s${USER_END_VAL}_e${START_EPOCH}to${END_EPOCH}_beta${BETA_LABEL}"
fi

FINAL_OUT="$TARGET_DIR/${FILE_LABEL}.safetensors"

echo -e "\n${YELLOW}[WAIT]${NC} Merging ${BOLD}${#EMA_FILES[@]}${NC} snapshots..."
print_info "Range: Epoch ${BOLD}$START_EPOCH${NC} to ${BOLD}$END_EPOCH${NC}"

SIGMA_FLAG=()
[ -n "$USER_SIGMA_REL" ] && SIGMA_FLAG=("--sigma_rel" "$USER_SIGMA_REL")

python3 "$REPO_DIR/lora_post_hoc_ema.py" \
    "${EMA_FILES[@]}" \
    --beta "$USER_BETA" \
    "${SIGMA_FLAG[@]}" \
    --output_file "$FINAL_OUT"

if [ $? -ne 0 ] || [ ! -f "$FINAL_OUT" ]; then
    print_error "EMA merge failed. Aborting conversion."
    exit 1
fi

# 8. ComfyUI Conversion
CONVERT_SCRIPT="$REPO_DIR/convert_lora.py"
if [ -f "$CONVERT_SCRIPT" ]; then
    COMFY_OUT="${TARGET_DIR}/${FILE_LABEL}_comfy.safetensors"
    print_status "Converting to ComfyUI..."
    python3 "$CONVERT_SCRIPT" --input "$FINAL_OUT" --output "$COMFY_OUT" --target other > /dev/null

    echo -e "\n${GREEN}[SUCCESS]${NC} Merge Complete!"
    echo -e "File: ${BOLD}$(basename "$COMFY_OUT")${NC}"
fi
