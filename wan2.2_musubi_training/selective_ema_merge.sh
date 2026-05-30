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

# 1. Load Config
CONFIG_FILE="${CONFIG_FILE:-wan_musubi_config.sh}"
if [ -f "$CONFIG_FILE" ]; then
    source "$CONFIG_FILE"
    print_success "Loaded Wan config: ${BOLD}$CONFIG_FILE${NC}"
else
    print_error "Config file $CONFIG_FILE not found!"
    exit 1
fi

REPO_DIR="$NETWORK_VOLUME/musubi-tuner"

# 2. Wan Task Selection
echo -e "\n${BOLD}${PURPLE}--- TASK SELECTION ---${NC}"
echo "1) Text-to-Video (t2v-A14B)"
echo "2) Image-to-Video (i2v-A14B)"
read -rp "Selection (1/2, default 1): " TASK_CHOICE
TASK_CHOICE=${TASK_CHOICE:-1}
if [ "$TASK_CHOICE" = "2" ]; then
    WAN_TASK="i2v-A14B"
else
    WAN_TASK="t2v-A14B"
fi

# 3. Flow Selection (Determines the primary reference table)
echo -e "\n${BOLD}${PURPLE}--- WAN 2.2 FLOW SELECTION ---${NC}"
echo "Note: Both flows will be merged automatically. This selection just sets the reference table."
echo "1) HIGH Noise Flow (875-1000)"
echo "2) LOW Noise Flow  (0-875)"
read -rp "Which flow are you exploring? (1/2, default 1): " FLOW_CHOICE
FLOW_CHOICE=${FLOW_CHOICE:-1}

# Set the processing order based on choice
if [ "$FLOW_CHOICE" = "2" ]; then
    PROCESSING_ORDER=("LOW" "HIGH")
    REF_TITLE="${TITLE_LOW:-Wan2.2_lora_low}"
    REF_LABEL="LOW"
else
    PROCESSING_ORDER=("HIGH" "LOW")
    REF_TITLE="${TITLE_HIGH:-Wan2.2_lora_high}"
    REF_LABEL="HIGH"
fi

# 4. Re-calculate Training Math
if [ "${DATASET_TYPE:-video}" = "video" ]; then
    FILE_COUNT=$(find "$DATASET_DIR" -maxdepth 1 -type f \
        \( -iname "*.mp4" -o -iname "*.webm" -o -iname "*.mov" \) | wc -l)
else
    FILE_COUNT=$(find "$DATASET_DIR" -maxdepth 1 -type f \
        \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.webp" \) | wc -l)
fi

SAMPLES_PER_EPOCH=$((FILE_COUNT * NUM_REPEATS))
EFFECTIVE_BATCH=$((BATCH_SIZE * GRAD_ACCUM_STEPS))
STEPS_PER_EPOCH_FLOAT=$(awk "BEGIN {printf \"%.2f\", $SAMPLES_PER_EPOCH / $EFFECTIVE_BATCH}")
STEPS_PER_EPOCH_INT=$(((SAMPLES_PER_EPOCH + EFFECTIVE_BATCH - 1) / EFFECTIVE_BATCH))
CALCULATED_TOTAL_STEPS=$((STEPS_PER_EPOCH_INT * MAX_TRAIN_EPOCHS))

echo -e "\n${BOLD}--- TRAINING STATS (Reference: $REF_LABEL) ---${NC}"
print_info "Dataset: ${BOLD}$FILE_COUNT${NC} | Repeats: ${BOLD}$NUM_REPEATS${NC} | Effective Batch: ${BOLD}$EFFECTIVE_BATCH${NC}"
print_info "Musubi Mapping: ${BOLD}$STEPS_PER_EPOCH_INT${NC} steps per Epoch."
print_info "Total Training: ${BOLD}$CALCULATED_TOTAL_STEPS${NC} steps."
echo "------------------------------------------------"

# 5. Build Reference Table from the chosen primary flow
REF_DIR="$NETWORK_VOLUME/output_folder_musubi/wan2.2/$WAN_TASK/$REF_TITLE"

shopt -s nullglob
ALL_REF=("$REF_DIR"/*.safetensors)
shopt -u nullglob

# Count pre_resume files to calculate extension epoch offset
PRE_RESUME_COUNT=$(printf '%s\n' "${ALL_REF[@]}" | grep -c "_pre_resume\.safetensors$" || true)

declare -A STEP_TO_FILE
AVAILABLE_STEPS=()

for f in "${ALL_REF[@]}"; do
    [[ "$f" == *"_comfy"* ]] && continue
    [[ "$f" == *"model_states"* ]] && continue
    [[ "$f" == *"_ema_"* ]] && continue

    base=$(basename "$f" .safetensors)

    # 1. Check for explicit "-stepXXXX" format first
    if [[ "$base" =~ -step([0-9]+) ]]; then
        step=$((10#${BASH_REMATCH[1]}))
        STEP_TO_FILE[$step]="$f"
        AVAILABLE_STEPS+=("$step")

    # 2. Check for "_pre_resume" epoch format
    elif [[ "$base" =~ _pre_resume$ ]]; then
        base="${base%_pre_resume}"
        [[ "$base" =~ -([0-9]+)$ ]] && {
            epoch=$((10#${BASH_REMATCH[1]}))
            step=$((epoch * STEPS_PER_EPOCH_INT))
            STEP_TO_FILE[$step]="$f"
            AVAILABLE_STEPS+=("$step")
        }

    # 3. Check for standard numeric suffix (treated as Epoch)
    elif [[ "$base" =~ -([0-9]+)$ ]]; then
        epoch=$((10#${BASH_REMATCH[1]} + PRE_RESUME_COUNT))
        step=$((epoch * STEPS_PER_EPOCH_INT))
        STEP_TO_FILE[$step]="$f"
        AVAILABLE_STEPS+=("$step")
    fi
done

if [ ${#AVAILABLE_STEPS[@]} -eq 0 ]; then
    print_error "No epoch snapshots found in reference directory: $REF_DIR"
    exit 1
fi

IFS=$'\n' AVAILABLE_STEPS=($(sort -n <<< "${AVAILABLE_STEPS[*]}"))
unset IFS

# Display the Map
printf "${BOLD}%-10s | %-12s | %-15s${NC}\n" "STEP" "EPOCH" "STATUS"
echo "------------------------------------------------"

for s in "${AVAILABLE_STEPS[@]}"; do
    CURRENT_EPOCH=$(awk "BEGIN {printf \"%.1f\", $s / $STEPS_PER_EPOCH_FLOAT}")
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

# 6. Interaction (Ask once, apply to both)
echo -e "${CYAN}Please specify the merge parameters for BOTH flows (skip leading zeros):${NC}"

DEFAULT_START_UI=$(echo "${AVAILABLE_STEPS[0]}" | sed 's/^0*//')
DEFAULT_END_UI=$(echo "${AVAILABLE_STEPS[-1]}" | sed 's/^0*//')

read -p "Enter START STEP (default $DEFAULT_START_UI): " USER_START_INPUT
USER_START_VAL=${USER_START_INPUT:-$DEFAULT_START_UI}

read -p "Enter END STEP (default $DEFAULT_END_UI): " USER_END_INPUT
USER_END_VAL=${USER_END_INPUT:-$DEFAULT_END_UI}

START_EPOCH=$(awk "BEGIN {printf \"%.0f\", $USER_START_VAL / $STEPS_PER_EPOCH_FLOAT}")
END_EPOCH=$(awk "BEGIN {printf \"%.0f\", $USER_END_VAL / $STEPS_PER_EPOCH_FLOAT}")
BETA_LABEL=$(echo "$USER_BETA" | tr -d '.')

# 7. AUTOMATED DUAL-MERGE LOOP
for CURRENT_FLOW in "${PROCESSING_ORDER[@]}"; do
    echo -e "\n${BOLD}${PURPLE}================================================${NC}"
    echo -e "${BOLD}${PURPLE}   CONFIGURING EMA FOR: ${CURRENT_FLOW} NOISE FLOW${NC}"
    echo -e "${BOLD}${PURPLE}================================================${NC}"

    # Inquire per-flow parameters
    read -p "Enter EMA Beta for $CURRENT_FLOW (default 0.99): " FLOW_BETA
    FLOW_BETA=${FLOW_BETA:-0.99}

    read -p "Enter sigma_rel for $CURRENT_FLOW (ENTER to skip Power EMA): " FLOW_SIGMA

    # Determine paths and labels
    if [ "$CURRENT_FLOW" = "HIGH" ]; then
        TARGET_TITLE="${TITLE_HIGH:-Wan2.2_lora_high}"
    else
        TARGET_TITLE="${TITLE_LOW:-Wan2.2_lora_low}"
    fi

    TARGET_DIR="$NETWORK_VOLUME/output_folder_musubi/wan2.2/$WAN_TASK/$TARGET_TITLE"
    FILE_LABEL="${TARGET_TITLE}_ema_s${USER_START_VAL}_to_s${USER_END_VAL}_e${START_EPOCH}to${END_EPOCH}_beta${BETA_LABEL}"

    FLOW_BETA_LABEL=$(echo "$FLOW_BETA" | tr -d '.')

    if [ -n "$FLOW_SIGMA" ]; then
        FLOW_SIGMA_LABEL=$(echo "$FLOW_SIGMA" | tr -d '.')
        FILE_LABEL="${TARGET_TITLE}_ema_s${USER_START_VAL}_to_s${USER_END_VAL}_e${START_EPOCH}to${END_EPOCH}_sigrel${FLOW_SIGMA_LABEL}"
    else
        FILE_LABEL="${TARGET_TITLE}_ema_s${USER_START_VAL}_to_s${USER_END_VAL}_e${START_EPOCH}to${END_EPOCH}_beta${FLOW_BETA_LABEL}"
    fi

    FINAL_OUT="$TARGET_DIR/${FILE_LABEL}.safetensors"

    # Gather matching files safely for this specific directory
    shopt -s nullglob
    ALL_DIR=("$TARGET_DIR"/*.safetensors)
    shopt -u nullglob

    TARGET_PRE_RESUME_COUNT=$(printf '%s\n' "${ALL_DIR[@]}" | grep -c "_pre_resume\.safetensors$" || true)

    EMA_FILES=()
    for f in "${ALL_DIR[@]}"; do
        [[ "$f" == *"_comfy"* ]] && continue
        [[ "$f" == *"model_states"* ]] && continue
        [[ "$f" == *"_ema_"* ]] && continue

        base=$(basename "$f" .safetensors)

        # Smart detection for the merge loop
        if [[ "$base" =~ -step([0-9]+) ]]; then
            step=$((10#${BASH_REMATCH[1]}))
            ((10#$step >= 10#$USER_START_VAL && 10#$step <= 10#$USER_END_VAL)) && EMA_FILES+=("$f::$step")

        elif [[ "$base" =~ _pre_resume$ ]]; then
            base="${base%_pre_resume}"
            [[ "$base" =~ -([0-9]+)$ ]] && {
                epoch=$((10#${BASH_REMATCH[1]}))
                step=$((epoch * STEPS_PER_EPOCH_INT))
                ((10#$step >= 10#$USER_START_VAL && 10#$step <= 10#$USER_END_VAL)) && EMA_FILES+=("$f::$step")
            }

        elif [[ "$base" =~ -([0-9]+)$ ]]; then
            epoch=$((10#${BASH_REMATCH[1]} + TARGET_PRE_RESUME_COUNT))
            step=$((epoch * STEPS_PER_EPOCH_INT))
            ((10#$step >= 10#$USER_START_VAL && 10#$step <= 10#$USER_END_VAL)) && EMA_FILES+=("$f::$step")
        fi
    done

    # Sort by step and strip the ::step suffix for the merge
    IFS=$'\n' EMA_FILES=($(sort -t: -k3 -n <<< "${EMA_FILES[*]}"))
    unset IFS
    EMA_FILES=("${EMA_FILES[@]%%::*}")

    if [ ${#EMA_FILES[@]} -lt 2 ]; then
        print_warning "Found ${#EMA_FILES[@]} files in $CURRENT_FLOW. Skipping merge (need at least 2)."
        continue
    fi

    # Merge execution
    echo -e "${YELLOW}[WAIT]${NC} Merging ${BOLD}${#EMA_FILES[@]}${NC} snapshots for $CURRENT_FLOW..."
    print_info "Range: Epoch ${BOLD}$START_EPOCH${NC} to ${BOLD}$END_EPOCH${NC}"

    SIGMA_FLAG=()
    [ -n "$FLOW_SIGMA" ] && SIGMA_FLAG=("--sigma_rel" "$FLOW_SIGMA")

    python3 "$REPO_DIR/lora_post_hoc_ema.py" \
        "${EMA_FILES[@]}" \
        --beta "$FLOW_BETA" \
        "${SIGMA_FLAG[@]}" \
        --output_file "$FINAL_OUT"

    if [ $? -ne 0 ] || [ ! -f "$FINAL_OUT" ]; then
        print_error "EMA merge failed. Aborting conversion."
        exit 1
    fi

    # ComfyUI Conversion for this iteration
    CONVERT_SCRIPT="$REPO_DIR/convert_lora.py"
    if [ -f "$CONVERT_SCRIPT" ]; then
        COMFY_OUT="${TARGET_DIR}/${FILE_LABEL}_comfy.safetensors"
        print_status "Converting $CURRENT_FLOW to ComfyUI..."
        python3 "$CONVERT_SCRIPT" --input "$FINAL_OUT" --output "$COMFY_OUT" --target other > /dev/null

        print_success "$CURRENT_FLOW Merge Complete!"
        echo -e "File: ${BOLD}$(basename "$COMFY_OUT")${NC}"
    else
        print_warning "convert_lora.py not found. Skipping ComfyUI conversion for $CURRENT_FLOW."
    fi
done

echo -e "\n${GREEN}[SUCCESS] All requested flows have been merged!${NC}\n"
