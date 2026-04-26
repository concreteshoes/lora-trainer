#!/bin/bash
# --- COLORS & UI ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

ONETRAINER_DIR="$NETWORK_VOLUME/OneTrainer"
ONETRAINER_CONFIG_DIR="$NETWORK_VOLUME/OneTrainer_config"
OUTPUT_BASE="$ONETRAINER_DIR/output_folder_onetrainer"

# --- 1. LOAD CONFIG ---
echo -e "\n${BLUE}🔍 Scanning for OneTrainer configs in:${NC} $ONETRAINER_CONFIG_DIR"
shopt -s nullglob
ALL_CONFIGS=("$ONETRAINER_CONFIG_DIR"/*.json)
shopt -u nullglob

AVAILABLE_CONFIGS=()
for cfg in "${ALL_CONFIGS[@]}"; do
    BASENAME=$(basename "$cfg")
    if [[ "$BASENAME" != *"samples"* ]] && [[ "$BASENAME" != *"concepts"* ]]; then
        AVAILABLE_CONFIGS+=("$cfg")
    fi
done

if [ ${#AVAILABLE_CONFIGS[@]} -eq 0 ]; then
    echo -e "${RED}❌ No training configs found in $ONETRAINER_CONFIG_DIR${NC}"
    exit 1
elif [ ${#AVAILABLE_CONFIGS[@]} -eq 1 ]; then
    SELECTED_CONFIG="${AVAILABLE_CONFIGS[0]}"
    echo -e "${GREEN}✅ Auto-selected:${NC} $(basename "$SELECTED_CONFIG")"
else
    echo -e "${CYAN}Multiple configs detected. Please select one:${NC}"
    for i in "${!AVAILABLE_CONFIGS[@]}"; do
        echo -e "  [$((i+1))] $(basename "${AVAILABLE_CONFIGS[$i]}")"
    done
    read -p "Enter number (Default 1): " USER_CHOICE
    USER_CHOICE=${USER_CHOICE:-1}
    if [[ "$USER_CHOICE" =~ ^[0-9]+$ ]] && [ "$USER_CHOICE" -ge 1 ] && [ "$USER_CHOICE" -le "${#AVAILABLE_CONFIGS[@]}" ]; then
        SELECTED_CONFIG="${AVAILABLE_CONFIGS[$((USER_CHOICE-1))]}"
    else
        echo -e "${YELLOW}⚠️ Invalid selection. Defaulting to Choice 1.${NC}"
        SELECTED_CONFIG="${AVAILABLE_CONFIGS[0]}"
    fi
fi

echo -e "${GREEN}✅ Config loaded:${NC} $(basename "$SELECTED_CONFIG")"

OUTPUT_NAME=$(python3 -c "
import json
d = json.load(open('$SELECTED_CONFIG'))
result = d.get('concepts', [{}])[0].get('name') or d.get('save_filename_prefix', '')
print(result)
")

if [ -z "$OUTPUT_NAME" ]; then
    echo -e "${RED}❌ Could not determine output name from config${NC}"
    exit 1
fi

# --- 2. SELECT MODEL TYPE ---
declare -a MODEL_TYPES=(
    "STABLE_DIFFUSION_15"
    "STABLE_DIFFUSION_15_INPAINTING"
    "STABLE_DIFFUSION_20"
    "STABLE_DIFFUSION_20_BASE"
    "STABLE_DIFFUSION_20_INPAINTING"
    "STABLE_DIFFUSION_20_DEPTH"
    "STABLE_DIFFUSION_21"
    "STABLE_DIFFUSION_21_BASE"
    "STABLE_DIFFUSION_3"
    "STABLE_DIFFUSION_35"
    "STABLE_DIFFUSION_XL_10_BASE"
    "STABLE_DIFFUSION_XL_10_BASE_INPAINTING"
    "WUERSTCHEN_2"
    "STABLE_CASCADE_1"
    "PIXART_ALPHA"
    "PIXART_SIGMA"
    "FLUX_DEV_1"
    "FLUX_FILL_DEV_1"
    "FLUX_2"
    "SANA"
    "HUNYUAN_VIDEO"
    "HI_DREAM_FULL"
    "CHROMA_1"
    "QWEN"
    "Z_IMAGE"
)

# Try to auto-detect from config first
CONFIG_MODEL_TYPE=$(python3 -c "
import json
d = json.load(open('$SELECTED_CONFIG'))
print(d.get('model_type', ''))
")

if [ -n "$CONFIG_MODEL_TYPE" ]; then
    echo -e "\n${GREEN}✅ Model type detected from config:${NC} ${BOLD}$CONFIG_MODEL_TYPE${NC}"
    read -p "Use this model type? [Y/n]: " USE_DETECTED
    if [[ "$USE_DETECTED" =~ ^[Nn]$ ]]; then
        CONFIG_MODEL_TYPE=""
    fi
fi

if [ -z "$CONFIG_MODEL_TYPE" ]; then
    echo -e "\n${CYAN}Select model type:${NC}"
    for i in "${!MODEL_TYPES[@]}"; do
        echo -e "  [$((i+1))] ${MODEL_TYPES[$i]}"
    done
    read -p "Enter number: " MODEL_CHOICE
    if [[ "$MODEL_CHOICE" =~ ^[0-9]+$ ]] && [ "$MODEL_CHOICE" -ge 1 ] && [ "$MODEL_CHOICE" -le "${#MODEL_TYPES[@]}" ]; then
        CONFIG_MODEL_TYPE="${MODEL_TYPES[$((MODEL_CHOICE-1))]}"
    else
        echo -e "${RED}❌ Invalid selection${NC}"
        exit 1
    fi
fi

SELECTED_MODEL_TYPE="$CONFIG_MODEL_TYPE"
echo -e "${GREEN}✅ Model type:${NC} $SELECTED_MODEL_TYPE"

# --- 3. SCAN OUTPUT SUBDIRECTORIES ---
echo -e "\n${BLUE}🔍 Scanning output directories in:${NC} $OUTPUT_BASE"
shopt -s nullglob
ALL_SUBDIRS=("$OUTPUT_BASE"/*/save/)
shopt -u nullglob

AVAILABLE_DIRS=()
for d in "${ALL_SUBDIRS[@]}"; do
    # Check if it contains any non-comfy safetensors
    shopt -s nullglob
    CKPTS=("$d"*.safetensors)
    shopt -u nullglob
    HAS_UNCONVERTED=false
    for f in "${CKPTS[@]}"; do
        if [[ "$f" != *"_comfy"* ]] && [[ "$f" != *"model_states"* ]]; then
            HAS_UNCONVERTED=true
            break
        fi
    done
    if [ "$HAS_UNCONVERTED" = true ]; then
        AVAILABLE_DIRS+=("$d")
    fi
done

if [ ${#AVAILABLE_DIRS[@]} -eq 0 ]; then
    echo -e "${RED}❌ No output directories with unconverted checkpoints found under $OUTPUT_BASE${NC}"
    exit 1
elif [ ${#AVAILABLE_DIRS[@]} -eq 1 ]; then
    SAVE_DIR="${AVAILABLE_DIRS[0]}"
    echo -e "${GREEN}✅ Auto-selected directory:${NC} $SAVE_DIR"
else
    echo -e "${CYAN}Multiple output directories found. Select one:${NC}"
    for i in "${!AVAILABLE_DIRS[@]}"; do
        echo -e "  [$((i+1))] ${AVAILABLE_DIRS[$i]}"
    done
    read -p "Enter number (Default 1): " DIR_CHOICE
    DIR_CHOICE=${DIR_CHOICE:-1}
    if [[ "$DIR_CHOICE" =~ ^[0-9]+$ ]] && [ "$DIR_CHOICE" -ge 1 ] && [ "$DIR_CHOICE" -le "${#AVAILABLE_DIRS[@]}" ]; then
        SAVE_DIR="${AVAILABLE_DIRS[$((DIR_CHOICE-1))]}"
    else
        echo -e "${YELLOW}⚠️ Invalid selection. Defaulting to Choice 1.${NC}"
        SAVE_DIR="${AVAILABLE_DIRS[0]}"
    fi
fi

# --- 4. SCAN CHECKPOINTS ---
echo -e "\n${BLUE}🔍 Scanning checkpoints in:${NC} $SAVE_DIR"
shopt -s nullglob
ALL_SAFETENSORS=("$SAVE_DIR"*.safetensors)
shopt -u nullglob

AVAILABLE_CKPTS=()
for f in "${ALL_SAFETENSORS[@]}"; do
    if [[ "$f" != *"_comfy"* ]] && [[ "$f" != *"model_states"* ]]; then
        AVAILABLE_CKPTS+=("$f")
    fi
done

if [ ${#AVAILABLE_CKPTS[@]} -eq 0 ]; then
    echo -e "${RED}❌ No unconverted checkpoints found${NC}"
    exit 1
fi

echo -e "\n${CYAN}Available checkpoints:${NC}"
for i in "${!AVAILABLE_CKPTS[@]}"; do
    CKPT_NAME=$(basename "${AVAILABLE_CKPTS[$i]}")
    COMFY_VERSION="${SAVE_DIR}${CKPT_NAME%.safetensors}_comfy.safetensors"
    if [ -f "$COMFY_VERSION" ]; then
        echo -e "  [$((i+1))] $CKPT_NAME ${YELLOW}(already converted)${NC}"
    else
        echo -e "  [$((i+1))] $CKPT_NAME"
    fi
done

# --- 5. SELECTION MODE ---
echo -e "\n${CYAN}Selection mode:${NC}"
echo -e "  [1] Convert all"
echo -e "  [2] Range (e.g. 2-5)"
echo -e "  [3] Specific (e.g. 1 3 5)"
read -p "Choose (Default 1): " MODE
MODE=${MODE:-1}

TO_CONVERT=()

if [ "$MODE" == "1" ]; then
    TO_CONVERT=("${AVAILABLE_CKPTS[@]}")

elif [ "$MODE" == "2" ]; then
    read -p "Enter range (e.g. 2-5): " RANGE_INPUT
    START=$(echo "$RANGE_INPUT" | cut -d'-' -f1)
    END=$(echo "$RANGE_INPUT" | cut -d'-' -f2)
    for ((i=START-1; i<=END-1; i++)); do
        [ $i -lt ${#AVAILABLE_CKPTS[@]} ] && TO_CONVERT+=("${AVAILABLE_CKPTS[$i]}")
    done

elif [ "$MODE" == "3" ]; then
    read -p "Enter numbers separated by spaces (e.g. 1 3 5): " SPECIFIC_INPUT
    for idx in $SPECIFIC_INPUT; do
        REAL_IDX=$((idx-1))
        [ $REAL_IDX -lt ${#AVAILABLE_CKPTS[@]} ] && TO_CONVERT+=("${AVAILABLE_CKPTS[$REAL_IDX]}")
    done
fi

if [ ${#TO_CONVERT[@]} -eq 0 ]; then
    echo -e "${RED}❌ No checkpoints selected${NC}"
    exit 1
fi

# --- 6. OUTPUT DTYPE ---
echo -e "\n${CYAN}Output dtype:${NC}"
echo -e "  [1] BFLOAT_16 (recommended, default)"
echo -e "  [2] FLOAT_16"
echo -e "  [3] FLOAT_32"
read -p "Choose (Default 1): " DTYPE_CHOICE
DTYPE_CHOICE=${DTYPE_CHOICE:-1}
case "$DTYPE_CHOICE" in
    2) OUTPUT_DTYPE="FLOAT_16" ;;
    3) OUTPUT_DTYPE="FLOAT_32" ;;
    *) OUTPUT_DTYPE="BFLOAT_16" ;;
esac
echo -e "${GREEN}✅ Output dtype:${NC} $OUTPUT_DTYPE"

# --- 7. CONVERT ---
echo -e "\n${BLUE}${BOLD}======================================================"
echo -e "      ONETRAINER COMFYUI CONVERSION"
echo -e "======================================================"
echo -e "${YELLOW}📊 Conversion Profile:${NC}"
echo -e "   > Model Type:  ${BOLD}$SELECTED_MODEL_TYPE${NC}"
echo -e "   > Output Name: ${BOLD}$OUTPUT_NAME${NC}"
echo -e "   > Save Dir:    ${BOLD}$SAVE_DIR${NC}"
echo -e "   > Converting:  ${BOLD}${#TO_CONVERT[@]} checkpoint(s)${NC}"
echo -e "   > Output Dtype:${BOLD}$OUTPUT_DTYPE${NC}"
echo -e "${BLUE}${BOLD}======================================================${NC}\n"

cd "$ONETRAINER_DIR" || exit

CONVERTED=0
SKIPPED=0
FAILED=0

for INPUT in "${TO_CONVERT[@]}"; do
    BASENAME=$(basename "$INPUT" .safetensors)
    OUTPUT="${SAVE_DIR}${BASENAME}_comfy.safetensors"

    if [ -f "$OUTPUT" ]; then
        echo -e "${YELLOW}⏩ Skipping (already converted):${NC} $(basename "$OUTPUT")"
        ((SKIPPED++))
        continue
    fi

    echo -e "\n${CYAN}🔄 Converting:${NC} $(basename "$INPUT")"
    ./run-cmd.sh convert_model \
        --model-type "$SELECTED_MODEL_TYPE" \
        --training-method LORA \
        --input-name "$INPUT" \
        --output-dtype "$OUTPUT_DTYPE" \
        --output-model-format COMFY_LORA \
        --output-model-destination "$OUTPUT"

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Saved:${NC} $(basename "$OUTPUT")"
        ((CONVERTED++))
    else
        echo -e "${RED}❌ Failed:${NC} $(basename "$INPUT")"
        ((FAILED++))
    fi
done

echo -e "\n${GREEN}${BOLD}======================================================"
echo -e "✅ CONVERSION COMPLETE"
echo -e "======================================================${NC}"
echo -e "   > Converted: ${GREEN}${BOLD}$CONVERTED${NC}"
echo -e "   > Skipped:   ${YELLOW}${BOLD}$SKIPPED${NC}"
echo -e "   > Failed:    ${RED}${BOLD}$FAILED${NC}"
echo -e "${BLUE}${BOLD}======================================================${NC}"
