#!/usr/bin/env bash

# --- COLORS & UI ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# --- 1. LOAD CONFIG & PATHS ---
CONFIG_FILE="${1:-z_image_musubi_config.sh}"
if [ -f "$CONFIG_FILE" ]; then
    source "$CONFIG_FILE"
    echo -e "${GREEN}✅ Config loaded:${NC} $CONFIG_FILE"
else
    echo -e "${RED}❌ Error: $CONFIG_FILE not found!${NC}"
    exit 1
fi

OUTPUT_DIR="$NETWORK_VOLUME/output_folder_musubi/z_image_turbo/$OUTPUT_NAME"
REPO_DIR="$NETWORK_VOLUME/musubi-tuner"
MODELS_DIR="$NETWORK_VOLUME/models/z_image"
ZIMAGE_MODEL="$MODELS_DIR/z_image_de_turbo_v1_bf16.safetensors"
ZIMAGE_VAE="$MODELS_DIR/ae.safetensors"
ZIMAGE_TEXT_ENCODER="$MODELS_DIR/qwen_3_4b.safetensors"

# We strictly keep fp8_llm and offload for the massive 3.4B text encoder to prevent OOM
FP_FLAG="--fp8_llm"
if [ "${FP8_BASE:-0}" -eq 1 ]; then FP_FLAG="$FP_FLAG --fp8"; fi
if [ "${FP8_SCALED:-0}" -eq 1 ]; then FP_FLAG="$FP_FLAG --fp8_scaled"; fi

export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
export PYTORCH_ALLOC_CONF=expandable_segments:True

# --- 2. Attention ---
ATTN_MODE="sdpa"
if python3 -c "import flash_attn" &> /dev/null; then
    ATTN_MODE="flash"
    echo -e "${CYAN}⚡ Flash Attention detected.${NC}"
fi

# --- 3. DEFINE FIXED EVAL PARAMETERS ---
# We use fixed seeds so we can see the IDENTITY evolve on the same composition
declare -a EVAL_PROMPTS=(
    "$OUTPUT_NAME, golden hour lighting, warm tones, shallow depth of field|101"
    "$OUTPUT_NAME, side profile, looking away from camera|102"
    "$OUTPUT_NAME, wearing a red sweater, sitting in a cafe, blurry background|103"
    "$OUTPUT_NAME, portrait photo, natural lighting, looking at camera, soft smile|104"
    "$OUTPUT_NAME, beach scene, bright daylight, wind in hair|105"
    "$OUTPUT_NAME, professional photoshoot, studio lighting, fashion pose|106"
    "$OUTPUT_NAME, sitting at a cafe, coffee cup, lifestyle shot|107"

)

# --- 4. DYNAMIC LORA SCANNING ---
echo -e "\n${BLUE}🔍 Scanning for LoRA checkpoints in:${NC} $OUTPUT_DIR"

shopt -s nullglob
ALL_LORAS=("$OUTPUT_DIR"/*.safetensors)
shopt -u nullglob

AVAILABLE_LORAS=()
for lora in "${ALL_LORAS[@]}"; do
    if [[ "$lora" != *"_comfy"* ]] && [[ "$lora" != *"model_states"* ]]; then
        AVAILABLE_LORAS+=("$lora")
    fi
done

if [ ${#AVAILABLE_LORAS[@]} -eq 0 ]; then
    echo -e "${RED}❌ Error: No checkpoints found!${NC}"
    exit 1
fi

# --- 5. DISPLAY LIST & GET RANGE ---
echo -e "${CYAN}Available Checkpoints:${NC}"
for i in "${!AVAILABLE_LORAS[@]}"; do
    LORA_NAME=$(basename "${AVAILABLE_LORAS[$i]}")
    # Highlight the final checkpoint
    if [[ "$LORA_NAME" == "$OUTPUT_NAME.safetensors" ]]; then
        echo -e "  [$((i + 1))] ${BOLD}$LORA_NAME (FINAL)${NC}"
    else
        echo -e "  [$((i + 1))] $LORA_NAME"
    fi
done

echo ""
read -p "Enter START index (1-${#AVAILABLE_LORAS[@]}): " START_IDX
read -p "Enter END index (1-${#AVAILABLE_LORAS[@]}): " END_IDX

# Convert to 0-based array indexing
START_I=$((START_IDX - 1))
END_I=$((END_IDX - 1))

# --- 6. THE LOOP ---
# The base folder where all images will eventually live
FINAL_DIR="$OUTPUT_DIR/eval_samples/mass_eval"
mkdir -p "$FINAL_DIR"

for ((i = START_I; i <= END_I; i++)); do
    LORA_PATH="${AVAILABLE_LORAS[$i]}"
    LORA_FILENAME=$(basename "$LORA_PATH" .safetensors)

    echo -e "\n${BOLD}${YELLOW}>>> Testing [$((i + 1))/${#AVAILABLE_LORAS[@]}]: $LORA_FILENAME${NC}"

    for item in "${EVAL_PROMPTS[@]}"; do
        IFS="|" read -r TEXT SEED <<< "$item"

        # 1. Create a temporary unique path for this specific run
        TMP_PATH="$FINAL_DIR/tmp_run"
        mkdir -p "$TMP_PATH"

        # 2. Run the inference pointing to the temp folder
        python3 "$REPO_DIR/zimage_generate_image.py" \
            --dit "$ZIMAGE_MODEL" \
            --vae "$ZIMAGE_VAE" \
            --text_encoder "$ZIMAGE_TEXT_ENCODER" \
            --prompt "$TEXT" \
            --image_size 1024 1024 \
            --infer_steps 25 \
            --flow_shift 3.0 \
            --guidance_scale 0.0 \
            --attn_mode "$ATTN_MODE" \
            --save_path "$TMP_PATH" \
            --seed "$SEED" \
            --lora_multiplier 1.0 \
            --lora_weight "$LORA_PATH" \
            $FP_FLAG

        mv "$TMP_PATH/"*.png "$FINAL_DIR/${LORA_FILENAME}_seed_${SEED}.png" 2> /dev/null
        rm -rf "$TMP_PATH"
    done
done

echo -e "\n${GREEN}${BOLD}✅ ALL SAMPLES GENERATED IN:${NC} ${CYAN}$OUTPUT_DIR/eval_samples/mass_eval/${NC}"
