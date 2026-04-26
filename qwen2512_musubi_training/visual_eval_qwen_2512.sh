#!/bin/bash

# --- COLORS & UI ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# --- 1. LOAD CONFIGURATION ---
CONFIG_FILE="${CONFIG_FILE:-qwen_musubi_config.sh}"

if [ -f "$CONFIG_FILE" ]; then
    source "$CONFIG_FILE"
    echo -e "${GREEN}✅ Config loaded:${NC} $CONFIG_FILE"
else
    echo -e "${RED}❌ Error: $CONFIG_FILE not found!${NC}"
    exit 1
fi

# Ensure DATASET_DIR was loaded and exists
if [ -z "$DATASET_DIR" ] || [ ! -d "$DATASET_DIR" ]; then
    echo -e "${RED}❌ Error: DATASET_DIR is missing or invalid in your config!${NC}"
    exit 1
fi

OUTPUT_DIR="$NETWORK_VOLUME/output_folder_musubi/qwen_image/$OUTPUT_NAME"

# --- 2. PATHS ---
REPO_DIR="$NETWORK_VOLUME/musubi-tuner"
MODELS_DIR="$NETWORK_VOLUME/models/Qwen-Image-2512"
TRIGGER="$OUTPUT_NAME"

QWEN_DIT=$(find "$MODELS_DIR/transformer" -name "*00001-of-*.safetensors" | head -n 1)
QWEN_VAE="$MODELS_DIR/vae/diffusion_pytorch_model.safetensors"
QWEN_TEXT_ENCODER=$(find "$MODELS_DIR/text_encoder" -name "*00001-of-*.safetensors" | head -n 1)

export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
export PYTORCH_ALLOC_CONF=expandable_segments:True

# --- 3. ATTENTION MODE DETECTION ---
ATTN_MODE="sdpa"
if python3 -c "import flash_attn" &> /dev/null; then
    ATTN_MODE="flash"
    echo -e "${CYAN}⚡ Flash Attention detected.${NC}"
fi

# --- 4. PREPARING PARAMETERS ---
# Default from config
CLEAN_RES=$(echo $RESOLUTION_LIST | tr -d '",')
IMAGE_SIZE_W=$(echo $CLEAN_RES | awk '{print $1}')
IMAGE_SIZE_H=$(echo $CLEAN_RES | awk '{print $2}')

echo -e "\n${CYAN}⚙️ Resolution Settings:${NC}"
echo -e "Current Config Default: ${BOLD}$IMAGE_SIZE_W x $IMAGE_SIZE_H${NC}"
read -p "Apply custom resolution? [y/N]: " USE_CUSTOM

if [[ "$USE_CUSTOM" =~ ^[Yy]$ ]]; then
    read -p "Enter resolution (e.g., 1024): " CUSTOM_VAL
    if [[ "$CUSTOM_VAL" =~ ^[0-9]+$ ]]; then
        IMAGE_SIZE_W=$CUSTOM_VAL
        IMAGE_SIZE_H=$CUSTOM_VAL
        echo -e "${GREEN}✅ Resolution set to ${IMAGE_SIZE_W}x${IMAGE_SIZE_H}${NC}"
    else
        echo -e "${RED}⚠️ Invalid input. Falling back to config default.${NC}"
    fi
fi

# Assemble the Flags
INFER_FLAGS="--model_version original \
--image_size $IMAGE_SIZE_W $IMAGE_SIZE_H \
--infer_steps 25 \
--guidance_scale 4.0 \
--attn_mode $ATTN_MODE \
--flow_shift 2.0"

# --- 4. DYNAMIC LORA SELECTION ---
echo -e "\n${BLUE}🔍 Scanning for raw LoRA checkpoints in:${NC} $OUTPUT_DIR"

shopt -s nullglob
ALL_LORAS=("$OUTPUT_DIR"/*.safetensors)
shopt -u nullglob

AVAILABLE_LORAS=()
for lora in "${ALL_LORAS[@]}"; do
    # Skip converted ComfyUI versions and model state files
    if [[ "$lora" != *"_comfy"* ]] && [[ "$lora" != *"model_states"* ]]; then
        AVAILABLE_LORAS+=("$lora")
    fi
done

if [ ${#AVAILABLE_LORAS[@]} -eq 0 ]; then
    echo -e "${RED}❌ Error: No raw training checkpoints found in $OUTPUT_DIR${NC}"
    exit 1
elif [ ${#AVAILABLE_LORAS[@]} -eq 1 ]; then
    SELECTED_LORA="${AVAILABLE_LORAS[0]}"
    echo -e "${GREEN}✅ Auto-selected only LoRA found:${NC} $(basename "$SELECTED_LORA")"
else
    echo -e "${CYAN}Multiple LoRAs detected. Please select one for inference:${NC}"
    for i in "${!AVAILABLE_LORAS[@]}"; do
        DISPLAY_IDX=$((i + 1))
        LORA_NAME=$(basename "${AVAILABLE_LORAS[$i]}")

        if [[ "$LORA_NAME" == "$OUTPUT_NAME.safetensors" ]]; then
            echo -e "  [$DISPLAY_IDX] ${BOLD}$LORA_NAME (FINAL CHECKPOINT)${NC}"
        else
            echo -e "  [$DISPLAY_IDX] $LORA_NAME"
        fi
    done

    read -p "Enter number (1-${#AVAILABLE_LORAS[@]}, Default 1): " USER_CHOICE
    USER_CHOICE=${USER_CHOICE:-1}

    # Validate: Is it a number? Is it within the 1-N range?
    if [[ "$USER_CHOICE" =~ ^[0-9]+$ ]] && [ "$USER_CHOICE" -ge 1 ] && [ "$USER_CHOICE" -le "${#AVAILABLE_LORAS[@]}" ]; then
        LORA_IDX=$((USER_CHOICE - 1))
        SELECTED_LORA="${AVAILABLE_LORAS[$LORA_IDX]}"
    else
        echo -e "${YELLOW}⚠️ Invalid selection. Defaulting to Choice 1.${NC}"
        SELECTED_LORA="${AVAILABLE_LORAS[0]}"
    fi
fi

# --- 5. SET DYNAMIC PATHS ---
LORA_PATH="$SELECTED_LORA"
LORA_FILENAME=$(basename "$LORA_PATH" .safetensors)
SAMPLES_DIR="$OUTPUT_DIR/eval_samples/$LORA_FILENAME"
echo -e "\n${GREEN}🎯 Using LoRA:${NC} ${BOLD}$(basename "$LORA_PATH")${NC}"
echo -e "${BLUE}📂 Saving samples to:${NC} $SAMPLES_DIR"
mkdir -p "$SAMPLES_DIR"
cd "$REPO_DIR" || exit

# --- 6. INFERENCE PROFILE ---
echo -e "${BLUE}${BOLD}======================================================"
echo -e "      QWEN 2512 AUTOMATED INFERENCE"
echo -e "======================================================"
echo -e "${YELLOW}📊 Inference Profile:${NC}"
echo -e "   > Resolution: ${BOLD}$IMAGE_SIZE_W x $IMAGE_SIZE_H${NC}"
echo -e "   > Rank/Alpha: ${BOLD}$LORA_RANK  / $LORA_ALPHA${NC}"
echo -e "   > Attention:  ${BOLD}$ATTN_MODE${NC}"
echo -e "   > Checkpoint: ${BOLD}$SELECTED_LORA${NC}"
echo -e "${BLUE}${BOLD}======================================================${NC}\n"

# --- 7. DEFINE PROMPTS ---
declare -a PROMPTS=(
    "$TRIGGER, wearing soft professional studio makeup for a close-up beauty portrait, looking at the camera with high-resolution skin texture|101"
    "$TRIGGER, walking on a busy New York street wearing a fashionable outfit, street style photography, bokeh background|102"
    "$TRIGGER, standing in a luxury hotel lobby wearing elegant evening wear, red carpet event aesthetic, cinematic lighting|103"
    "$TRIGGER, smiling warmly in a sun-drenched flower garden, lifestyle blogger aesthetic, golden hour|104"
    "$TRIGGER, sitting in a minimalist designer cafe holding a latte, candid moment, soft natural window light|105"
    "$TRIGGER, posing in a bright and airy yoga studio wearing stylish athleisure, fit-check aesthetic|106"
    "$TRIGGER, sitting in a first-class airplane seat with a glass of champagne, luxury travel aesthetic|107"
    "$TRIGGER, sitting on a plush white bed wearing silk pajamas, relaxed morning routine, soft morning sun|108"
    "$TRIGGER, wearing a sharp tailored designer blazer in a modern glass office, boss-chic aesthetic|109"
    "$TRIGGER, gazing out a rain-streaked window in a high-rise apartment, moody artistic profile, cool tones|110"
    "$TRIGGER, standing outside a dimly lit cocktail bar at night, wearing a fitted satin dress, neon reflections on wet pavement|111"
    "$TRIGGER, standing in a luxury marble hallway wearing a form-fitting cocktail dress, full body shot, head to toe visible, sharp focus|112"
    "$TRIGGER, standing on a white sand beach wearing a minimalist black bikini, beach body aesthetic, ocean waves in background, golden hour lighting|113"
    "$TRIGGER, sunbathing on a striped beach towel, wearing a simple white bikini, leaning back on her hands, high-detail skin texture|114"
    "$TRIGGER, walking along the shoreline looking back over her shoulder, wearing a sheer silk sarong and bikini top, sunset backlighting|115"
    "$TRIGGER, standing at the edge of a turquoise infinity pool wearing a high-cut athletic one-piece swimsuit, afternoon sun, sharp focus|116"
    "$TRIGGER, leaning against a weathered wooden lifeguard tower, wearing a sheer white summer cover-up over a bikini, tropical beach morning light|117"
    "$TRIGGER, mid-workout in a high-end gym wearing a sports bra and tight athletic shorts, fitness aesthetic, natural sweat sheen, athletic proportions|118"
    "$TRIGGER, sitting on a sun-drenched wooden pier with legs dangling, wearing a colorful string bikini, crystal clear water, 8k resolution|119"
    "$TRIGGER, walking through a lush palm garden wearing a thin translucent linen shirt unbuttoned over a bikini, dappled sunlight, resort wear aesthetic|120"
)

# --- 8. EXECUTION ---
echo -e "${BLUE}${BOLD}>>> Starting Batch Inference...${NC}"

for item in "${PROMPTS[@]}"; do
    IFS="|" read -r TEXT SEED <<< "$item"

    # The filename we actually want
    TARGET_FILENAME="${LORA_FILENAME}_seed_${SEED}.png"
    FINAL_PATH="${SAMPLES_DIR}/${TARGET_FILENAME}"

    # 1. Check if the renamed file already exists (Resume capability)
    if [ -f "$FINAL_PATH" ]; then
        echo -e "${YELLOW}⏩ Skipping: $TARGET_FILENAME (Already exists)${NC}"
        continue
    fi

    echo -e "\n${CYAN}🎨 Generating: ${BOLD}$TEXT${NC} (Seed: $SEED)"

    # Execute Python Script
    python3 "$REPO_DIR/qwen_image_generate_image.py" \
        --dit "$QWEN_DIT" \
        --vae "$QWEN_VAE" \
        --text_encoder "$QWEN_TEXT_ENCODER" \
        --lora_weight "$LORA_PATH" \
        --lora_multiplier 1.0 \
        --prompt "$PROMPT" \
        --seed "$SEED" \
        --save_path "$SAMPLES_DIR"
    $INFER_FLAGS

    LATEST_FILE=$(ls -t "$SAMPLES_DIR"/*.png | head -1)
    if [ -n "$LATEST_FILE" ] && [ "$(basename "$LATEST_FILE")" != "$TARGET_FILENAME" ]; then
        mv "$LATEST_FILE" "$FINAL_PATH"
        echo -e "${GREEN}💾 Saved as: $TARGET_FILENAME${NC}"
    fi
done

echo -e "\n${GREEN}${BOLD}✅ ALL EDITS SAVED IN:${NC} ${CYAN}$SAMPLES_DIR${NC}"
