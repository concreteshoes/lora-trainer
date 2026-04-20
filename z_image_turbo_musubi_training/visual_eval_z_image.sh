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
CONFIG_FILE="${1:-z_image_musubi_config.sh}"
if [ -f "$CONFIG_FILE" ]; then
    source "$CONFIG_FILE"
    echo -e "${GREEN}✅ Config loaded:${NC} $CONFIG_FILE"
else
    echo -e "${RED}❌ Error: $CONFIG_FILE not found!${NC}"
    exit 1
fi

# --- 2. PATHS & VARIABLES ---
REPO_DIR="$NETWORK_VOLUME/musubi-tuner"
MODELS_DIR="$NETWORK_VOLUME/models/z_image"
ZIMAGE_MODEL="$MODELS_DIR/z_image_de_turbo_v1_bf16.safetensors"
ZIMAGE_VAE="$MODELS_DIR/ae.safetensors"
ZIMAGE_TEXT_ENCODER="$MODELS_DIR/qwen_3_4b.safetensors"

TRIGGER="$OUTPUT_NAME"

OUTPUT_DIR="$NETWORK_VOLUME/output_folder_musubi/z_image_turbo/$OUTPUT_NAME"

export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
export PYTORCH_ALLOC_CONF=expandable_segments:True

# --- 3. CONFIG-AWARE PARAMETER PREP ---
# 1. Clean up RESOLUTION_LIST from config
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

# 3. Precision Logic tied to Config
# We strictly keep fp8_llm and offload for the massive 3.4B text encoder to prevent OOM
FP_FLAG="--fp8_llm"

# 4. Conservative Attention Mode (Currently bugged with the inference script, torch needs to be enforced)
#ATTN_MODE="torch"
#if python3 -c "import flash_attn" &> /dev/null; then
#    ATTN_MODE="flash"
#    echo -e "${CYAN}⚡ Flash Attention detected.${NC}"
#fi

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
clear
echo -e "${BLUE}${BOLD}======================================================"
echo -e "      Z-IMAGE TURBO AUTOMATED INFERENCE"
echo -e "======================================================"
echo -e "${YELLOW}📊 Inference Profile:${NC}"
echo -e "   > Resolution: ${BOLD}$IMAGE_SIZE_W x $IMAGE_SIZE_H${NC}"
echo -e "   > Rank/Alpha: ${BOLD}$LORA_RANK  / $LORA_ALPHA${NC}"
echo -e "   > Attention:  ${BOLD}$ATTN_MODE${NC}"
echo -e "   > Checkpoint: ${BOLD}$SELECTED_LORA${NC}"
echo -e "${BLUE}${BOLD}======================================================${NC}\n"

# --- 7. DEFINE PROMPTS ---
declare -a PROMPTS=(
    "$TRIGGER, a beauty portrait close-up, high-resolution skin texture, natural skin pores, soft professional studio lighting, looking at camera|101"
    "$TRIGGER, street style photography, OOTD, walking in a fashionable outfit on a busy New York street, bokeh background|102"
    "$TRIGGER, elegant evening wear, red carpet event aesthetic, luxury hotel lobby, cinematic lighting|103"
    "$TRIGGER, lifestyle blogger aesthetic, smiling warmly in a sun-drenched flower garden, golden hour|104"
    "$TRIGGER, candid moment, holding a latte in a minimalist designer cafe, soft natural window light|105"
    "$TRIGGER, fit-check aesthetic, wearing stylish athleisure, yoga studio setting, bright and airy|106"
    "$TRIGGER, luxury travel aesthetic, sitting in a first-class airplane seat with a glass of champagne|107"
    "$TRIGGER, relaxed morning routine, wearing silk pajamas, sitting on a plush white bed, soft morning sun|108"
    "$TRIGGER, boss-chic aesthetic, wearing a sharp tailored designer blazer in a modern glass office|109"
    "$TRIGGER, moody artistic profile, gazing out a rain-streaked window in a high-rise apartment, cool tones|110"
    "$TRIGGER, mirror selfie, stylish bathroom interior, phone held up, designer accessories visible|111"
    "$TRIGGER, full body shot, standing in a luxury marble hallway, wearing a form-fitting cocktail dress, head to toe visible, sharp focus|112"
    "$TRIGGER, beach body aesthetic, standing on a white sand beach, wearing a minimalist black bikini, ocean waves in background, golden hour lighting, full body shot|113"
    "$TRIGGER, relaxed sunbathing, sitting on a striped beach towel on the sand, wearing a simple white bikini, leaning back on hands, high-detail skin texture and natural anatomy|114"
    "$TRIGGER, coastal walk, walking along the shoreline, looking back over shoulder, wearing a sheer silk sarong and bikini top, sunset backlighting, realistic proportions|115"
    "$TRIGGER, poolside elegance, standing at the edge of a turquoise infinity pool, wearing a high-cut athletic one-piece swimsuit, afternoon sun, sharp focus on silhouette|116"
    "$TRIGGER, lifeguard tower pose, leaning against a weathered wooden tower on a tropical beach, wearing a sheer white summer cover-up over a bikini, morning light, realistic height reference|117"
    "$TRIGGER, fitness aesthetic, mid-workout in a high-end gym, wearing a sports bra and tight athletic shorts, high muscle definition and skin pores, natural sweat sheen, athletic proportions|118"
    "$TRIGGER, tropical vacation vibe, sitting on a sun-drenched wooden pier, legs dangling, wearing a colorful string bikini, crystal clear water below, 8k resolution|119"
    "$TRIGGER, resort wear aesthetic, walking through a lush palm garden, wearing a thin translucent linen shirt unbuttoned over a bikini, dappled sunlight, high-fidelity body frame|120"
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

    # 2. Run the generator.
    # We point save_path to the directory.
    python3 "$REPO_DIR/zimage_generate_image.py" \
        --dit "$ZIMAGE_MODEL" \
        --vae "$ZIMAGE_VAE" \
        --text_encoder "$ZIMAGE_TEXT_ENCODER" \
        --lora_weight "$LORA_PATH" \
        --lora_multiplier 1.0 \
        --prompt "$TEXT" \
        --seed "$SEED" \
        --save_path "$SAMPLES_DIR" \
        --image_size $IMAGE_SIZE_W $IMAGE_SIZE_H \
        --infer_steps 25 \
        --guidance_scale 0.0 \
        --flow_shift 3.0 \
        --attn_mode "torch" \
        $FP_FLAG

    LATEST_FILE=$(ls -t "$SAMPLES_DIR"/*.png | head -1)
    if [ -n "$LATEST_FILE" ] && [ "$(basename "$LATEST_FILE")" != "$TARGET_FILENAME" ]; then
        mv "$LATEST_FILE" "$FINAL_PATH"
        echo -e "${GREEN}💾 Saved as: $TARGET_FILENAME${NC}"
    fi
done

echo -e "\n${GREEN}${BOLD}✅ ALL SAMPLES GENERATED IN:${NC} ${CYAN}$SAMPLES_DIR${NC}"
