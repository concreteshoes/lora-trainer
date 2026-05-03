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

OUTPUT_DIR="$NETWORK_VOLUME/output_folder_musubi/qwen_edit2511/$OUTPUT_NAME"

# --- 2. PATHS ---
REPO_DIR="$NETWORK_VOLUME/musubi-tuner"
MODELS_DIR="$NETWORK_VOLUME/models/Qwen-Image"
TRIGGER="$OUTPUT_NAME"

QWEN_DIT="$MODELS_DIR/qwen_image_edit_2511_bf16.safetensors"
QWEN_VAE="$MODELS_DIR/qwen_image_vae.safetensors"
QWEN_TEXT_ENCODER="$MODELS_DIR/qwen_2.5_vl_7b.safetensors"

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

# Lora Multiplier
echo -e "\n${CYAN}⚖️ LoRA Multiplier Settings:${NC}"
read -p "Enter LoRA multiplier or press ENTER for default (e.g. 1.5 default: 1.0): " LORA_MULT_INPUT

# Use 1.0 if the input is empty
LORA_MULTIPLIER=${LORA_MULT_INPUT:-1.0}

# Simple regex check to ensure it's a number/float
if [[ ! "$LORA_MULTIPLIER" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
    echo -e "${RED}⚠️ Invalid number. Falling back to 1.0${NC}"
    LORA_MULTIPLIER="1.0"
fi

echo -e "${GREEN}✅ Multiplier set to:${NC} ${BOLD}$LORA_MULTIPLIER${NC}"

# Assemble the Flags
INFER_FLAGS="--model_version edit-2511 \
--image_size $IMAGE_SIZE_W $IMAGE_SIZE_H \
--infer_steps 25 \
--guidance_scale 4.0 \
--resize_control_to_official_size \
--attn_mode $ATTN_MODE"

# Dynamic Memory Optimization
if [ "${FP8_SCALED:-0}" -eq 1 ]; then
    INFER_FLAGS="$FP_FLAG --fp8_scaled"
    echo -e "${BLUE}ℹ️ Imported from config: FP8_SCALED${NC}"
fi

# --- 5. ASSEMBLE IMAGE POOL ---
echo -e "${BLUE}🔍 Scanning for reference images in:${NC} $DATASET_DIR"
shopt -s nullglob nocaseglob
IMAGE_POOL=("$DATASET_DIR"/*.{jpg,jpeg,png,webp})
shopt -u nullglob nocaseglob

if [ ${#IMAGE_POOL[@]} -eq 0 ]; then
    echo -e "${RED}❌ Error: No images found in $DATASET_DIR${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Found ${#IMAGE_POOL[@]} images to use as references.${NC}"

# --- 6. DYNAMIC LORA SELECTION ---
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

# --- 8. SET DYNAMIC PATHS ---
LORA_PATH="$SELECTED_LORA"
LORA_FILENAME=$(basename "$LORA_PATH" .safetensors)
SAMPLES_DIR="$OUTPUT_DIR/eval_samples/$LORA_FILENAME"
echo -e "\n${GREEN}🎯 Using LoRA:${NC} ${BOLD}$(basename "$LORA_PATH")${NC}"
echo -e "${BLUE}📂 Saving samples to:${NC} $SAMPLES_DIR"
mkdir -p "$SAMPLES_DIR"
cd "$REPO_DIR" || exit

# --- 9. INFERENCE PROFILE ---
echo -e "\n${BLUE}${BOLD}======================================================"
echo -e "      QWEN EDIT-2511 AUTOMATED EVAL (RANDOM REF)"
echo -e "======================================================"
echo -e "${NC}"
echo -e "${YELLOW}📊 Inference Profile:${NC}"
echo -e "   > Resolution: ${BOLD}$IMAGE_SIZE_W x $IMAGE_SIZE_H${NC}"
echo -e "   > Rank/Alpha: ${BOLD}$LORA_RANK / $LORA_ALPHA${NC}"
echo -e "   > Attention:  ${BOLD}$ATTN_MODE${NC}"
echo -e "   > Checkpoint: ${BOLD}$(basename "$LORA_PATH")${NC}"
echo -e "   > Multiplier: ${BOLD}$LORA_MULTIPLIER${NC}"
echo -e "------------------------------------------------------"

# --- 10. DEFINE MODIFIERS (Instead of full prompts) ---
# We will combine these with your actual DATASET captions!
# Format: "Edit Instruction|Seed"
declare -a MODIFIERS=(
    "reconstruct the original image|101"                           # Baseline: Does she look like the dataset?
    "change the outfit to a sleek black leather jacket|102"        # Outfit Swap (Fashion test)
    "change the outfit to a white summer sundress|103"             # Color/Style contrast test
    "change the background to a luxury marble penthouse|104"       # Lifestyle/High-end test
    "change the background to a sunny tropical beach|105"          # Travel Influencer test
    "change the background to a busy Parisian cafe|106"            # Urban/Authentic test
    "apply professional studio flash photography lighting|107"     # Lighting/Texture test
    "change the lighting to soft pink neon sunset|108"             # Aesthetic/Mood test
    "add a pair of stylish designer sunglasses|109"                # Accessory addition test
    "make it look like a grainy iPhone mirror selfie|110"          # Social Media authenticity test
    "change the hair color to a vibrant platinum blonde|111"       # Color Swap test (Does LoRA permit hair changes?)
    "change the outfit to a red satin evening gown|112"            # Material/Texture test (Satin sheen vs. LoRA skin)
    "add heavy rain with realistic droplets on skin and hair|113"  # Physics/Detail test (Water interaction)
    "change the background to a snow-covered mountain at dusk|114" # Temperature/Lighting test (Cool tones)
    "make it look like a high-grain 35mm cinematic film still|115" # Post-processing test (Film look)
    "add a delicate gold necklace and small hoop earrings|116"     # Fine Detail test (Small accessory placement)
    "change the environment to a futuristic cyberpunk street|117"  # Concept Stress test (Neon/Synthetic world)
    "change the hairstyle to a sleek high ponytail|118"            # Silhouette test (Can it change hair shape?)
    "change the outfit to a professional grey business suit|119"   # Structural test (Rigid clothing vs. soft LoRA)
    "transform the image into a black and white noir portrait|120" # Value/Contrast test (Identity without color)
)

# --- 11. CAPTION-AWARE EXECUTION ---
echo -e "\n${BLUE}${BOLD}>>> Starting Caption-Aware Edit Batch...${NC}"

for item in "${MODIFIERS[@]}"; do
    IFS="|" read -r EDIT SEED <<< "$item"

    # Define standardized naming
    TARGET_FILENAME="${LORA_FILENAME}_seed_${SEED}.png"
    FINAL_PATH="${SAMPLES_DIR}/${TARGET_FILENAME}"

    # 1. Check if the specific renamed file already exists (Resume capability)
    if [ -f "$FINAL_PATH" ]; then
        echo -e "${YELLOW}⏩ Skipping: $TARGET_FILENAME (Already exists)${NC}"
        continue
    fi

    # 2. Pick a random image from the pool
    RANDOM_INDEX=$((RANDOM % ${#IMAGE_POOL[@]}))
    CONTROL_PATH="${IMAGE_POOL[$RANDOM_INDEX]}"
    REF_NAME=$(basename "$CONTROL_PATH")

    # 3. Locate and read the corresponding caption file (.txt)
    CAPTION_FILE="${CONTROL_PATH%.*}.txt"
    if [ -f "$CAPTION_FILE" ]; then
        CAPTION=$(cat "$CAPTION_FILE" | tr -d '\r\n')
    else
        echo -e "${YELLOW}⚠️ No .txt found for $REF_NAME. Using fallback caption.${NC}"
        CAPTION="a photo of a person"
    fi

    # 4. Construct the Smart Prompt (Double-Trigger Safe)
    if [[ "${CAPTION,,}" == *"${TRIGGER,,}"* ]]; then
        BASE_PROMPT="${CAPTION}"
    else
        BASE_PROMPT="${TRIGGER}, ${CAPTION}"
    fi

    if [[ "$EDIT" == *"reconstruct"* ]]; then
        PROMPT="reconstruct this original image: ${BASE_PROMPT}"
        echo -e "\n${CYAN}🎨 Seed: $SEED - [RECONSTRUCTION]${NC}"
    else
        PROMPT="${BASE_PROMPT}. ${EDIT}"
        echo -e "\n${CYAN}🎨 Seed: $SEED - [TARGETED EDIT]${NC}"
    fi

    echo -e "${BOLD}Ref Image:${NC} $REF_NAME"
    echo -e "${BOLD}Smart Prompt:${NC} $PROMPT"

    # 5. Execute Python Script
    # We point --save_path to the directory to prevent nested .png/ folders
    python3 "$REPO_DIR/qwen_image_generate_image.py" \
        --dit "$QWEN_DIT" \
        --vae "$QWEN_VAE" \
        --text_encoder "$QWEN_TEXT_ENCODER" \
        --lora_weight "$LORA_PATH" \
        --lora_multiplier $LORA_MULTIPLIER \
        --control_image_path "$CONTROL_PATH" \
        --prompt "$PROMPT" \
        --seed "$SEED" \
        --save_path "$SAMPLES_DIR" \
        --output_type images \
        --negative_prompt " " \
        $INFER_FLAGS

    LATEST_FILE=$(ls -t "$SAMPLES_DIR"/*.png | head -1)
    if [ -n "$LATEST_FILE" ] && [ "$(basename "$LATEST_FILE")" != "$TARGET_FILENAME" ]; then
        mv "$LATEST_FILE" "$FINAL_PATH"
        echo -e "${GREEN}💾 Saved as: $TARGET_FILENAME${NC}"
    fi
done

echo -e "\n${GREEN}${BOLD}✅ ALL EDITS SAVED IN:${NC} ${CYAN}$SAMPLES_DIR${NC}"
