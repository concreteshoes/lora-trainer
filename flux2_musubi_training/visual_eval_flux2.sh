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
CONFIG_FILE="${CONFIG_FILE:-flux2_musubi_config.sh}"
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

# --- 2. PATHS & VARIABLES ---
REPO_DIR="$NETWORK_VOLUME/musubi-tuner"
MODELS_DIR="$NETWORK_VOLUME/models/flux2"
TRIGGER="$OUTPUT_NAME"
OUTPUT_DIR="$NETWORK_VOLUME/output_folder_musubi/flux2/$OUTPUT_NAME"

FLUX2_DIT="$MODELS_DIR/flux2-klein-base-9b.safetensors"
FLUX2_VAE="$MODELS_DIR/ae.safetensors"
FLUX2_TEXT_ENCODER=$(find "$MODELS_DIR/text_encoder" -name "*00001-of-*.safetensors" | head -n 1)

export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
export PYTORCH_ALLOC_CONF=expandable_segments:True

# --- 3. ATTENTION MODE DETECTION ---
ATTN_MODE="torch"
if python3 -c "import flash_attn" &> /dev/null; then
    ATTN_MODE="flash_attn"
    echo -e "${CYAN}⚡ Flash Attention detected.${NC}"
fi

# --- 4. CONFIG-AWARE PARAMETER PREP ---
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

# 2. Lora Multiplier
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

# 3. Precision Logic
FP_FLAG=""
if [ "${FP8_SCALED:-0}" -eq 1 ]; then FP_FLAG="$FP_FLAG --fp8_scaled"; fi
if [ "${FP8_TEXT_ENCODER:-0}" -eq 1 ]; then FP_FLAG="$FP_FLAG --fp8_text_encoder"; fi

# 4. Assemble the Flags dynamically
INFER_FLAGS="--model_version klein-base-9b \
--image_size $IMAGE_SIZE_W $IMAGE_SIZE_H \
--infer_steps 50 \
--embedded_cfg_scale 4.0 \
--attn_mode $ATTN_MODE \
--flow_shift 2.2 \
--output_type images \
$FP_FLAG"

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

# --- 7. SET DYNAMIC PATHS ---
LORA_PATH="$SELECTED_LORA"
LORA_FILENAME=$(basename "$LORA_PATH" .safetensors)
SAMPLES_DIR="$OUTPUT_DIR/eval_samples/$LORA_FILENAME"
echo -e "\n${GREEN}🎯 Using LoRA:${NC} ${BOLD}$(basename "$LORA_PATH")${NC}"
echo -e "${BLUE}📂 Saving samples to:${NC} $SAMPLES_DIR"
mkdir -p "$SAMPLES_DIR"
cd "$REPO_DIR" || exit

# --- 8. INFERENCE PROFILE ---
echo -e "${BLUE}${BOLD}======================================================"
echo -e "      FLUX.2-KLEIN AUTOMATED EVAL (RANDOM REF)"
echo -e "======================================================"
echo -e "${YELLOW}📊 Inference Profile:${NC}"
echo -e "   > Resolution: ${BOLD}$IMAGE_SIZE_W x $IMAGE_SIZE_H${NC}"
echo -e "   > Rank/Alpha: ${BOLD}$LORA_RANK  / $LORA_ALPHA${NC}"
echo -e "   > Attention:  ${BOLD}$ATTN_MODE${NC}"
echo -e "   > Checkpoint: ${BOLD}$SELECTED_LORA${NC}"
echo -e "${BLUE}${BOLD}======================================================${NC}\n"

# [INSERT YOUR MODIFIERS ARRAY HERE]
declare -a MODIFIERS=(

    # --- RECONSTRUCTION BASELINE ---
    "exact reconstruction|101"

    # --- CLOTHING (core composability test) ---
    "change the outfit to a sleek black leather jacket|102"
    "change the outfit to an elegant evening dress|103"
    "change the outfit to casual streetwear, hoodie and jeans|104"
    "change the outfit to athletic gym wear|106"

    # --- ENVIRONMENT ---
    "change the background to a sunny tropical beach|105"
    "change the background to a modern luxury apartment|107"
    "change the background to a busy city street|108"
    "change the background to a cozy cafe interior|109"

    # --- LIGHTING (VERY IMPORTANT) ---
    "change lighting to cinematic low light with soft shadows|110"
    "change lighting to golden hour warm sunlight|111"
    "change lighting to bright studio lighting|112"
    "change lighting to night scene with neon lights|113"

    # --- POSE / COMPOSITION ---
    "make it a full body shot, standing pose|114"
    "make it a close-up portrait, face focus|115"
    "make it a side profile shot|116"
    "make it an over-the-shoulder shot|117"

    # --- STYLE / INFLUENCER BEHAVIOR ---
    "turn this into a social media influencer selfie, holding phone|118"
    "turn this into a professional fashion photoshoot|119"
    "make it candid street photography style|120"

    # --- HAIR / ACCESSORIES ---
    "change hairstyle to tied up hair|121"
    "add sunglasses|122"

    # --- HARD STRESS TESTS ---
    "change the outfit to a formal business suit|123"
    "place the scene in a cyberpunk neon city at night|124"
    "apply dramatic high contrast editorial lighting|125"
)

# --- 9. EXECUTION LOOP ---
echo -e "\n${BLUE}${BOLD}>>> Starting Flux.2-Klein Evaluation...${NC}"

for item in "${MODIFIERS[@]}"; do
    IFS="|" read -r EDIT SEED <<< "$item"

    # Define the specific target filename to match previous script
    TARGET_FILENAME="${LORA_FILENAME}_seed_${SEED}.png"
    FINAL_PATH="${SAMPLES_DIR}/${TARGET_FILENAME}"

    # 1. Check if the renamed file already exists (Resume capability)
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
        CAPTION=$(xargs < "$CAPTION_FILE")
    else
        echo -e "${YELLOW}⚠️ No .txt found for $REF_NAME. Using fallback caption.${NC}"
        CAPTION="a professional photograph"
    fi

    # 4. Smart Prompt Construction
    if [[ "${CAPTION,,}" == *"${TRIGGER,,}"* ]]; then
        BASE_PROMPT="${CAPTION}"
    else
        BASE_PROMPT="${TRIGGER}, ${CAPTION}"
    fi

    # 5. Handle Reconstruction vs Edit
    if [[ "$EDIT" == *"reconstruction"* ]]; then
        PROMPT="reconstruct this original image: ${BASE_PROMPT}"
        echo -e "\n${CYAN}🎨 [RECONSTRUCTION] (Seed: $SEED)${NC}"
    else
        PROMPT="${BASE_PROMPT}. ${EDIT}"
        echo -e "\n${CYAN}🎨 [TARGETED EDIT] (Seed: $SEED)${NC}"
    fi

    echo -e "${BOLD}Ref Image:${NC} $REF_NAME"
    echo -e "${BOLD}Final Prompt:${NC} $PROMPT"

    # 6. Execute using the Musubi documentation syntax
    # Note: We point --save_path to the DIRECTORY to avoid folder-nesting bugs
    python3 "$REPO_DIR/flux_2_generate_image.py" \
        --dit "$FLUX2_DIT" \
        --vae "$FLUX2_VAE" \
        --text_encoder "$FLUX2_TEXT_ENCODER" \
        --lora_weight "$LORA_PATH" \
        --lora_multiplier $LORA_MULTIPLIER \
        --control_image_path "$CONTROL_PATH" \
        --prompt "$PROMPT" \
        --seed "$SEED" \
        --save_path "$SAMPLES_DIR" \
        $INFER_FLAGS

    LATEST_FILE=$(ls -t "$SAMPLES_DIR"/*.png | head -1)
    if [ -n "$LATEST_FILE" ] && [ "$(basename "$LATEST_FILE")" != "$TARGET_FILENAME" ]; then
        mv "$LATEST_FILE" "$FINAL_PATH"
        echo -e "${GREEN}💾 Saved as: $TARGET_FILENAME${NC}"
    fi
done

echo -e "\n${GREEN}${BOLD}✅ ALL FLUX EDITS SAVED IN:${NC} ${CYAN}$SAMPLES_DIR${NC}"
