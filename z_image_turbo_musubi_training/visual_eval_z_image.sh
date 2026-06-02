#!/bin/bash

# --- COLORS & UI ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
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
export PYTORCH_ALLOC_CONF="expandable_segments:True"

# --- 3. CONFIG-AWARE PARAMETER PREP ---
# Clean up RESOLUTION_LIST from config
CLEAN_RES=$(echo $RESOLUTION_LIST | tr -d '",')
IMAGE_SIZE_H=$(echo $CLEAN_RES | awk '{print $1}')
IMAGE_SIZE_W=$(echo $CLEAN_RES | awk '{print $2}')

echo -e "\n${CYAN}⚙️ Resolution Settings:${NC}"
echo -e "Current Config Default: ${BOLD}$IMAGE_SIZE_H x $IMAGE_SIZE_W${NC}"
read -p "Apply custom resolution? [y/N]: " USE_CUSTOM

if [[ "$USE_CUSTOM" =~ ^[Yy]$ ]]; then
    read -p "Enter resolution (e.g., 1024): " CUSTOM_VAL
    if [[ "$CUSTOM_VAL" =~ ^[0-9]+$ ]]; then
        IMAGE_SIZE_H=$CUSTOM_VAL
        IMAGE_SIZE_W=$CUSTOM_VAL
        echo -e "${GREEN}✅ Resolution set to ${IMAGE_SIZE_H}x${IMAGE_SIZE_W}${NC}"
    else
        echo -e "${RED}⚠️ Invalid input. Falling back to config default.${NC}"
    fi
fi

# Lora Multiplier
echo -e "\n${CYAN}⚖️ LoRA Multiplier Settings:${NC}"
read -p "Enter LoRA multiplier or press ENTER for default (e.g. 1.5 default: 1.0): " LORA_MULT_INPUT

# Use 1.0 if the input is empty
LORA_MULTIPLIER=${LORA_MULT_INPUT:-1.0}

if [[ ! "$LORA_MULTIPLIER" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
    echo -e "${RED}⚠️ Invalid number. Falling back to 1.0${NC}"
    LORA_MULTIPLIER="1.0"
fi
echo -e "${GREEN}✅ Multiplier set to:${NC} ${BOLD}$LORA_MULTIPLIER${NC}"

# Dynamic Flags
INFER_FLAGS="--fp8_llm"
echo -e "${BLUE}ℹ️ Using default: FP8_LLM${NC}"

if [ "${FP8_BASE:-0}" -eq 1 ]; then
    INFER_FLAGS="$INFER_FLAGS --fp8_base"
    echo -e "${BLUE}ℹ️ Imported from config: FP8_BASE${NC}"
fi
if [ "${FP8_SCALED:-0}" -eq 1 ]; then
    INFER_FLAGS="$INFER_FLAGS --fp8_scaled"
    echo -e "${BLUE}ℹ️ Imported from config: FP8_SCALED${NC}"
fi

if [ -n "$BLOCKS_TO_SWAP" ]; then
    INFER_FLAGS="$INFER_FLAGS --blocks_to_swap $BLOCKS_TO_SWAP --sample_with_offloading"
    echo -e "${BLUE}ℹ️ Offloading Enabled: Swapping $BLOCKS_TO_SWAP blocks to CPU RAM${NC}"
fi

# Attention Mode
ATTN_MODE="torch"
#if python3 -c "import flash_attn" &> /dev/null; then
#    ATTN_MODE="flash"
#fi

# --- DYNAMIC LORA SELECTION ---
echo -e "\n${BLUE}🔍 Scanning for LoRA checkpoints in:${NC} $OUTPUT_DIR"

shopt -s nullglob
ALL_LORAS=("$OUTPUT_DIR"/*.safetensors)
shopt -u nullglob

# Sort files so latest epochs/steps appear at the bottom
IFS=$'\n' ALL_LORAS=($(sort <<< "${ALL_LORAS[*]}"))
unset IFS

AVAILABLE_LORAS=()
for lora in "${ALL_LORAS[@]}"; do
    [[ "$lora" == *"_comfy"* ]] && continue
    [[ "$lora" == *"model_states"* ]] && continue
    [[ "$lora" == *"_ema_s"* ]] && continue
    AVAILABLE_LORAS+=("$lora")
done

if [ ${#AVAILABLE_LORAS[@]} -eq 0 ]; then
    print_error "No raw training checkpoints found in $OUTPUT_DIR"
    exit 1
else
    echo -e "${CYAN}Please select a checkpoint for inference:${NC}"
    for i in "${!AVAILABLE_LORAS[@]}"; do
        DISPLAY_IDX=$((i + 1))
        LORA_NAME=$(basename "${AVAILABLE_LORAS[$i]}")

        if [[ "$LORA_NAME" == *"_pre_resume"* ]]; then
            LABEL="${YELLOW}(Pre-Resume Archive)${NC}"
        elif [[ "$LORA_NAME" == *"-step"* ]]; then
            LABEL="${MAGENTA}(EMA Step)${NC}"
        elif [[ "$LORA_NAME" == "$OUTPUT_NAME.safetensors" ]]; then
            LABEL="${BOLD}${GREEN}(FINAL)${NC}"
        else
            LABEL="(Epoch Save)"
        fi

        printf "  [%2d] %-45s %b\n" "$DISPLAY_IDX" "$LORA_NAME" "$LABEL"
    done

    read -p "Selection (1-${#AVAILABLE_LORAS[@]}, Default ${#AVAILABLE_LORAS[@]}): " USER_CHOICE
    USER_CHOICE=${USER_CHOICE:-${#AVAILABLE_LORAS[@]}}

    if [[ "$USER_CHOICE" =~ ^[0-9]+$ ]] && [ "$USER_CHOICE" -ge 1 ] && [ "$USER_CHOICE" -le "${#AVAILABLE_LORAS[@]}" ]; then
        SELECTED_LORA="${AVAILABLE_LORAS[$((USER_CHOICE - 1))]}"
    else
        print_warning "Invalid selection. Defaulting to latest."
        SELECTED_LORA="${AVAILABLE_LORAS[-1]}"
    fi
fi

# --- SET DYNAMIC PATHS ---
LORA_PATH="$SELECTED_LORA"
LORA_FILENAME=$(basename "$LORA_PATH" .safetensors)
SAMPLES_DIR="$OUTPUT_DIR/eval_samples/$LORA_FILENAME"
echo -e "\n${GREEN}🎯 Using LoRA:${NC} ${BOLD}$(basename "$LORA_PATH")${NC}"
echo -e "${BLUE}📂 Saving samples to:${NC} $SAMPLES_DIR"
mkdir -p "$SAMPLES_DIR"
cd "$REPO_DIR" || exit

# --- INFERENCE PROFILE ---
clear
echo -e "${BLUE}${BOLD}======================================================"
echo -e "      Z-IMAGE TURBO AUTOMATED INFERENCE"
echo -e "======================================================"
echo -e "${YELLOW}📊 Inference Profile:${NC}"
echo -e "   > Resolution: ${BOLD}$IMAGE_SIZE_H x $IMAGE_SIZE_W${NC}"
echo -e "   > Rank/Alpha: ${BOLD}$LORA_RANK  / $LORA_ALPHA${NC}"
echo -e "   > Attention:  ${BOLD}$ATTN_MODE${NC}"
echo -e "   > Checkpoint: ${BOLD}$(basename "$LORA_PATH")${NC}"
echo -e "   > Multiplier: ${BOLD}$LORA_MULTIPLIER${NC}"
if [ -n "$BLOCKS_TO_SWAP" ]; then
    echo -e "   > VRAM Swap:  ${BOLD}${YELLOW}$BLOCKS_TO_SWAP Blocks Offloaded to CPU${NC}"
fi
echo -e "${BLUE}${BOLD}======================================================${NC}\n"

# --- DEFINE PROMPTS ---
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
    "$TRIGGER, extreme close-up beauty portrait, neutral expression, direct eye contact, ultra high-resolution skin texture, studio lighting, sharp focus on eyes|201"
    "$TRIGGER, close-up portrait with soft natural window light, minimal makeup, relaxed expression, realistic skin detail, shallow depth of field|202"
    "$TRIGGER, tight headshot with dramatic Rembrandt lighting, cinematic shadows, high contrast, detailed skin pores, professional photography|203"
    "$TRIGGER, close-up smiling portrait in golden hour sunlight, warm tones, natural glow on skin, shallow depth of field, lifestyle aesthetic|204"
    "$TRIGGER, extreme close-up of face with wet skin look, dewy makeup, specular highlights, beauty editorial style, ultra detailed texture|205"
    "$TRIGGER, close-up portrait in a dimly lit cocktail bar, neon reflections on skin, moody lighting, sharp focus on eyes, cinematic aesthetic|206"
    "$TRIGGER, close-up candid shot laughing, soft motion blur in background, natural lighting, realistic skin texture, spontaneous moment|207"
    "$TRIGGER, tight portrait with wind gently moving hair across face, outdoor natural light, high detail skin texture, editorial photography|208"
    "$TRIGGER, close-up portrait wearing sunglasses pushed slightly down, eyes visible, fashion editorial look, sharp facial detail|209"
    "$TRIGGER, extreme close-up side profile portrait, soft diffused lighting, clean skin texture, sharp jawline definition, studio quality|210"
    "$TRIGGER, close-up portrait with messy bun hairstyle, soft morning light, natural skin imperfections visible, cozy indoor aesthetic|211"
    "$TRIGGER, close-up beauty shot with bold makeup, glossy lips, high detail skin texture, studio flash lighting, magazine editorial|212"
    "$TRIGGER, tight close-up portrait under harsh midday sunlight, strong shadows, realistic skin response, high dynamic range|213"
    "$TRIGGER, close-up portrait in rain with wet hair and droplets on skin, cinematic lighting, ultra detailed facial texture|214"
    "$TRIGGER, extreme close-up with soft bokeh background lights, night city setting, sharp eyes, natural skin tones|215"
)

# --- EXECUTION ---
echo -e "${BLUE}${BOLD}>>> Starting Batch Inference...${NC}"

for item in "${PROMPTS[@]}"; do
    IFS="|" read -r TEXT SEED <<< "$item"

    TARGET_FILENAME="${LORA_FILENAME}_mult_${LORA_MULTIPLIER}_seed_${SEED}.png"
    FINAL_PATH="${SAMPLES_DIR}/${TARGET_FILENAME}"

    if [ -f "$FINAL_PATH" ]; then
        echo -e "${YELLOW}⏩ Skipping: $TARGET_FILENAME (Already exists)${NC}"
        continue
    fi

    echo -e "\n${CYAN}🎨 Generating: ${BOLD}$TEXT${NC} (Seed: $SEED)"

    # Get file count before Python execution
    shopt -s nullglob
    BEFORE_FILES=("$SAMPLES_DIR"/*.png)
    BEFORE_COUNT=${#BEFORE_FILES[@]}
    shopt -u nullglob

    python3 "$REPO_DIR/zimage_generate_image.py" \
        --dit "$ZIMAGE_MODEL" \
        --vae "$ZIMAGE_VAE" \
        --text_encoder "$ZIMAGE_TEXT_ENCODER" \
        --lora_weight "$LORA_PATH" \
        --lora_multiplier $LORA_MULTIPLIER \
        --prompt "$TEXT" \
        --seed "$SEED" \
        --save_path "$SAMPLES_DIR" \
        --image_size $IMAGE_SIZE_H $IMAGE_SIZE_W \
        --infer_steps 25 \
        --flow_shift 3.0 \
        --attn_mode "$ATTN_MODE" \
        $INFER_FLAGS

    # Check if a new file was created
    shopt -s nullglob
    AFTER_FILES=("$SAMPLES_DIR"/*.png)
    AFTER_COUNT=${#AFTER_FILES[@]}
    shopt -u nullglob

    if [ "$AFTER_COUNT" -gt "$BEFORE_COUNT" ]; then
        # Safely grab the newest file (silencing errors)
        LATEST_FILE=$(ls -t "$SAMPLES_DIR"/*.png 2> /dev/null | head -1)
        if [ -n "$LATEST_FILE" ] && [ "$(basename "$LATEST_FILE")" != "$TARGET_FILENAME" ]; then
            mv "$LATEST_FILE" "$FINAL_PATH"
            echo -e "${GREEN}💾 Saved as: $TARGET_FILENAME${NC}"
        fi
    else
        print_error "Python script failed to generate an image for seed $SEED."
    fi
done

echo -e "\n${GREEN}${BOLD}✅ ALL SAMPLES GENERATED IN:${NC} ${CYAN}$SAMPLES_DIR${NC}"
