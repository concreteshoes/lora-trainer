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
ONETRAINER_CONFIG_DIR="$NETWORK_VOLUME/OneTrainer_config"

echo -e "\n${BLUE}🔍 Scanning for OneTrainer configs in:${NC} $ONETRAINER_CONFIG_DIR"

shopt -s nullglob
ALL_CONFIGS=("$ONETRAINER_CONFIG_DIR"/*.json)
shopt -u nullglob

# Filter out non-training configs (samples, concepts)
AVAILABLE_CONFIGS=()
for cfg in "${ALL_CONFIGS[@]}"; do
    BASENAME=$(basename "$cfg")
    if [[ "$BASENAME" != *"samples"* ]]; then
        AVAILABLE_CONFIGS+=("$cfg")
    fi
done

if [ ${#AVAILABLE_CONFIGS[@]} -eq 0 ]; then
    echo -e "${RED}❌ Error: No training config JSON files found in $ONETRAINER_CONFIG_DIR${NC}"
    exit 1
elif [ ${#AVAILABLE_CONFIGS[@]} -eq 1 ]; then
    SELECTED_CONFIG="${AVAILABLE_CONFIGS[0]}"
    echo -e "${GREEN}✅ Auto-selected:${NC} $(basename "$SELECTED_CONFIG")"
else
    echo -e "${CYAN}Multiple configs detected. Please select one:${NC}"
    for i in "${!AVAILABLE_CONFIGS[@]}"; do
        DISPLAY_IDX=$((i + 1))
        echo -e "  [$DISPLAY_IDX] $(basename "${AVAILABLE_CONFIGS[$i]}")"
    done
    read -p "Enter number (1-${#AVAILABLE_CONFIGS[@]}, Default 1): " USER_CHOICE
    USER_CHOICE=${USER_CHOICE:-1}
    if [[ "$USER_CHOICE" =~ ^[0-9]+$ ]] && [ "$USER_CHOICE" -ge 1 ] && [ "$USER_CHOICE" -le "${#AVAILABLE_CONFIGS[@]}" ]; then
        SELECTED_CONFIG="${AVAILABLE_CONFIGS[$((USER_CHOICE - 1))]}"
    else
        echo -e "${YELLOW}⚠️ Invalid selection. Defaulting to Choice 1.${NC}"
        SELECTED_CONFIG="${AVAILABLE_CONFIGS[0]}"
    fi
fi

# --- PARSE JSON VALUES ---
OUTPUT_NAME=$(python3 -c "
import json
d = json.load(open('$SELECTED_CONFIG'))
result = d.get('concepts', [{}])[0].get('name') or d.get('save_filename_prefix', '')
print(result)
")
LORA_RANK=$(python3 -c "import json; d=json.load(open('$SELECTED_CONFIG')); print(d.get('lora_rank','16'))")
LORA_ALPHA=$(python3 -c "import json; d=json.load(open('$SELECTED_CONFIG')); print(d.get('lora_alpha','16'))")

if [ -z "$OUTPUT_NAME" ]; then
    echo -e "${RED}❌ Error: Could not parse concept name from config${NC}"
    exit 1
fi

TRIGGER="$OUTPUT_NAME"
OUTPUT_DIR="$NETWORK_VOLUME/OneTrainer/output_folder_onetrainer/chroma/save"

echo -e "${GREEN}✅ Trigger:${NC} $TRIGGER"
echo -e "${GREEN}✅ Rank/Alpha:${NC} $LORA_RANK / $LORA_ALPHA"
echo -e "${GREEN}✅ Output dir:${NC} $OUTPUT_DIR"

# --- 2. PATHS & VARIABLES ---
HF_SNAPSHOT=$(ls -d "$HOME/.cache/huggingface/hub/models--lodestones--Chroma1-HD/snapshots"/*)
MODELS_DIR="$HF_SNAPSHOT"

export PYTORCH_ALLOC_CONF=expandable_segments:True

# --- 3. RESOLUTION ---
IMAGE_SIZE_W=$(python3 -c "import json; d=json.load(open('$SELECTED_CONFIG')); print(d.get('resolution','1024'))")
IMAGE_SIZE_H=$IMAGE_SIZE_W

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

# --- 4. LORA MULTIPLIER ---
echo -e "\n${CYAN}⚖️ LoRA Multiplier Settings:${NC}"
read -p "Enter LoRA multiplier or press ENTER for default (e.g. 1.5, default: 1.0): " LORA_MULT_INPUT
LORA_MULTIPLIER=${LORA_MULT_INPUT:-1.0}
if [[ ! "$LORA_MULTIPLIER" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
    echo -e "${RED}⚠️ Invalid number. Falling back to 1.0${NC}"
    LORA_MULTIPLIER="1.0"
fi
echo -e "${GREEN}✅ Multiplier set to:${NC} ${BOLD}$LORA_MULTIPLIER${NC}"

# --- 5. FP8 FLAGS ---
# Note: FP8 T5 is not supported — T5 layer norm mixes fp8/float32 which
# PyTorch does not allow. T5 always runs in bf16.
FP_FLAG=""

# --- 6. DYNAMIC LORA SELECTION ---
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
    echo -e "${RED}❌ Error: No LoRA checkpoints found in $OUTPUT_DIR${NC}"
    exit 1
elif [ ${#AVAILABLE_LORAS[@]} -eq 1 ]; then
    SELECTED_LORA="${AVAILABLE_LORAS[0]}"
    echo -e "${GREEN}✅ Auto-selected:${NC} $(basename "$SELECTED_LORA")"
else
    echo -e "${CYAN}Multiple LoRAs detected. Please select one:${NC}"
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
    if [[ "$USER_CHOICE" =~ ^[0-9]+$ ]] && [ "$USER_CHOICE" -ge 1 ] && [ "$USER_CHOICE" -le "${#AVAILABLE_LORAS[@]}" ]; then
        LORA_IDX=$((USER_CHOICE - 1))
        SELECTED_LORA="${AVAILABLE_LORAS[$LORA_IDX]}"
    else
        echo -e "${YELLOW}⚠️ Invalid selection. Defaulting to Choice 1.${NC}"
        SELECTED_LORA="${AVAILABLE_LORAS[0]}"
    fi
fi

LORA_PATH="$SELECTED_LORA"
LORA_FILENAME=$(basename "$LORA_PATH" .safetensors)
SAMPLES_DIR="$PWD/eval_samples/$LORA_FILENAME"
echo -e "\n${GREEN}🎯 Using LoRA:${NC} ${BOLD}$(basename "$LORA_PATH")${NC}"
echo -e "${BLUE}📂 Saving samples to:${NC} $SAMPLES_DIR"
mkdir -p "$SAMPLES_DIR"

# --- 7. INFERENCE PROFILE ---
echo -e "${BLUE}${BOLD}======================================================"
echo -e "      CHROMA1-HD AUTOMATED INFERENCE"
echo -e "======================================================"
echo -e "${YELLOW}📊 Inference Profile:${NC}"
echo -e "   > Resolution: ${BOLD}$IMAGE_SIZE_W x $IMAGE_SIZE_H${NC}"
echo -e "   > Rank/Alpha: ${BOLD}$LORA_RANK / $LORA_ALPHA${NC}"
echo -e "   > Attention:  ${BOLD}SDPA${NC}"
echo -e "   > Checkpoint: ${BOLD}$(basename "$LORA_PATH")${NC}"
echo -e "   > Multiplier: ${BOLD}$LORA_MULTIPLIER${NC}"
echo -e "${BLUE}${BOLD}======================================================${NC}"
echo -e "${YELLOW}ℹ️  Note: Chroma has no guidance scale — architecture hardcodes guidance to 0${NC}\n"

# --- 8. DEFINE PROMPTS ---
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

# --- 9. EXECUTION ---
echo -e "${BLUE}${BOLD}>>> Starting Batch Inference...${NC}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INFERENCE_SCRIPT="$SCRIPT_DIR/chroma_generate_image.py"

if [ ! -f "$INFERENCE_SCRIPT" ]; then
    echo -e "${RED}❌ Error: chroma_generate_image.py not found at $INFERENCE_SCRIPT${NC}"
    exit 1
fi

# Write all prompts to a temp file — pipeline loads once and loops internally
PROMPT_FILE=$(mktemp /tmp/chroma_prompts_XXXXXX.txt)
for item in "${PROMPTS[@]}"; do
    echo "$item" >> "$PROMPT_FILE"
done

echo -e "${CYAN}ℹ️  ${#PROMPTS[@]} prompts written. Pipeline loads once for all generations.${NC}"
echo ""

python3 "$INFERENCE_SCRIPT" \
    --model_path "$MODELS_DIR" \
    --lora_weight "$LORA_PATH" \
    --lora_multiplier $LORA_MULTIPLIER \
    --prompt_file "$PROMPT_FILE" \
    --save_path "$SAMPLES_DIR" \
    --output_prefix "$LORA_FILENAME" \
    --image_size $IMAGE_SIZE_H $IMAGE_SIZE_W \
    --infer_steps 30 \
    $FP_FLAG

rm -f "$PROMPT_FILE"

echo -e "\n${GREEN}${BOLD}✅ ALL SAMPLES GENERATED IN:${NC} ${CYAN}$SAMPLES_DIR${NC}"
