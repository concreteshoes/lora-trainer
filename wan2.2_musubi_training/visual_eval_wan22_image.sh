#!/usr/bin/env bash

# --- COLORS & UI ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

print_header() {
    echo -e "\n${BOLD}${PURPLE}================================================================${NC}"
    echo -e "${BOLD}${CYAN}  $1 ${NC}"
    echo -e "${BOLD}${PURPLE}================================================================${NC}"
}

print_warning() { echo -e "${YELLOW}$1${NC}"; }
print_status() { echo -e "${BLUE}[WAIT]${NC} $1"; }
print_success() { echo -e "${GREEN}[OK]  ${NC} $1"; }
print_error() { echo -e "${RED}[FAIL]${NC} $1"; }

# --- 1. LOAD CONFIGURATION ---
CONFIG_FILE="${1:-wan_musubi_config.sh}"
if [ -f "$CONFIG_FILE" ]; then
    source "$CONFIG_FILE"
    echo -e "${GREEN}✅ Config loaded:${NC} $CONFIG_FILE"
else
    echo -e "${RED}❌ Error: $CONFIG_FILE not found!${NC}"
    exit 1
fi

# --- 2. PATHS & DIRECTORIES ---
OUT_HIGH="${OUT_HIGH:-$NETWORK_VOLUME/output_folder_musubi/wan22/$TITLE_HIGH}"
OUT_LOW="${OUT_LOW:-$NETWORK_VOLUME/output_folder_musubi/wan22/$TITLE_LOW}"
DATASET_TYPE="${DATASET_TYPE:-image}"

REPO_DIR="$NETWORK_VOLUME/musubi-tuner"
MODELS_DIR="$NETWORK_VOLUME/models/Wan"

WAN_VAE="$MODELS_DIR/Wan2.1_VAE.pth"
WAN_T5="$MODELS_DIR/models_t5_umt5-xxl-enc-bf16.pth"

export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
export PYTORCH_ALLOC_CONF=expandable_segments:True

# --- EXPLICIT EXPERT PATHS ---
WAN_DIT_T2V_HIGH="$MODELS_DIR/wan2.2_t2v_high_noise_14B_fp16.safetensors"
WAN_DIT_T2V_LOW="$MODELS_DIR/wan2.2_t2v_low_noise_14B_fp16.safetensors"
WAN_DIT_I2V_HIGH="$MODELS_DIR/wan2.2_i2v_high_noise_14B_fp16.safetensors"
WAN_DIT_I2V_LOW="$MODELS_DIR/wan2.2_i2v_low_noise_14B_fp16.safetensors"

# --- 3. STAGE 1: TASK, MEDIA & TRIGGER SELECTION ---
print_header "STAGE 1: TASK, MEDIA & TRIGGER SELECTION"

echo -e "${CYAN}Enter the Trigger Word/Phrase for your LoRA:${NC}"
read -rp "Trigger (e.g., 'ohwx man'): " USER_TRIGGER
TRIGGER="${USER_TRIGGER:-Wan2.2_LoRA}"
echo -e "${GREEN}✅ Trigger set to:${NC} $TRIGGER"
echo ""

echo -e "${CYAN}Select Inference Task:${NC}"
echo "1) Text-to-Video (t2v-A14B)"
echo "2) Image-to-Video (i2v-A14B)"
read -rp "Selection (1/2, default 1): " TASK_PICK
TASK_PICK=${TASK_PICK:-1}
WAN_TASK=$([ "$TASK_PICK" == "2" ] && echo "i2v-A14B" || echo "t2v-A14B")

echo -e "\n${CYAN}Select Media Type:${NC}"
echo "1) Image Eval (1 Frame - High + Low Noise)"
echo "2) Video Eval (41 Frames - Standard Dual-Flow)"
read -rp "Selection (1/2, default 1): " MEDIA_PICK
MEDIA_PICK=${MEDIA_PICK:-1}

if [ "$MEDIA_PICK" == "2" ]; then
    GEN_LENGTH=41
    IS_VIDEO=true
    declare -a EVAL_LIST=(
        "walking forward confidently towards the camera|201"
        "turning head slowly to look at the camera, smiling|202"
        "standing still as the wind blows hair across her face|203"
    )
else
    GEN_LENGTH=1
    IS_VIDEO=false
    print_warning "Image Mode: Multi-noise LoRA matching enabled."

    declare -a EVAL_LIST=(
        "close-up beauty portrait, professional studio makeup, high-resolution skin texture|101"
        "walking on a New York street, fashionable outfit, street style, bokeh background|102"
        "luxury hotel lobby, elegant evening wear, red carpet aesthetic, cinematic lighting|103"
        "smiling in a sun-drenched flower garden, lifestyle blogger aesthetic, golden hour|104"
        "sitting in a minimalist cafe with a latte, candid, natural window light|105"
        "gazing out a rain-streaked window in a high-rise apartment, moody artistic profile, cool tones|110"
        "standing outside a dimly lit cocktail bar at night, wearing a fitted satin dress, neon reflections on wet pavement|111"
        "standing in a luxury marble hallway wearing a form-fitting cocktail dress, full body shot, head to toe visible, sharp focus|112"
        "standing on a white sand beach wearing a minimalist black bikini, beach body aesthetic, ocean waves in background, golden hour lighting|113"
        "sunbathing on a striped beach towel, wearing a simple white bikini, leaning back on her hands, high-detail skin texture|114"
        "walking along the shoreline looking back over her shoulder, wearing a sheer silk sarong and bikini top, sunset backlighting|115"
        "standing at the edge of a turquoise infinity pool wearing a high-cut athletic one-piece swimsuit, afternoon sun, sharp focus|116"
        "leaning against a weathered wooden lifeguard tower, wearing a sheer white summer cover-up over a bikini, tropical beach morning light|117"
        "mid-workout in a high-end gym wearing a sports bra and tight athletic shorts, fitness aesthetic, natural sweat sheen, athletic proportions|118"
        "close-up portrait wearing sunglasses pushed slightly down, eyes visible, fashion editorial look, sharp facial detail|209"
        "extreme close-up side profile portrait, soft diffused lighting, clean skin texture, sharp jawline definition, studio quality|210"
        "close-up portrait with messy bun hairstyle, soft morning light, natural skin imperfections visible, cozy indoor aesthetic|211"
        "close-up beauty shot with bold makeup, glossy lips, high detail skin texture, studio flash lighting, magazine editorial|212"
    )
fi

# Assign MoE Experts based on Task
if [ "$WAN_TASK" == "i2v-A14B" ]; then
    WAN_DIT="$WAN_DIT_I2V_LOW"
    WAN_DIT_HIGH="$WAN_DIT_I2V_HIGH"
else
    WAN_DIT="$WAN_DIT_T2V_LOW"
    WAN_DIT_HIGH="$WAN_DIT_T2V_HIGH"
fi

# --- 4. CONFIG-AWARE PARAMETER PREP ---
CLEAN_RES=$(echo $RESOLUTION_LIST | tr -d '",')
IMAGE_SIZE_H=$(echo $CLEAN_RES | awk '{print $1}')
IMAGE_SIZE_W=$(echo $CLEAN_RES | awk '{print $2}')

if [ "$IS_VIDEO" = true ]; then
    IMAGE_SIZE_H=832
    IMAGE_SIZE_W=480
    echo -e "\n${YELLOW}⚠️ Video Mode Active: Using safe eval resolution (832x480).${NC}"
else
    echo -e "\n${CYAN}⚙️ Resolution Settings (Image Mode):${NC}"
    echo -e "Current Config Default: ${BOLD}$IMAGE_SIZE_H x $IMAGE_SIZE_W${NC}"
fi

read -p "Enter LoRA multiplier (Default 1.0): " LORA_MULT_INPUT
LORA_MULTIPLIER=${LORA_MULT_INPUT:-1.0}
SAFE_MULT=$(echo "$LORA_MULTIPLIER" | tr '.' '-')

FP_FLAG="--fp8_t5"
if [[ "$FP_FLAG" == *"--fp8_t5"* ]]; then echo -e "${BLUE}ℹ️ Using: FP8_T5${NC}"; fi
if [ "${FP8_BASE:-0}" -eq 1 ]; then
    FP_FLAG="$FP_FLAG --fp8"
    echo -e "${BLUE}ℹ️ Imported from config: FP8_BASE${NC}"
fi
if [ "${FP8_SCALED:-0}" -eq 1 ]; then
    FP_FLAG="$FP_FLAG --fp8_scaled"
    echo -e "${BLUE}ℹ️ Imported from config: FP8_SCALED${NC}"
fi

# Attention Logic
ATTN_MODE="torch"
if python3 -c "import sageattention" &> /dev/null; then ATTN_MODE="sageattn"; elif python3 -c "import flash_attn" &> /dev/null; then ATTN_MODE="flash"; fi

# --- 5. UNIVERSAL LORA SELECTION (MoE MATCHING) ---
print_header "STAGE 2: LORA SELECTION"

# Build master lists from both directories
shopt -s nullglob
LOW_LORAS=()
for lora in "$OUT_LOW"/*.safetensors; do [[ "$lora" != *"_comfy"* ]] && [[ "$lora" != *"model_states"* ]] && LOW_LORAS+=("$lora"); done
HIGH_LORAS=()
for lora in "$OUT_HIGH"/*.safetensors; do [[ "$lora" != *"_comfy"* ]] && [[ "$lora" != *"model_states"* ]] && HIGH_LORAS+=("$lora"); done
shopt -u nullglob

ALL_AVAILABLE=("${LOW_LORAS[@]}" "${HIGH_LORAS[@]}")

if [ ${#ALL_AVAILABLE[@]} -eq 0 ]; then
    print_error "No LoRAs found in output directories!"
    exit 1
fi

echo -e "${CYAN}Select the LOW LoRA (Matched partner will be found automatically):${NC}"
for i in "${!ALL_AVAILABLE[@]}"; do
    FILE_PATH="${ALL_AVAILABLE[$i]}"
    [[ "$FILE_PATH" == "$OUT_LOW"* ]] && LABEL="${BLUE}[LOW]${NC}" || LABEL="${RED}[HIGH]${NC}"
    echo -e "  [$((i + 1))] $LABEL $(basename "$FILE_PATH")"
done

read -p "Selection (Default 1): " USER_CHOICE
SELECTED_PATH="${ALL_AVAILABLE[$((${USER_CHOICE:-1} - 1))]}"
SELECTED_NAME=$(basename "$SELECTED_PATH")

# Determine Regime & Set Search Target
if [[ "$SELECTED_PATH" == "$OUT_LOW"* ]]; then
    LORA_LOW="$SELECTED_PATH"
    SEARCH_DIR="$OUT_HIGH"
    PARTNER_REGIME="HIGH"
else
    LORA_HIGH="$SELECTED_PATH"
    SEARCH_DIR="$OUT_LOW"
    PARTNER_REGIME="LOW"
fi

# Attempt Auto-Match
PARTNER_PATH="$SEARCH_DIR/$SELECTED_NAME"
if [ -f "$PARTNER_PATH" ]; then
    print_success "Auto-matched $PARTNER_REGIME partner: $SELECTED_NAME"
    if [ "$PARTNER_REGIME" == "HIGH" ]; then LORA_HIGH="$PARTNER_PATH"; else LORA_LOW="$PARTNER_PATH"; fi
else
    print_warning "No exact filename match for $PARTNER_REGIME noise in $SEARCH_DIR."
    echo -e "${CYAN}Manual selection for $PARTNER_REGIME noise:${NC}"

    PARTNER_POOL=()
    [ "$PARTNER_REGIME" == "HIGH" ] && PARTNER_POOL=("${HIGH_LORAS[@]}") || PARTNER_POOL=("${LOW_LORAS[@]}")

    if [ ${#PARTNER_POOL[@]} -gt 0 ]; then
        for i in "${!PARTNER_POOL[@]}"; do echo "  [$((i + 1))] $(basename "${PARTNER_POOL[$i]}")"; done
        echo "  [0] Use primary for both (No MoE)"
        read -rp "Selection: " PARTNER_PICK
        if [ "${PARTNER_PICK:-0}" -eq 0 ]; then
            LORA_LOW="$SELECTED_PATH"
            LORA_HIGH="$SELECTED_PATH"
        else
            [ "$PARTNER_REGIME" == "HIGH" ] && LORA_HIGH="${PARTNER_POOL[$((PARTNER_PICK - 1))]}" || LORA_LOW="${PARTNER_POOL[$((PARTNER_PICK - 1))]}"
        fi
    else
        LORA_LOW="$SELECTED_PATH"
        LORA_HIGH="$SELECTED_PATH"
    fi
fi

# --- 6. EXECUTION ---
SAMPLES_DIR="$(dirname "$SELECTED_PATH")/eval_samples/$(basename "$SELECTED_NAME" .safetensors)"
TEMP_RUN_DIR="$SAMPLES_DIR/run_mult_${SAFE_MULT}"
mkdir -p "$TEMP_RUN_DIR"

print_header "STAGE 3: INFERENCE"
[ "$WAN_TASK" == "i2v-A14B" ] && CURRENT_SHIFT="5.0" || CURRENT_SHIFT="12.0"

INFER_FLAGS="--task $WAN_TASK --dit $WAN_DIT --dit_high_noise $WAN_DIT_HIGH --vae $WAN_VAE --t5 $WAN_T5 \
--lora_weight $LORA_LOW --lora_multiplier $LORA_MULTIPLIER \
--lora_weight_high_noise $LORA_HIGH --lora_multiplier_high_noise $LORA_MULTIPLIER \
--save_path $TEMP_RUN_DIR --video_size $IMAGE_SIZE_H $IMAGE_SIZE_W \
--video_length $GEN_LENGTH --infer_steps 30 --guidance_scale 5.0 --guidance_scale_high_noise 5.0 \
--flow_shift $CURRENT_SHIFT --attn_mode $ATTN_MODE $FP_FLAG --lazy_loading"

cd "$REPO_DIR" || exit
if [ "$WAN_TASK" == "t2v-A14B" ]; then
    PROMPT_FILE="$SAMPLES_DIR/temp_prompts.txt"
    > "$PROMPT_FILE"
    for item in "${EVAL_LIST[@]}"; do
        IFS="|" read -r TEXT SEED <<< "$item"
        echo "$TRIGGER, $TEXT. --d $SEED" >> "$PROMPT_FILE"
    done
    python3 "wan_generate_video.py" --from_file "$PROMPT_FILE" $INFER_FLAGS
else
    shopt -s nullglob nocaseglob
    IMAGE_POOL=("$DATASET_DIR"/*.{jpg,jpeg,png,webp})
    shopt -u nullglob nocaseglob
    for item in "${EVAL_LIST[@]}"; do
        IFS="|" read -r TEXT SEED <<< "$item"
        REF_IMAGE="${IMAGE_POOL[$((RANDOM % ${#IMAGE_POOL[@]}))]}"
        echo -e "\n${CYAN}🚀 Gen:${NC} $(basename "$REF_IMAGE")"
        python3 "wan_generate_video.py" --prompt "$TRIGGER, $TEXT" --image_path "$REF_IMAGE" --seed "$SEED" $INFER_FLAGS
    done
fi

# --- 7. POST-PROCESSING ---
print_header "STAGE 4: RENAMING & CLEANUP"
cd "$TEMP_RUN_DIR" || exit
shopt -s nullglob
for vid in *.mp4; do
    if [ "$IS_VIDEO" = false ]; then
        ffmpeg -i "$vid" -frames:v 1 -q:v 2 "$SAMPLES_DIR/${vid%.mp4}_mult${SAFE_MULT}.png" -loglevel error -y
    else
        mv "$vid" "$SAMPLES_DIR/${vid%.mp4}_mult${SAFE_MULT}.mp4"
    fi
done
rm -rf "$TEMP_RUN_DIR"
print_header "EVALUATION COMPLETE"
