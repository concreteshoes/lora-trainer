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

print_warning() { echo -e "${YELLOW}[WARN]${NC} $1"; }

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
REPO_DIR="$NETWORK_VOLUME/musubi-tuner"
MODELS_DIR="$NETWORK_VOLUME/models/Wan"
DATABASE_DIR="$NETWORK_VOLUME/image_dataset_here"
TRIGGER="${TITLE_HIGH:-Wan2.2_LoRA}"

WAN_VAE="$MODELS_DIR/wan_2.1_vae.safetensors"
WAN_T5="$MODELS_DIR/models_t5_umt5-xxl-enc-bf16.pth"

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
FP_FLAG=""
if [ "${FP8_T5:-0}" -eq 1 ]; then
    FP_FLAG="$FP_FLAG --fp8_t5"
fi

# 4. Conservative Attention Mode (Sage is risky for video generation)
ATTN_MODE="torch"
if python3 -c "import sage_attn" &> /dev/null || python3 -c "import sageattention" &> /dev/null; then
    ATTN_MODE="sageattn"
    echo -e "${PURPLE}🚀 SageAttention detected (High-Speed Inference Mode).${NC}"
elif python3 -c "import flash_attn" &> /dev/null; then
    ATTN_MODE="flash"
    echo -e "${CYAN}⚡ Flash Attention detected.${NC}"
fi

# --- 4. TASK SELECTION ---
print_header "STAGE 1: TASK SELECTION"
echo -e "${CYAN}Select Inference Task:${NC}"
echo "1) Text-to-Video (t2v-A14B)"
echo "2) Image-to-Video (i2v-A14B)"
read -rp "Selection (1/2, default 1): " TASK_PICK
TASK_PICK=${TASK_PICK:-1}

if [ "$TASK_PICK" == "2" ]; then
    WAN_TASK="i2v-A14B"
    # Fallback to DATASET_DIR if DATABASE_DIR isn't in config
    REF_DIR="${DATABASE_DIR:-$DATASET_DIR}"
    if [ ! -d "$REF_DIR" ]; then
        echo -e "${RED}❌ Error: Reference directory not found: $REF_DIR${NC}"
        exit 1
    fi
else
    WAN_TASK="t2v-A14B"
fi

# --- 5. DYNAMIC LORA SELECTION ---
print_header "STAGE 2: LORA SELECTION"
echo -e "${BLUE}🔍 Scanning for LoRAs in:${NC} $OUT_HIGH"

shopt -s nullglob
# Grab everything first
RAW_SCAN=("$OUT_HIGH"/*.safetensors)
shopt -u nullglob

AVAILABLE_LORAS=()
for lora in "${RAW_SCAN[@]}"; do
    # Filter out the comfy-converted ones so we only test the raw training output
    if [[ "$lora" != *"_comfy"* ]]; then
        AVAILABLE_LORAS+=("$lora")
    fi
done

if [ ${#AVAILABLE_LORAS[@]} -eq 0 ]; then
    echo -e "${RED}❌ Error: No raw training checkpoints found in $OUT_HIGH${NC}"
    exit 1
elif [ ${#AVAILABLE_LORAS[@]} -eq 1 ]; then
    SELECTED_LORA="${AVAILABLE_LORAS[0]}"
else
    echo -e "${CYAN}Multiple LoRAs detected. Please select one for inference:${NC}"
    for i in "${!AVAILABLE_LORAS[@]}"; do
        # Human-friendly index (starts at 1)
        DISPLAY_IDX=$((i + 1))
        LORA_NAME=$(basename "${AVAILABLE_LORAS[$i]}")

        # Label the final checkpoint clearly
        if [[ "$LORA_NAME" == "$OUTPUT_NAME.safetensors" ]]; then
            echo -e "  [$DISPLAY_IDX] ${BOLD}$LORA_NAME (FINAL CHECKPOINT)${NC}"
        else
            echo -e "  [$DISPLAY_IDX] ${BOLD}$LORA_NAME${NC}"
        fi
    done

    read -p "Enter number (1-${#AVAILABLE_LORAS[@]}, Default 1): " USER_CHOICE
    USER_CHOICE=${USER_CHOICE:-1}

    # Convert human-friendly choice back to 0-based index
    LORA_IDX=$((USER_CHOICE - 1))

    if [[ "$USER_CHOICE" =~ ^[0-9]+$ ]] && [ "$LORA_IDX" -ge 0 ] && [ "$LORA_IDX" -lt "${#AVAILABLE_LORAS[@]}" ]; then
        SELECTED_LORA="${AVAILABLE_LORAS[$LORA_IDX]}"
    else
        echo -e "${RED}❌ Invalid selection. Defaulting to 1.${NC}"
        SELECTED_LORA="${AVAILABLE_LORAS[0]}"
    fi
fi

# --- Define Paths AFTER Selection ---
LORA_FILENAME=$(basename "$SELECTED_LORA" .safetensors)
SAMPLES_DIR="$OUT_HIGH/eval_samples/$LORA_FILENAME"
mkdir -p "$SAMPLES_DIR"

# WAN 2.2 LOGIC:
# Using High-Noise DiT to check Layout/Identity mapping.
if [ "$WAN_TASK" == "i2v-A14B" ]; then
    WAN_DIT="$MODELS_DIR/wan2.2_i2v_high_noise_14B_fp16.safetensors"
    CURRENT_SHIFT=5.0
else
    CURRENT_SHIFT=8.0
    WAN_DIT="$MODELS_DIR/wan2.2_t2v_high_noise_14B_fp16.safetensors"
fi

cd "$REPO_DIR" || exit

# --- 6. DEFINE PROMPTS ---
declare -a EVAL_LIST=(

    # --- IDENTITY BASELINE ---
    "portrait extreme close-up neutral expression studio lighting, highly detailed face|101"
    "clean studio portrait, sharp focus, natural skin texture, centered composition|102"

    # --- FULL BODY / COMPOSITION ---
    "full body shot standing confidently, balanced composition, fashion photography|103"
    "walking forward, dynamic pose, natural motion blur, street photography|104"

    # --- LIFESTYLE / INFLUENCER ---
    "phone selfie holding a coffee cup, cozy cafe, social media influencer style|105"
    "mirror selfie in modern apartment, casual outfit, relaxed pose|106"

    # --- CLOTHING VARIATION ---
    "wearing an elegant evening gown, luxury hotel lobby, cinematic lighting|107"
    "wearing casual streetwear, hoodie and jeans, urban environment|108"
    "wearing athletic gym outfit, fitness setting, energetic pose|109"

    # --- ENVIRONMENT SHIFT ---
    "outdoor park, sunlight through trees, soft shadows, natural colors|110"
    "busy city street, cars and people, urban atmosphere|111"
    "tropical beach, bright daylight, wind in hair|112"

    # --- LIGHTING STRESS (VERY IMPORTANT) ---
    "low light cinematic scene, soft shadows, dramatic lighting|113"
    "golden hour lighting, warm tones, shallow depth of field|114"
    "night scene with neon lights, high contrast, cinematic look|115"

    # --- CAMERA / ANGLES ---
    "side profile shot, looking away from camera, shallow depth of field|116"
    "over the shoulder shot, depth of field, cinematic framing|117"
    "top-down angle, looking up, dynamic perspective|118"

    # --- STYLE VARIATION ---
    "professional fashion photoshoot, editorial style, high detail|119"
    "candid street photography, natural imperfections, documentary style|120"

    # --- HARD STRESS TESTS ---
    "wearing a formal business suit, professional office environment|121"
    "cyberpunk neon city at night, wet pavement reflections|122"
    "dramatic high contrast lighting, editorial magazine style|123"
)

# --- 7. EXECUTION ---

echo -e "\n${BLUE}${BOLD}======================================================"
print_header "STAGE 3: RUNNING INFERENCE FOR $WAN_TASK"
print_warning "NOTE: Using High-Noise DiT. Results will be soft/painterly."
echo -e "======================================================"
echo -e "${YELLOW}📊 Inference Profile:${NC}"
echo -e "   > Resolution: ${BOLD}$IMAGE_SIZE_W x $IMAGE_SIZE_H${NC}"
echo -e "   > Rank/Alpha: ${BOLD}$LORA_RANK / $LORA_ALPHA${NC}"
echo -e "   > Attention:  ${BOLD}$ATTN_MODE${NC}"
echo -e "   > Checkpoint: ${BOLD}$SELECTED_LORA${NC}"
echo -e "------------------------------------------------------"

# Assemble standard flags
INFER_FLAGS="--task $WAN_TASK \
--dit $WAN_DIT \
--vae $WAN_VAE \
--t5 $WAN_T5 \
--lora_weight $SELECTED_LORA \
--lora_multiplier 1.0 \
--save_path $SAMPLES_DIR \
--video_size $IMAGE_SIZE_W $IMAGE_SIZE_H \
--video_length 1 \
--infer_steps 30 \
--guidance_scale 4.5 \
--flow_shift $CURRENT_SHIFT \
--attn_mode $ATTN_MODE \
$FP_FLAG"

if [ "$WAN_TASK" == "t2v-A14B" ]; then
    # --- T2V BATCH MODE ---
    PROMPT_FILE="$SAMPLES_DIR/temp_prompts.txt"
    > "$PROMPT_FILE"
    for item in "${EVAL_LIST[@]}"; do
        IFS="|" read -r TEXT SEED <<< "$item"
        # Since these are generic, the check is simple:
        if [[ "${TEXT,,}" == *"${TRIGGER,,}"* ]]; then
            echo "$TEXT --seed $SEED" >> "$PROMPT_FILE"
        else
            echo "$TRIGGER. $TEXT. --seed $SEED" >> "$PROMPT_FILE"
        fi
    done

    python3 "$REPO_DIR/wan_generate_video.py" \
        --from_file "$PROMPT_FILE" \
        $INFER_FLAGS

else
    # --- I2V SEQUENTIAL MODE ---
    shopt -s nullglob nocaseglob
    IMAGE_POOL=("$REF_DIR"/*.{jpg,jpeg,png,webp})
    shopt -u nullglob nocaseglob

    if [ ${#IMAGE_POOL[@]} -eq 0 ]; then
        echo -e "${RED}❌ Error: No images found in $REF_DIR for I2V!${NC}"
        exit 1
    fi

    for item in "${EVAL_LIST[@]}"; do
        IFS="|" read -r TEXT SEED <<< "$item"

        RANDOM_IDX=$(shuf -i 0-$((${#IMAGE_POOL[@]} - 1)) -n 1)
        REF_IMAGE="${IMAGE_POOL[$RANDOM_IDX]}"

        # 1. Clean Captioning Logic
        CAPTION=""
        CAP_FILE="${REF_IMAGE%.*}.txt"
        if [ -f "$CAP_FILE" ]; then
            CAPTION=$(xargs < "$CAP_FILE")
        else
            CAPTION="a professional photo of a woman"
        fi

        # 2. Smart Prompt Construction for Wan 2.2
        # If the trigger is already there, don't double up.
        if [[ "${CAPTION,,}" == *"${TRIGGER,,}"* ]]; then
            # Caption already has the name, just append the 'action/text'
            FINAL_PROMPT="${CAPTION}. ${TEXT}"
        else
            # Prepend trigger if missing
            FINAL_PROMPT="${TRIGGER}, ${CAPTION}. ${TEXT}"
        fi

        echo -e "\n${CYAN}🚀 Wan 2.2 I2V (Seed $SEED)${NC}"
        echo -e "${BOLD}Ref:${NC} $(basename "$REF_IMAGE")"
        echo -e "${BOLD}Prompt:${NC} $FINAL_PROMPT"

        # 3. Execution (Note: guidance_scale for Wan is best at 4.0-5.0)
        python3 "$REPO_DIR/wan_generate_video.py" \
            --prompt "$FINAL_PROMPT" \
            --image_path "$REF_IMAGE" \
            --seed "$SEED" \
            --lora_multiplier 1.0 \
            --guidance_scale 4.5 \
            $INFER_FLAGS
    done
fi

# --- 8. POST-PROCESSING (MP4 -> PNG) ---
print_header "STAGE 4: EXTRACTING FRAMES"
cd "$SAMPLES_DIR" || exit

shopt -s nullglob
for vid in *.mp4; do
    img="${vid%.mp4}.png"
    ffmpeg -i "$vid" -frames:v 1 -q:v 2 "$img" -loglevel error -y

    if [ -f "$img" ]; then
        echo -e "${GREEN}✨ Created Image:${NC} $img"
        rm "$vid"
    fi
done
shopt -u nullglob

print_header "EVALUATION COMPLETE"
echo -e "Results saved in: ${BOLD}$SAMPLES_DIR${NC}"
