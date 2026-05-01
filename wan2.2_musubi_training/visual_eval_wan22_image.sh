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
OUT_HIGH="${OUT_HIGH:-$NETWORK_VOLUME/output_folder_musubi/wan22/$TITLE_HIGH}"
OUT_LOW="${OUT_LOW:-$NETWORK_VOLUME/output_folder_musubi/wan22/$TITLE_LOW}"
DATASET_TYPE="${DATASET_TYPE:-image}"

REPO_DIR="$NETWORK_VOLUME/musubi-tuner"
MODELS_DIR="$NETWORK_VOLUME/models/Wan"
TRIGGER="${TITLE_HIGH:-Wan2.2_LoRA}"

WAN_VAE="$MODELS_DIR/wan_2.1_vae.safetensors"
WAN_T5="$MODELS_DIR/models_t5_umt5-xxl-enc-bf16.pth"

export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
export PYTORCH_ALLOC_CONF=expandable_segments:True

# --- 3. STAGE 1: TASK & MEDIA SELECTION ---
print_header "STAGE 1: TASK & MEDIA SELECTION"
echo -e "${CYAN}Select Inference Task:${NC}"
echo "1) Text-to-Video (t2v-A14B)"
echo "2) Image-to-Video (i2v-A14B)"
read -rp "Selection (1/2, default 1): " TASK_PICK
TASK_PICK=${TASK_PICK:-1}
WAN_TASK=$([ "$TASK_PICK" == "2" ] && echo "i2v-A14B" || echo "t2v-A14B")

echo -e "\n${CYAN}Select Media Type:${NC}"
echo "1) Image Eval (1 Frame - High-Noise DiT)"
echo "2) Video Eval (41 Frames - Standard DiT)"
read -rp "Selection (1/2, default 1): " MEDIA_PICK
MEDIA_PICK=${MEDIA_PICK:-1}

if [ "$MEDIA_PICK" == "2" ]; then
    GEN_LENGTH=41
    IS_VIDEO=true
    WAN_DIT_SUFFIX=$([ "$WAN_TASK" == "i2v-A14B" ] && echo "i2v_14B_bf16" || echo "t2v_14B_bf16")
    declare -a EVAL_LIST=(
        "walking forward confidently towards the camera|201"
        "turning head slowly to look at the camera, smiling|202"
        "standing still as the wind blows hair across her face|203"
    )
else
    GEN_LENGTH=1
    IS_VIDEO=false
    WAN_DIT_SUFFIX=$([ "$WAN_TASK" == "i2v-A14B" ] && echo "i2v_high_noise_14B_fp16" || echo "t2v_high_noise_14B_fp16")
    declare -a EVAL_LIST=(
        "portrait extreme close-up neutral expression studio lighting, highly detailed face|101"
        "full body shot standing confidently, fashion photography|103"
        "phone selfie holding a coffee cup, cozy cafe|105"
        "mirror selfie in modern apartment, casual outfit|106"
        "side profile shot, looking away from camera|116"
        # ... (Rest of your 23 original prompts)
    )
fi
WAN_DIT="$MODELS_DIR/wan2.2_${WAN_DIT_SUFFIX}.safetensors"

# --- 4. CONFIG-AWARE PARAMETER PREP (WITH SAFEGUARDS) ---
CLEAN_RES=$(echo $RESOLUTION_LIST | tr -d '",')
IMAGE_SIZE_W=$(echo $CLEAN_RES | awk '{print $1}')
IMAGE_SIZE_H=$(echo $CLEAN_RES | awk '{print $2}')

if [ "$IS_VIDEO" = true ]; then
    # FORCING SAFE VIDEO RESOLUTION
    IMAGE_SIZE_W=480
    IMAGE_SIZE_H=480
    echo -e "\n${YELLOW}⚠️ Video Mode Active: Using safe eval resolution ($IMAGE_SIZE_W x $IMAGE_SIZE_H).${NC}"
    echo -e "${YELLOW}   Custom resolution disabled to prevent OOM/Costly render.${NC}"
else
    echo -e "\n${CYAN}⚙️ Resolution Settings (Image Mode):${NC}"
    echo -e "Current Config Default: ${BOLD}$IMAGE_SIZE_W x $IMAGE_SIZE_H${NC}"
    read -p "Apply custom resolution? [y/N]: " USE_CUSTOM
    if [[ "$USE_CUSTOM" =~ ^[Yy]$ ]]; then
        read -p "Enter square resolution (e.g., 1024): " CUSTOM_VAL
        if [[ "$CUSTOM_VAL" =~ ^[0-9]+$ ]]; then
            IMAGE_SIZE_W=$CUSTOM_VAL
            IMAGE_SIZE_H=$CUSTOM_VAL
            echo -e "${GREEN}✅ Image Resolution set to ${IMAGE_SIZE_W}x${IMAGE_SIZE_H}${NC}"
        fi
    fi
fi

# Multiplier & Attention Logic
read -p "Enter LoRA multiplier (Default: 1.0): " LORA_MULT_INPUT
LORA_MULTIPLIER=${LORA_MULT_INPUT:-1.0}
FP_FLAG=$([ "${FP8_T5:-0}" -eq 1 ] && echo "--fp8_t5" || echo "")
ATTN_MODE="torch"
if python3 -c "import flash_attn" &> /dev/null; then ATTN_MODE="flash"; fi

# --- 5. DYNAMIC LORA SELECTION ---
TARGET_DIR=$([ "$DATASET_TYPE" == "video" ] && echo "$OUT_LOW" || echo "$OUT_HIGH")
print_header "STAGE 2: LORA SELECTION (Scanning $DATASET_TYPE output)"
shopt -s nullglob
AVAILABLE_LORAS=()
for lora in "$TARGET_DIR"/*.safetensors; do
    [[ "$lora" != *"_comfy"* ]] && [[ "$lora" != *"model_states"* ]] && AVAILABLE_LORAS+=("$lora")
done
shopt -u nullglob

if [ ${#AVAILABLE_LORAS[@]} -eq 0 ]; then
    echo -e "${RED}❌ No LoRAs in $TARGET_DIR${NC}"
    exit 1
fi

for i in "${!AVAILABLE_LORAS[@]}"; do
    LORA_NAME=$(basename "${AVAILABLE_LORAS[$i]}")
    [[ "$LORA_NAME" == "$OUTPUT_NAME.safetensors" ]] && LABEL="(FINAL)" || LABEL=""
    echo -e "  [$((i + 1))] ${BOLD}$LORA_NAME $LABEL${NC}"
done
read -p "Select number (Default 1): " USER_CHOICE
SELECTED_LORA="${AVAILABLE_LORAS[$((${USER_CHOICE:-1} - 1))]}"
SAMPLES_DIR="$TARGET_DIR/eval_samples/$(basename "$SELECTED_LORA" .safetensors)"
mkdir -p "$SAMPLES_DIR"

# --- 6. EXECUTION ---
CURRENT_SHIFT=$([ "$WAN_TASK" == "i2v-A14B" ] && echo "5.0" || echo "8.0")
print_header "STAGE 3: RUNNING INFERENCE"
echo -e "${YELLOW}📊 Profile:${NC} ${BOLD}$IMAGE_SIZE_W x $IMAGE_SIZE_H${NC} | Task: ${BOLD}$WAN_TASK${NC}"

INFER_FLAGS="--task $WAN_TASK --dit $WAN_DIT --vae $WAN_VAE --t5 $WAN_T5 --lora_weight $SELECTED_LORA --lora_multiplier $LORA_MULTIPLIER --save_path $SAMPLES_DIR --video_size $IMAGE_SIZE_W $IMAGE_SIZE_H --video_length $GEN_LENGTH --infer_steps 30 --guidance_scale 4.5 --flow_shift $CURRENT_SHIFT --attn_mode $ATTN_MODE $FP_FLAG"

cd "$REPO_DIR" || exit
if [ "$WAN_TASK" == "t2v-A14B" ]; then
    PROMPT_FILE="$SAMPLES_DIR/temp_prompts.txt"
    > "$PROMPT_FILE"
    for item in "${EVAL_LIST[@]}"; do
        IFS="|" read -r TEXT SEED <<< "$item"
        [[ "${TEXT,,}" == *"${TRIGGER,,}"* ]] && P="$TEXT" || P="$TRIGGER. $TEXT."
        echo "$P --seed $SEED" >> "$PROMPT_FILE"
    done
    python3 "wan_generate_video.py" --from_file "$PROMPT_FILE" $INFER_FLAGS
else
    shopt -s nullglob nocaseglob
    IMAGE_POOL=("$DATASET_DIR"/*.{jpg,jpeg,png,webp})
    shopt -u nullglob nocaseglob
    for item in "${EVAL_LIST[@]}"; do
        IFS="|" read -r TEXT SEED <<< "$item"
        REF_IMAGE="${IMAGE_POOL[$((RANDOM % ${#IMAGE_POOL[@]}))]}"
        CAP_FILE="${REF_IMAGE%.*}.txt"
        CAPTION=$([ -f "$CAP_FILE" ] && xargs < "$CAP_FILE" || echo "a professional photo of a woman")
        [[ "${CAPTION,,}" == *"${TRIGGER,,}"* ]] && FINAL_P="${CAPTION}. ${TEXT}" || FINAL_P="${TRIGGER}, ${CAPTION}. ${TEXT}"
        echo -e "\n${CYAN}🚀 Gen:${NC} $(basename "$REF_IMAGE") (Seed $SEED)"
        python3 "wan_generate_video.py" --prompt "$FINAL_P" --image_path "$REF_IMAGE" --seed "$SEED" $INFER_FLAGS
    done
fi

# --- 7. POST-PROCESSING ---
print_header "STAGE 4: CLEANUP"
cd "$SAMPLES_DIR" || exit
shopt -s nullglob
for vid in *.mp4; do
    img="${vid%.mp4}.png"
    ffmpeg -i "$vid" -frames:v 1 -q:v 2 "$img" -loglevel error -y
    if [ "$IS_VIDEO" = false ]; then
        echo -e "${GREEN}✨ Created Image:${NC} $img"
        rm "$vid"
    else echo -e "${BLUE}🎬 Created Video:${NC} $vid"; fi
done
shopt -u nullglob

print_header "EVALUATION COMPLETE"
echo -e "Results saved in: ${BOLD}$SAMPLES_DIR${NC}"
