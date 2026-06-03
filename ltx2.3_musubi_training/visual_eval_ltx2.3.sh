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
print_success() { echo -e "${GREEN}[OK]   ${NC} $1"; }
print_error() { echo -e "${RED}[FAIL]${NC} $1"; }

# --- LOAD CONFIGURATION ---
CONFIG_FILE="${1:-ltx_musubi_config.sh}"
if [ -f "$CONFIG_FILE" ]; then
    source "$CONFIG_FILE"
    echo -e "${GREEN}✅ Config loaded:${NC} $CONFIG_FILE"
else
    echo -e "${RED}❌ Error: $CONFIG_FILE not found!${NC}"
    exit 1
fi

# --- PATHS SETUP ---
REPO_DIR="$NETWORK_VOLUME/musubi-tuner"
MODELS_DIR="$NETWORK_VOLUME/models/ltx"
LTX_DIT="$MODELS_DIR/ltx-2.3-22b-dev-fp8.safetensors"
LTX_TE="$MODELS_DIR/gemma_3_12B_it_fp8_e4m3fn.safetensors"
OUTPUT_DIR="$NETWORK_VOLUME/output_folder_musubi/ltx23/$OUTPUT_NAME"

export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
export PYTORCH_ALLOC_CONF="expandable_segments:True"

# --- DATASET DETECTION (For I2V Context) ---
if [[ "$DATASET_DIR" == *"video_dataset_here"* ]]; then
    ACTIVE_MODE="VIDEO"
    EXT_PATTERN="*.{mp4,mkv,mov,avi}"
else
    ACTIVE_MODE="IMAGE"
    EXT_PATTERN="*.{jpg,jpeg,png,webp}"
fi

# --- STAGE 1: TASK & TRIGGER SELECTION ---
print_header "STAGE 1: TASK & TRIGGER SELECTION"
echo -e "${CYAN}Enter the trigger word you used for your dataset:${NC}"
read -rp "Trigger: " USER_TRIGGER
TRIGGER="${USER_TRIGGER:-LTX2.3_LoRA}"

echo -e "\n${CYAN}Select inference task:${NC}"
echo "1) Text-to-Media (T2V / T2I)"
echo "2) Image-to-Video (I2V / I2I)"
read -rp "Selection (1/2, default 1): " TASK_PICK
TASK_PICK=${TASK_PICK:-1}

if [ "$TASK_PICK" == "2" ]; then
    LTX23_TASK="i2v"
    if [ "$ACTIVE_MODE" == "IMAGE" ]; then
        GEN_LENGTH=1
        IS_VIDEO=false
        I2V_SAMPLE_COUNT=5
        echo -e "\n${CYAN}ℹ️  I2V — Image dataset detected: 5 random images, 1 frame each (I2I).${NC}"
    else
        GEN_LENGTH=45
        IS_VIDEO=true
        I2V_SAMPLE_COUNT=1
        echo -e "\n${CYAN}ℹ️  I2V — Video dataset detected: 1 random video, 45 frames (I2V).${NC}"
    fi
else
    LTX23_TASK="t2v"
    echo -e "\n${CYAN}Select Media Type:${NC}"
    echo "1) Standard Image Eval (1 Frame - T2I)"
    echo "2) Standard Video Eval (45 Frames - T2V)"
    read -rp "Selection (1/2, default 1): " MEDIA_PICK
    MEDIA_PICK=${MEDIA_PICK:-1}

    if [ "$MEDIA_PICK" == "2" ]; then
        GEN_LENGTH=45
        IS_VIDEO=true
        declare -a EVAL_LIST=(
            "walking forward confidently towards the camera|201"
            "turning head slowly to look at the camera, smiling|202"
            "standing still as the wind blows hair across her face|203"
            "laughing looking at the camera, natural glow on skin, shallow depth of field, lifestyle aesthetic|204"
        )
    else
        GEN_LENGTH=1
        IS_VIDEO=false
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
            "close-up portrait wearing sunglasses pushed slightly down, eyes visible, fashion editorial look, sharp facial detail|119"
            "close-up portrait with messy bun hairstyle, soft morning light, natural skin imperfections visible, cozy indoor aesthetic|120"
        )
    fi
fi

# --- HELPERS FOR ROUNDING ---
round_to_32() {
    echo $((($1 + 16) / 32 * 32))
}

# --- INTERACTIVE RESOLUTION SELECTION ---
echo -e "\n${BOLD}${YELLOW} Select Inference Resolution:${NC}"
echo -e "  1) 1024 x 1024 (Square)"
echo -e "  2) 1024 x 576  (16:9 Landscape)"
echo -e "  3) 576 x 1024  (9:16 Portrait)"
echo -e "  4) Custom Resolution"
read -r RES_CHOICE

case "$RES_CHOICE" in
    2)
        IMAGE_SIZE_H=576
        IMAGE_SIZE_W=1024
        ;;
    3)
        IMAGE_SIZE_H=1024
        IMAGE_SIZE_W=576
        ;;
    4)
        echo -n "Enter Custom Height (e.g., 576): "
        read -r raw_h
        echo -n "Enter Custom Width (e.g., 1024): "
        read -r raw_w
        IMAGE_SIZE_H=$(round_to_32 "${raw_h:-576}")
        IMAGE_SIZE_W=$(round_to_32 "${raw_w:-1024}")
        ;;
    1 | *)
        IMAGE_SIZE_H=1024
        IMAGE_SIZE_W=1024
        ;;
esac

echo -e "${GREEN}🎯 Inference Resolution Set To:${NC} ${IMAGE_SIZE_W}x${IMAGE_SIZE_H} (Rounded to multiple of 32)"

read -rp "Enter LoRA multiplier (Default 1.0): " LORA_MULT_INPUT
LORA_MULTIPLIER=${LORA_MULT_INPUT:-1.0}
SAFE_MULT=$(echo "$LORA_MULTIPLIER" | tr '.' '-')

read -rp "Enter Denoising Steps (Default 25): " STEPS_INPUT
INFER_STEPS=${STEPS_INPUT:-25}

# Attention Logic
ATTN_FLAG="--sdpa"
if python3 -c "import flash_attn" &> /dev/null; then
    ATTN_FLAG="--flash_attn"
fi

# --- STAGE 2: MANUAL LORA SELECTION ---
print_header "STAGE 2: MANUAL LORA SELECTION"

select_lora_checkpoint() {
    local dir="$1"
    shopt -s nullglob
    local all_files=("$dir"/*.safetensors)
    shopt -u nullglob
    IFS=$'\n' all_files=($(sort <<< "${all_files[*]}"))
    unset IFS
    local filtered_files=()
    for f in "${all_files[@]}"; do
        [[ "$f" == *"_comfy"* ]] && continue
        [[ "$f" == *"model_states"* ]] && continue
        filtered_files+=("$f")
    done
    if [ ${#filtered_files[@]} -eq 0 ]; then
        print_error "No snapshots found in $dir" >&2
        exit 1
    fi
    echo -e "${BLUE}Select LTX-2.3 LoRA snapshot (from $dir):${NC}" >&2
    for i in "${!filtered_files[@]}"; do
        local display_idx=$((i + 1))
        local name=$(basename "${filtered_files[$i]}")
        printf "  [%2d] %-45s\n" "$display_idx" "$name" >&2
    done
    local default_idx=${#filtered_files[@]}
    read -rp "Choice (1-$default_idx, default $default_idx): " user_pick < /dev/tty
    local final_pick=${user_pick:-$default_idx}
    if [[ "$final_pick" =~ ^[0-9]+$ ]] && [ "$final_pick" -ge 1 ] && [ "$final_pick" -le "$default_idx" ]; then
        echo "${filtered_files[$((final_pick - 1))]}"
    else
        echo "${filtered_files[$((default_idx - 1))]}"
    fi
}

# --- SET DYNAMIC PATHS ---
SELECTED_LORA=$(select_lora_checkpoint "$OUTPUT_DIR")
LORA_PATH="$SELECTED_LORA"
LORA_FILENAME=$(basename "$LORA_PATH" .safetensors)
print_success "Selected LoRA: $LORA_FILENAME"

# --- GPU DETECTION ---
GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader 2> /dev/null | wc -l)
GPU_COUNT=${GPU_COUNT:-1}
if [ "$GPU_COUNT" -ge 2 ]; then
    print_success "Dual GPU detected — parallel inference enabled."
    [ "$LTX23_TASK" == "i2v" ] && [ "$ACTIVE_MODE" == "VIDEO" ] && I2V_SAMPLE_COUNT=2
else
    print_status "Single GPU — sequential inference."
fi

# --- OUTPUT DIRECTORIES SETUP ---
SAMPLES_DIR="$OUTPUT_DIR/eval_samples/$LORA_FILENAME"
mkdir -p "$SAMPLES_DIR"

if [ "$GPU_COUNT" -ge 2 ]; then
    TEMP_RUN_DIR_0="$SAMPLES_DIR/run_mult_${SAFE_MULT}_gpu0"
    TEMP_RUN_DIR_1="$SAMPLES_DIR/run_mult_${SAFE_MULT}_gpu1"
    mkdir -p "$TEMP_RUN_DIR_0" "$TEMP_RUN_DIR_1"
else
    TEMP_RUN_DIR_0="$SAMPLES_DIR/run_mult_${SAFE_MULT}"
    mkdir -p "$TEMP_RUN_DIR_0"
fi

# --- STAGE 3: EXECUTION ---
print_header "STAGE 3: INFERENCE"

# Handle dynamic extraction of config swap size
SWAP_FLAG=""
if [ -n "${BLOCKS_TO_SWAP}" ]; then
    SWAP_FLAG="--blocks_to_swap $BLOCKS_TO_SWAP"
fi

echo -e "${BLUE}${BOLD}======================================================"
echo -e "      LTX-2.3 IMAGE & VIDEO AUTOMATED INFERENCE"
echo -e "======================================================"
echo -e "${YELLOW}📊 Inference Profile:${NC}"
echo -e "   > Mode:          ${BOLD}$LTX23_TASK${NC}"
echo -e "   > Target Frame:  ${BOLD}$GEN_LENGTH${NC}"
echo -e "   > Resolution:    ${BOLD}$IMAGE_SIZE_H x $IMAGE_SIZE_W${NC}"
echo -e "   > Attention:     ${BOLD}$ATTN_FLAG${NC}"
echo -e "   > Multiplier:    ${BOLD}$LORA_MULTIPLIER${NC}"
echo -e "   > Denoise Steps: ${BOLD}$INFER_STEPS${NC}"
if [ -n "$BLOCKS_TO_SWAP" ]; then
    echo -e "   > VRAM Swap:     ${BOLD}${YELLOW}$BLOCKS_TO_SWAP Blocks Offloaded via CPU System RAM${NC}"
else
    echo -e "   > VRAM Swap:     ${BOLD}${GREEN}Disabled (Full GPU Mode)${NC}"
fi
echo -e "${BLUE}${BOLD}======================================================${NC}\n"

# COMBINING PARSED TOML FLAGS WITH HARDCODED MODELS & INTERACTIVE FLAGS
BASE_FLAGS="$PARSED_FLAGS \
--ltx2_checkpoint $LTX_DIT \
--gemma_safetensors $LTX_TE --gemma_fp8_weight_offload \
--lora_weight $SELECTED_LORA --lora_multiplier $LORA_MULTIPLIER \
--height $IMAGE_SIZE_H --width $IMAGE_SIZE_W --num_frames $GEN_LENGTH \
--steps $INFER_STEPS --sampling_preset ltx23 $ATTN_FLAG \
--fp8_base \
$SWAP_FLAG"

cd "$REPO_DIR" || exit

if [ "$GPU_COUNT" -ge 2 ]; then
    print_status "Splitting generation workflow across GPU 0 and GPU 1..."
    if [ "$LTX23_TASK" == "i2v" ]; then
        shopt -s nullglob nocaseglob
        eval "MEDIA_POOL=($DATASET_DIR/$EXT_PATTERN)"
        shopt -u nullglob nocaseglob
        if [ ${#MEDIA_POOL[@]} -eq 0 ]; then
            print_error "No source media files located in $DATASET_DIR"
            exit 1
        fi
        GPU0_COUNT=$(((I2V_SAMPLE_COUNT + 1) / 2))
        GPU1_COUNT=$((I2V_SAMPLE_COUNT / 2))

        (
            export CUDA_VISIBLE_DEVICES=0
            for ((i = 1; i <= GPU0_COUNT; i++)); do
                RAND_MEDIA="${MEDIA_POOL[$((RANDOM % ${#MEDIA_POOL[@]}))]}"
                BASE_NAME=$(basename "${RAND_MEDIA%.*}")
                CAPTION=$([ -f "$DATASET_DIR/$BASE_NAME.txt" ] && cat "$DATASET_DIR/$BASE_NAME.txt" || echo "cinematic portrait")
                REF_IMAGE="$RAND_MEDIA"
                if [ "$ACTIVE_MODE" == "VIDEO" ]; then
                    REF_IMAGE="$TEMP_RUN_DIR_0/frame_ref_${i}.jpg"
                    ffmpeg -i "$RAND_MEDIA" -vframes 1 -q:v 2 "$REF_IMAGE" -loglevel error -y
                fi
                echo -e "\n${CYAN}🚀 [GPU0] I2V [$i/$GPU0_COUNT]:${NC} $BASE_NAME"
                python3 ltx2_generate_video.py --prompt "$TRIGGER, $CAPTION" --reference_image "$REF_IMAGE" \
                    --seed $((100 + i)) $BASE_FLAGS --output_dir "$TEMP_RUN_DIR_0" --output_name "sample_gpu0_${i}"
            done
        ) &
        PID_0=$!

        (
            export CUDA_VISIBLE_DEVICES=1
            for ((i = 1; i <= GPU1_COUNT; i++)); do
                RAND_MEDIA="${MEDIA_POOL[$((RANDOM % ${#MEDIA_POOL[@]}))]}"
                BASE_NAME=$(basename "${RAND_MEDIA%.*}")
                CAPTION=$([ -f "$DATASET_DIR/$BASE_NAME.txt" ] && cat "$DATASET_DIR/$BASE_NAME.txt" || echo "cinematic portrait")
                REF_IMAGE="$RAND_MEDIA"
                if [ "$ACTIVE_MODE" == "VIDEO" ]; then
                    REF_IMAGE="$TEMP_RUN_DIR_1/frame_ref_${i}.jpg"
                    ffmpeg -i "$RAND_MEDIA" -vframes 1 -q:v 2 "$REF_IMAGE" -loglevel error -y
                fi
                echo -e "\n${CYAN}🚀 [GPU1] I2V [$i/$GPU1_COUNT]:${NC} $BASE_NAME"
                python3 ltx2_generate_video.py --prompt "$TRIGGER, $CAPTION" --reference_image "$REF_IMAGE" \
                    --seed $((100 + GPU0_COUNT + i)) $BASE_FLAGS --output_dir "$TEMP_RUN_DIR_1" --output_name "sample_gpu1_${i}"
            done
        ) &
        PID_1=$!
    else
        (
            export CUDA_VISIBLE_DEVICES=0
            for i in "${!EVAL_LIST[@]}"; do
                ((i % 2 != 0)) && continue
                IFS="|" read -r TEXT SEED <<< "${EVAL_LIST[$i]}"
                echo -e "\n${CYAN}🚀 [GPU0] T2V/I:${NC} $TEXT"
                python3 ltx2_generate_video.py --prompt "$TRIGGER, $TEXT" --seed "$SEED" \
                    $BASE_FLAGS --output_dir "$TEMP_RUN_DIR_0" --output_name "sample_${SEED}"
            done
        ) &
        PID_0=$!

        (
            export CUDA_VISIBLE_DEVICES=1
            for i in "${!EVAL_LIST[@]}"; do
                ((i % 2 == 0)) && continue
                IFS="|" read -r TEXT SEED <<< "${EVAL_LIST[$i]}"
                echo -e "\n${CYAN}🚀 [GPU1] T2V/I:${NC} $TEXT"
                python3 ltx2_generate_video.py --prompt "$TRIGGER, $TEXT" --seed "$SEED" \
                    $BASE_FLAGS --output_dir "$TEMP_RUN_DIR_1" --output_name "sample_${SEED}"
            done
        ) &
        PID_1=$!
    fi
    wait $PID_0 $PID_1
    print_success "Dual GPU inference completed successfully."
else
    # Single GPU Fallback
    if [ "$LTX23_TASK" == "i2v" ]; then
        shopt -s nullglob nocaseglob
        eval "MEDIA_POOL=($DATASET_DIR/$EXT_PATTERN)"
        shopt -u nullglob nocaseglob
        if [ ${#MEDIA_POOL[@]} -eq 0 ]; then
            print_error "No media found in $DATASET_DIR"
            exit 1
        fi
        for ((i = 1; i <= I2V_SAMPLE_COUNT; i++)); do
            RAND_MEDIA="${MEDIA_POOL[$((RANDOM % ${#MEDIA_POOL[@]}))]}"
            BASE_NAME=$(basename "${RAND_MEDIA%.*}")
            CAPTION=$([ -f "$DATASET_DIR/$BASE_NAME.txt" ] && cat "$DATASET_DIR/$BASE_NAME.txt" || echo "cinematic portrait")
            REF_IMAGE="$RAND_MEDIA"
            if [ "$ACTIVE_MODE" == "VIDEO" ]; then
                REF_IMAGE="$TEMP_RUN_DIR_0/frame_ref_${i}.jpg"
                ffmpeg -i "$RAND_MEDIA" -vframes 1 -q:v 2 "$REF_IMAGE" -loglevel error -y
            fi
            echo -e "\n${CYAN}🚀 I2V [$i/$I2V_SAMPLE_COUNT]:${NC} $BASE_NAME"
            python3 ltx2_generate_video.py --prompt "$TRIGGER, $CAPTION" --reference_image "$REF_IMAGE" \
                --seed $((100 + i)) $BASE_FLAGS --output_dir "$TEMP_RUN_DIR_0" --output_name "sample_${i}"
        done
    else
        for item in "${EVAL_LIST[@]}"; do
            IFS="|" read -r TEXT SEED <<< "$item"
            echo -e "\n${CYAN}🚀 T2V/I:${NC} $TEXT"
            python3 ltx2_generate_video.py --prompt "$TRIGGER, $TEXT" --seed "$SEED" \
                $BASE_FLAGS --output_dir "$TEMP_RUN_DIR_0" --output_name "sample_${SEED}"
        done
    fi
fi

# --- STAGE 4: POST-PROCESSING & RENAMING ---
print_header "STAGE 4: RENAMING & CLEANUP"
ALL_TEMP_DIRS=("$TEMP_RUN_DIR_0")
[ "$GPU_COUNT" -ge 2 ] && ALL_TEMP_DIRS+=("$TEMP_RUN_DIR_1")

for dir in "${ALL_TEMP_DIRS[@]}"; do
    [ -d "$dir" ] || continue
    cd "$dir" || continue
    shopt -s nullglob

    for file in *.{mp4,png,jpg,jpeg}; do
        filename="${file%.*}"

        if [ "$IS_VIDEO" = false ]; then
            cp "$file" "$SAMPLES_DIR/${filename}_mult${SAFE_MULT}.png"
            echo -e "${GREEN}✨ Image saved:${NC} ${filename}_mult${SAFE_MULT}.png"
        else
            cp "$file" "$SAMPLES_DIR/${filename}_mult${SAFE_MULT}.mp4"
            echo -e "${BLUE}🎬 Video saved:${NC} ${filename}_mult${SAFE_MULT}.mp4"
        fi
    done
    shopt -u nullglob
    rm -rf "$dir"
done

print_header "EVALUATION COMPLETE"
echo -e "Results saved to target output folder: ${BOLD}$SAMPLES_DIR${NC}"
