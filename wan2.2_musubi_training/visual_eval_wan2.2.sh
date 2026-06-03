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

# --- 1. LOAD CONFIGURATION ---
CONFIG_FILE="${1:-wan_musubi_config.sh}"
if [ -f "$CONFIG_FILE" ]; then
    source "$CONFIG_FILE"
    echo -e "${GREEN}✅ Config loaded:${NC} $CONFIG_FILE"
else
    echo -e "${RED}❌ Error: $CONFIG_FILE not found!${NC}"
    exit 1
fi

# --- 2. DATASET DETECTION ---
if [[ "$DATASET_DIR" == *"video_dataset_here"* ]]; then
    ACTIVE_MODE="VIDEO"
    EXT_PATTERN="*.{mp4,mkv,mov,avi}"
else
    ACTIVE_MODE="IMAGE"
    EXT_PATTERN="*.{jpg,jpeg,png,webp}"
fi

REPO_DIR="$NETWORK_VOLUME/musubi-tuner"
MODELS_DIR="$NETWORK_VOLUME/models/wan"
WAN_VAE="$MODELS_DIR/Wan2_1_VAE_bf16.safetensors"
WAN_T5="$MODELS_DIR/models_t5_umt5-xxl-enc-bf16.pth"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
export PYTORCH_ALLOC_CONF="expandable_segments:True"

# Explicit Expert Paths
WAN_DIT_T2V_HIGH="$MODELS_DIR/Wan-2.2-T2V-High-Noise-BF16.safetensors"
WAN_DIT_T2V_LOW="$MODELS_DIR/Wan-2.2-T2V-Low-Noise-BF16.safetensors"
WAN_DIT_I2V_HIGH="$MODELS_DIR/Wan-2.2-I2V-High-Noise-BF16.safetensors"
WAN_DIT_I2V_LOW="$MODELS_DIR/Wan-2.2-I2V-Low-Noise-BF16.safetensors"

# --- 3. STAGE 1: TASK & TRIGGER SELECTION ---
print_header "STAGE 1: TASK & TRIGGER SELECTION"
echo -e "${CYAN}Enter the trigger word you used for your dataset:${NC}"
read -rp "Trigger: " USER_TRIGGER
TRIGGER="${USER_TRIGGER:-Wan2.2_LoRA}"

echo -e "\n${CYAN}Select inference task for either images or videos:${NC}"
echo "1) Text-to-Video (t2v-A14B)"
echo "2) Image-to-Video (i2v-A14B)"
read -rp "Selection (1/2, default 1): " TASK_PICK
TASK_PICK=${TASK_PICK:-1}

if [ "$TASK_PICK" == "2" ]; then
    WAN_TASK="i2v-A14B"
    if [ "$ACTIVE_MODE" == "IMAGE" ]; then
        GEN_LENGTH=1
        IS_VIDEO=false
        I2V_SAMPLE_COUNT=5
        echo -e "\n${CYAN}ℹ️  I2V — Image dataset detected: 5 random images, 1 frame each.${NC}"
    else
        GEN_LENGTH=41
        IS_VIDEO=true
        I2V_SAMPLE_COUNT=1
        echo -e "\n${CYAN}ℹ️  I2V — Video dataset detected: 1 random video, 41 frames.${NC}"
    fi
else
    WAN_TASK="t2v-A14B"
    echo -e "\n${CYAN}Select Media Type:${NC}"
    echo "1) Standard Image Eval (1 Frame)"
    echo "2) Standard Video Eval (41 Frames)"
    read -rp "Selection (1/2, default 1): " MEDIA_PICK
    MEDIA_PICK=${MEDIA_PICK:-1}
    if [ "$MEDIA_PICK" == "2" ]; then
        GEN_LENGTH=41
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

# --- IMAGE INFERENCE TOGGLE ---
echo -e "\n${CYAN}Do you want to run image inference?${NC}"
read -rp "Run inference? (y/n, default y): " RUN_INFER_INPUT
RUN_INFER="${RUN_INFER_INPUT:-y}"

# --- LOW-FOR-BOTH PROMPT (only if inference is enabled) ---
USE_LOW_FOR_BOTH=false
if [[ "$RUN_INFER" =~ ^[Yy]$ ]]; then
    echo -e "\n${CYAN}Load LOW LoRA for both HIGH and LOW DiT inputs?${NC}"
    echo -e "  ${BLUE}(Recommended when training image LoRA on LOW DiT only)${NC}"
    read -rp "LOW for both DiTs? (y/n, default n): " LOW_BOTH_INPUT
    if [[ "${LOW_BOTH_INPUT:-n}" =~ ^[Yy]$ ]]; then
        USE_LOW_FOR_BOTH=true
        echo -e "${BLUE}ℹ️  LOW LoRA will be loaded for both DiT inputs.${NC}"
    else
        echo -e "${BLUE}ℹ️  Separate HIGH and LOW LoRA inputs will be used.${NC}"
    fi
fi

# --- 4. PREP PARAMETERS ---
CLEAN_RES=$(echo $RESOLUTION_LIST | tr -d '",')
IMAGE_SIZE_H=$(echo $CLEAN_RES | awk '{print $1}')
IMAGE_SIZE_W=$(echo $CLEAN_RES | awk '{print $2}')
if [ "$IS_VIDEO" = true ]; then
    IMAGE_SIZE_H=832
    IMAGE_SIZE_W=480
fi

read -p "Enter LoRA multiplier (Default 1.0): " LORA_MULT_INPUT
LORA_MULTIPLIER=${LORA_MULT_INPUT:-1.0}
SAFE_MULT=$(echo "$LORA_MULTIPLIER" | tr '.' '-')

# --- FLAG INITIALIZATION & LOGIC ---
FP_FLAGS="--fp8_t5"
echo -e "${BLUE}ℹ️ Using default: FP8_T5${NC}"

if [ "${FP8_BASE:-0}" -eq 1 ]; then
    FP_FLAGS="$FP_FLAGS --fp8"
    echo -e "${BLUE}ℹ️ Imported from config: FP8_BASE${NC}"
fi
if [ "${FP8_SCALED:-0}" -eq 1 ]; then
    FP_FLAGS="$FP_FLAGS --fp8_scaled"
    echo -e "${BLUE}ℹ️ Imported from config: FP8_SCALED${NC}"
fi

# --- Safe Blocks to Swap Injection ---
INFER_FLAG=""
if [ -n "$BLOCKS_TO_SWAP" ]; then
    INFER_FLAG="--blocks_to_swap $BLOCKS_TO_SWAP"
    echo -e "${BLUE}ℹ️ Offloading Enabled: Swapping $BLOCKS_TO_SWAP blocks to CPU RAM${NC}"
fi

# Attention Logic
ATTN_MODE="torch"
if python3 -c "import sageattention" &> /dev/null; then
    ATTN_MODE="sageattn"
elif python3 -c "import flash_attn" &> /dev/null; then
    ATTN_MODE="flash"
fi

# --- DiT LOADING STRATEGY ---
DIT_RAM_THRESHOLD_GB=32
FREE_RAM_GB=$(awk '/MemAvailable/ {printf "%.0f", $2/1024/1024}' /proc/meminfo)
if [ "$FREE_RAM_GB" -ge "$DIT_RAM_THRESHOLD_GB" ]; then
    DIT_LOAD_FLAG="--offload_inactive_dit"
    print_success "RAM available: ${FREE_RAM_GB} GB — DiT offload mode: ${BOLD}CPU RAM${NC}"
else
    DIT_LOAD_FLAG="--lazy_loading"
    print_warning "RAM available: ${FREE_RAM_GB} GB (< ${DIT_RAM_THRESHOLD_GB} GB threshold) — DiT offload mode: DISK (lazy loading)"
fi

# --- EARLY EXIT IF INFERENCE SKIPPED ---
if [[ ! "$RUN_INFER" =~ ^[Yy]$ ]]; then
    print_warning "Inference skipped. Exiting."
    exit 0
fi

# --- 5. LORA SELECTION ---
print_header "STAGE 2: MANUAL LORA SELECTION"

# Dynamic output paths
OUT_HIGH="$NETWORK_VOLUME/output_folder_musubi/wan2.2/$WAN_TASK/$TITLE_HIGH"
OUT_LOW="$NETWORK_VOLUME/output_folder_musubi/wan2.2/$WAN_TASK/$TITLE_LOW"

# --- HELPER FUNCTION FOR SELECTION ---
select_lora_expert() {
    local dir="$1"
    local expert_label="$2"
    local color="$3"
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
    echo -e "${color}Select $expert_label LoRA (from $dir):${NC}" >&2
    for i in "${!filtered_files[@]}"; do
        local display_idx=$((i + 1))
        local name=$(basename "${filtered_files[$i]}")
        local tag=""
        if [[ "$name" == *"_pre_resume"* ]]; then
            tag="${YELLOW}(Archived)${NC}"
        elif [[ "$name" == *"-step"* ]]; then
            tag="${PURPLE}(EMA Step)${NC}"
        else
            tag="(Epoch)"
        fi
        printf "  [%2d] %-45s %b\n" "$display_idx" "$name" "$tag" >&2
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

# --- PERFORM SELECTIONS ---
SELECTED_LOW=$(select_lora_expert "$OUT_LOW" "LOW-Noise" "$BLUE")
echo -e "${GREEN}✅ Selected Low:${NC} $(basename "$SELECTED_LOW")\n"

if [ "$USE_LOW_FOR_BOTH" = true ]; then
    SELECTED_HIGH="$SELECTED_LOW"
    echo -e "${BLUE}ℹ️  HIGH DiT input: using LOW LoRA — skipping separate HIGH selection.${NC}\n"
else
    SELECTED_HIGH=$(select_lora_expert "$OUT_HIGH" "HIGH-Noise" "$RED")
    echo -e "${GREEN}✅ Selected High:${NC} $(basename "$SELECTED_HIGH")\n"
fi

LORA_LOW="$SELECTED_LOW"
LORA_HIGH="$SELECTED_HIGH"
HIGH_NAME=$(basename "$SELECTED_HIGH" .safetensors)
LOW_NAME=$(basename "$SELECTED_LOW" .safetensors)

# --- GPU DETECTION ---
GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader 2> /dev/null | wc -l)
GPU_COUNT=${GPU_COUNT:-1}
if [ "$GPU_COUNT" -ge 2 ]; then
    print_success "Dual GPU detected — parallel inference enabled."
    if [ "$WAN_TASK" == "i2v-A14B" ] && [ "$ACTIVE_MODE" == "VIDEO" ]; then
        I2V_SAMPLE_COUNT=2
    fi
else
    print_status "Single GPU — sequential inference."
fi

# --- SAMPLES OUTPUT DIR ---
if [ "$USE_LOW_FOR_BOTH" = true ]; then
    SAMPLES_DIR="$NETWORK_VOLUME/output_folder_musubi/wan2.2/$WAN_TASK/eval_samples/${LOW_NAME}__low_for_both"
else
    SAMPLES_DIR="$NETWORK_VOLUME/output_folder_musubi/wan2.2/$WAN_TASK/eval_samples/${HIGH_NAME}__${LOW_NAME}"
fi
mkdir -p "$SAMPLES_DIR"

if [ "$GPU_COUNT" -ge 2 ]; then
    TEMP_RUN_DIR_0="$SAMPLES_DIR/run_mult_${SAFE_MULT}_gpu0"
    TEMP_RUN_DIR_1="$SAMPLES_DIR/run_mult_${SAFE_MULT}_gpu1"
    mkdir -p "$TEMP_RUN_DIR_0" "$TEMP_RUN_DIR_1"
else
    TEMP_RUN_DIR_0="$SAMPLES_DIR/run_mult_${SAFE_MULT}"
    mkdir -p "$TEMP_RUN_DIR_0"
fi

# --- 6. EXECUTION ---
print_header "STAGE 3: INFERENCE"

if [ "$WAN_TASK" == "i2v-A14B" ]; then
    WAN_DIT="$WAN_DIT_I2V_LOW"
    WAN_DIT_HIGH="$WAN_DIT_I2V_HIGH"
    CURRENT_SHIFT="5.0"
else
    WAN_DIT="$WAN_DIT_T2V_LOW"
    WAN_DIT_HIGH="$WAN_DIT_T2V_HIGH"
    CURRENT_SHIFT="12.0"
fi

echo -e "${BLUE}${BOLD}======================================================"
echo -e "      WAN 2.2 IMAGE & VIDEO AUTOMATED INFERENCE"
echo -e "======================================================"
echo -e "${YELLOW}📊 Inference Profile:${NC}"
echo -e "   > Task:         ${BOLD}$WAN_TASK${NC}"
echo -e "   > Resolution:   ${BOLD}$IMAGE_SIZE_H x $IMAGE_SIZE_W${NC}"
echo -e "   > Rank/Alpha:   ${BOLD}$LORA_RANK / $LORA_ALPHA${NC}"
echo -e "   > Attention:    ${BOLD}$ATTN_MODE${NC}"
echo -e "   > Multiplier:   ${BOLD}$LORA_MULTIPLIER${NC}"
echo -e "   > LOW for both: ${BOLD}$USE_LOW_FOR_BOTH${NC}"
echo -e "   > DiT offload:  ${BOLD}$DIT_LOAD_FLAG${NC}"
if [ -n "$BLOCKS_TO_SWAP" ]; then
    echo -e "   > VRAM Swap:    ${BOLD}${YELLOW}$BLOCKS_TO_SWAP Blocks Offloaded via CPU System RAM${NC}"
fi
echo -e "${BLUE}${BOLD}======================================================${NC}\n"

# --- SAFELY ASSEMBLE BASE FLAGS ---
BASE_FLAGS="--task $WAN_TASK --dit $WAN_DIT --dit_high_noise $WAN_DIT_HIGH --vae $WAN_VAE --t5 $WAN_T5 \
--lora_weight $LORA_LOW --lora_multiplier $LORA_MULTIPLIER \
--lora_weight_high_noise $LORA_HIGH --lora_multiplier_high_noise $LORA_MULTIPLIER \
--video_size $IMAGE_SIZE_H $IMAGE_SIZE_W \
--video_length $GEN_LENGTH --infer_steps 30 --guidance_scale 5.0 --guidance_scale_high_noise 5.0 \
--flow_shift $CURRENT_SHIFT --attn_mode $ATTN_MODE"

[ -n "$FP_FLAGS" ] && BASE_FLAGS="$BASE_FLAGS $FP_FLAGS"
[ -n "$INFER_FLAG" ] && BASE_FLAGS="$BASE_FLAGS $INFER_FLAG"
[ -n "$DIT_LOAD_FLAG" ] && BASE_FLAGS="$BASE_FLAGS $DIT_LOAD_FLAG"

cd "$REPO_DIR" || exit

if [ "$GPU_COUNT" -ge 2 ]; then
    print_status "Splitting work across GPU 0 and GPU 1..."
    if [ "$WAN_TASK" == "i2v-A14B" ]; then
        shopt -s nullglob nocaseglob
        eval "MEDIA_POOL=($DATASET_DIR/$EXT_PATTERN)"
        shopt -u nullglob nocaseglob
        if [ ${#MEDIA_POOL[@]} -eq 0 ]; then
            print_error "No media found in $DATASET_DIR"
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
                python3 "wan_generate_video.py" --prompt "$TRIGGER, $CAPTION" --image_path "$REF_IMAGE" \
                    --seed $((100 + i)) $BASE_FLAGS --save_path "$TEMP_RUN_DIR_0"
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
                python3 "wan_generate_video.py" --prompt "$TRIGGER, $CAPTION" --image_path "$REF_IMAGE" \
                    --seed $((100 + GPU0_COUNT + i)) $BASE_FLAGS --save_path "$TEMP_RUN_DIR_1"
            done
        ) &
        PID_1=$!
    else
        (
            export CUDA_VISIBLE_DEVICES=0
            for i in "${!EVAL_LIST[@]}"; do
                ((i % 2 != 0)) && continue
                IFS="|" read -r TEXT SEED <<< "${EVAL_LIST[$i]}"
                echo -e "\n${CYAN}🚀 [GPU0] T2V:${NC} $TEXT"
                python3 "wan_generate_video.py" --prompt "$TRIGGER, $TEXT" --seed "$SEED" \
                    $BASE_FLAGS --save_path "$TEMP_RUN_DIR_0"
            done
        ) &
        PID_0=$!
        (
            export CUDA_VISIBLE_DEVICES=1
            for i in "${!EVAL_LIST[@]}"; do
                ((i % 2 == 0)) && continue
                IFS="|" read -r TEXT SEED <<< "${EVAL_LIST[$i]}"
                echo -e "\n${CYAN}🚀 [GPU1] T2V:${NC} $TEXT"
                python3 "wan_generate_video.py" --prompt "$TRIGGER, $TEXT" --seed "$SEED" \
                    $BASE_FLAGS --save_path "$TEMP_RUN_DIR_1"
            done
        ) &
        PID_1=$!
    fi
    wait $PID_0 $PID_1
    print_success "Dual GPU inference complete."
else
    if [ "$WAN_TASK" == "i2v-A14B" ]; then
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
            python3 "wan_generate_video.py" --prompt "$TRIGGER, $CAPTION" --image_path "$REF_IMAGE" \
                --seed $((100 + i)) $BASE_FLAGS --save_path "$TEMP_RUN_DIR_0"
        done
    else
        for item in "${EVAL_LIST[@]}"; do
            IFS="|" read -r TEXT SEED <<< "$item"
            echo -e "\n${CYAN}🚀 T2V:${NC} $TEXT"
            python3 "wan_generate_video.py" --prompt "$TRIGGER, $TEXT" --seed "$SEED" \
                $BASE_FLAGS --save_path "$TEMP_RUN_DIR_0"
        done
    fi
fi

# --- 8. POST-PROCESSING ---
print_header "STAGE 4: RENAMING & CLEANUP"
ALL_TEMP_DIRS=("$TEMP_RUN_DIR_0")
[ "${GPU_COUNT:-1}" -ge 2 ] && ALL_TEMP_DIRS+=("$TEMP_RUN_DIR_1")

for dir in "${ALL_TEMP_DIRS[@]}"; do
    [ -d "$dir" ] || continue
    cd "$dir" || continue
    shopt -s nullglob
    for vid in *.mp4; do
        if [ "$IS_VIDEO" = false ]; then
            ffmpeg -i "$vid" -frames:v 1 -q:v 2 "$SAMPLES_DIR/${vid%.mp4}_mult${SAFE_MULT}.jpeg" -loglevel error -y
            echo -e "${GREEN}✨ Image:${NC} ${vid%.mp4}_mult${SAFE_MULT}.jpeg"
        else
            mv "$vid" "$SAMPLES_DIR/${vid%.mp4}_mult${SAFE_MULT}.mp4"
            echo -e "${BLUE}🎬 Video:${NC} ${vid%.mp4}_mult${SAFE_MULT}.mp4"
        fi
    done
    shopt -u nullglob
    rm -rf "$dir"
done

print_header "EVALUATION COMPLETE"
echo -e "Results saved in: ${BOLD}$SAMPLES_DIR${NC}"
