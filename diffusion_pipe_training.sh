#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status

# Colors for better UX - compatible with both light and dark terminals
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
BOLD='\033[1m'
WHITE='\033[0;37m'
NC='\033[0m' # No Color

# Function to print colored output
print_header() {
    echo -e "${CYAN}================================================${NC}"
    echo -e "${BOLD}$1${NC}"
    echo -e "${CYAN}================================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

# Welcome message
clear
print_header "Welcome to Comprehensive LoRA Trainer using Diffusion Pipe"
echo ""
echo -e "${PURPLE}This interactive script will guide you through setting up and starting a LoRA training session.${NC}"
echo -e "${RED}Before you start, make sure to add your datasets to their respective folders.${NC}"
echo ""

# Create logs directory
mkdir -p "$NETWORK_VOLUME/logs"

# Flash Attention — installed via prebuilt wheel in start.sh
if python -c "import flash_attn" &> /dev/null; then
    print_success "flash-attn is installed and ready."
else
    print_warning "flash-attn not found — training will fall back to standard attention."
fi
echo ""

# Model selection
echo -e "${BOLD}Please select the model you want to train:${NC}"
echo ""
echo "1) Flux1-dev"
echo "2) SDXL"
echo "3) Wan 1.3B"
echo "4) Wan 14B Text-To-Video"
echo "5) Wan 14B Image-To-Video"
echo "6) Qwen Image"
echo "7) Z-Image Turbo - Ostris v2 adapter"

echo ""

while true; do
    read -p "Enter your choice (1-7): " model_choice
    case $model_choice in
        1)
            MODEL_TYPE="flux"
            MODEL_NAME="Flux"
            TOML_FILE="flux.toml"
            break
            ;;
        2)
            MODEL_TYPE="sdxl"
            MODEL_NAME="SDXL"
            TOML_FILE="sdxl.toml"
            break
            ;;
        3)
            MODEL_TYPE="wan13"
            MODEL_NAME="Wan 1.3B"
            TOML_FILE="wan13_video.toml"
            break
            ;;
        4)
            MODEL_TYPE="wan14b_t2v"
            MODEL_NAME="Wan 14B Text-To-Video"
            TOML_FILE="wan14b_t2v.toml"
            break
            ;;
        5)
            MODEL_TYPE="wan14b_i2v"
            MODEL_NAME="Wan 14B Image-To-Video"
            TOML_FILE="wan14b_i2v.toml"
            break
            ;;
        6)
            MODEL_TYPE="qwen"
            MODEL_NAME="Qwen Image"
            TOML_FILE="qwen.toml"
            break
            ;;
        7)
            MODEL_TYPE="z_image_turbo"
            MODEL_NAME="Z Image Turbo"
            TOML_FILE="z_image_turbo.toml"
            break
            ;;
        *)
            print_error "Invalid choice. Please enter a number between 1-7."
            ;;
    esac
done

echo ""

# Check and set required API keys
if [ "$MODEL_TYPE" = "flux" ]; then
    if [[ -z "${HF_TOKEN:-}" || "$HF_TOKEN" == "token_here" ]]; then
        print_warning "Hugging Face token is required for Flux model."
        echo ""
        echo "You can get your token from: https://huggingface.co/settings/tokens"
        echo ""
        read -p "Please enter your Hugging Face token: " hf_token
        if [[ -z "${hf_token:-}" ]]; then
            print_error "Token cannot be empty. Exiting."
            exit 1
        fi
        export HF_TOKEN="$hf_token"
        print_success "Hugging Face token set successfully."
    else
        print_success "Hugging Face token already set."
    fi
fi

echo ""

# Dataset selection
print_header "Dataset Configuration"
echo ""
echo -e "${BOLD}Do you want to caption images and/or videos?${NC}"
echo ""
echo "1) Images only"
echo "2) Videos only"
echo "3) Both images and videos"
echo "4) Skip captioning (use existing captions)"
echo ""

while true; do
    read -p "Enter your choice (1-4): " caption_choice
    case $caption_choice in
        1)
            CAPTION_MODE="images"
            break
            ;;
        2)
            CAPTION_MODE="videos"
            break
            ;;
        3)
            CAPTION_MODE="both"
            break
            ;;
        4)
            CAPTION_MODE="skip"
            break
            ;;
        *)
            print_error "Invalid choice. Please enter a number between 1-4."
            ;;
    esac
done

echo ""

# Check dataset directories
if [ "$CAPTION_MODE" != "skip" ]; then
    IMAGE_DIR="$NETWORK_VOLUME/image_dataset_here"
    VIDEO_DIR="$NETWORK_VOLUME/video_dataset_here"

    # Select video captioner if video captioning is needed
    if [ "$CAPTION_MODE" = "videos" ] || [ "$CAPTION_MODE" = "both" ]; then
        echo -e "${BOLD}Video Captioner Selection:${NC}"
        echo ""
        echo "1) Qwen2.5-VL (local, no API key required — recommended)"
        echo "2) Gemini (cloud-based, requires API key — free tier has frame limits)"
        echo ""

        while true; do
            read -p "Enter your choice (1-2): " captioner_choice
            case $captioner_choice in
                1)
                    VIDEO_CAPTIONER="qwen"
                    print_success "Qwen2.5-VL selected for video captioning."
                    break
                    ;;
                2)
                    VIDEO_CAPTIONER="gemini"
                    print_success "Gemini selected for video captioning."
                    break
                    ;;
                *)
                    print_error "Invalid choice. Please enter 1 or 2."
                    ;;
            esac
        done
        echo ""

        # Only prompt for Gemini API key if Gemini was selected
        if [ "$VIDEO_CAPTIONER" = "gemini" ]; then
            if [ -z "$GEMINI_API_KEY" ] || [ "$GEMINI_API_KEY" = "token_here" ]; then
                print_warning "Gemini API key is required for video captioning."
                echo ""
                echo "You can get your API key from: https://aistudio.google.com/app/apikey"
                echo ""
                read -p "Please enter your Gemini API key: " gemini_key
                if [ -z "$gemini_key" ]; then
                    print_error "API key cannot be empty. Exiting."
                    exit 1
                fi
                export GEMINI_API_KEY="$gemini_key"
                print_success "Gemini API key set successfully."
            else
                print_success "Gemini API key already set."
            fi
            echo ""
        fi
    fi

    # Ask for trigger word if image captioning is needed
    TRIGGER_WORD=""
    if [ "$CAPTION_MODE" = "images" ] || [ "$CAPTION_MODE" = "both" ]; then
        echo -e "${BOLD}Image Captioning Configuration:${NC}"
        echo ""
        read -p "Enter a trigger word for image captions (or press Enter for none): " TRIGGER_WORD
        if [ -n "$TRIGGER_WORD" ]; then
            print_success "Trigger word set: '$TRIGGER_WORD'"
        else
            print_info "No trigger word set"
        fi
        echo ""
    fi

    # Function to check if directory has files
    check_directory() {
        local dir=$1
        local type=$2

        if [ ! -d "$dir" ]; then
            print_error "$type directory does not exist: $dir"
            return 1
        fi

        # Check for files (not just directories)
        if [ "$type" = "Image" ]; then
            file_count=$(find "$dir" -maxdepth 1 -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.bmp" -o -iname "*.gif" -o -iname "*.tiff" -o -iname "*.webp" \) | wc -l)
        else
            file_count=$(find "$dir" -maxdepth 1 -type f \( -iname "*.mp4" -o -iname "*.avi" -o -iname "*.mov" -o -iname "*.mkv" -o -iname "*.webm" \) | wc -l)
        fi

        if [ "$file_count" -eq 0 ]; then
            print_error "No $type files found in: $dir"
            return 1
        fi

        print_success "Found $file_count $type file(s) in: $dir"
        return 0
    }

    # Check based on caption mode
    case $CAPTION_MODE in
        "images")
            if ! check_directory "$IMAGE_DIR" "Image"; then
                echo ""
                print_error "Please add images to $IMAGE_DIR and re-run this script."
                exit 1
            fi
            ;;
        "videos")
            if ! check_directory "$VIDEO_DIR" "Video"; then
                echo ""
                print_error "Please add videos to $VIDEO_DIR and re-run this script."
                exit 1
            fi
            ;;
        "both")
            images_ok=true
            videos_ok=true

            if ! check_directory "$IMAGE_DIR" "Image"; then
                images_ok=false
            fi

            if ! check_directory "$VIDEO_DIR" "Video"; then
                videos_ok=false
            fi

            if [ "$images_ok" = false ] || [ "$videos_ok" = false ]; then
                echo ""
                print_error "Please add the missing files and re-run this script."
                if [ "$images_ok" = false ]; then
                    echo "  - Add images to: $IMAGE_DIR"
                fi
                if [ "$videos_ok" = false ]; then
                    echo "  - Add videos to: $VIDEO_DIR"
                fi
                exit 1
            fi
            ;;
    esac
fi

echo ""
print_success "Dataset validation completed successfully!"
echo ""

echo "     To access JupyterLab, use the SSH command and port provided by you host"
echo "     ssh -p <SSH_PORT> hostname@<SERVER_IP> -L 8888:localhost:8888"
echo "     Then open your local browser: http://localhost:8888/lab"

# Summary
print_header "Training Configuration Summary"
echo ""
echo -e "${WHITE}Model:${NC} $MODEL_NAME"
echo -e "${WHITE}TOML Config:${NC} $TOML_FILE"
echo -e "${WHITE}Caption Mode:${NC} $CAPTION_MODE"

if [ "$MODEL_TYPE" = "flux" ]; then
    echo -e "${WHITE}Hugging Face Token:${NC} Set ✓"
fi

if [ "$CAPTION_MODE" = "videos" ] || [ "$CAPTION_MODE" = "both" ]; then
    if [ "$VIDEO_CAPTIONER" = "gemini" ]; then
        echo -e "${WHITE}Video Captioner:${NC} Gemini | API Key: Set ✓"
    else
        echo -e "${WHITE}Video Captioner:${NC} Qwen2.5-VL (local)"
    fi
fi

echo ""
print_info "Configuration completed! Starting model download and setup..."
echo ""

# CUDA compatibility check
check_cuda_compatibility() {
    python3 << 'PYTHON_EOF'
import sys
try:
    import torch
    if torch.cuda.is_available():
        # Try a simple CUDA operation to test kernel compatibility
        x = torch.randn(1, device='cuda')
        y = x * 2
        print("CUDA compatibility check passed")
    else:
        print("\n" + "="*70)
        print("CUDA NOT AVAILABLE")
        print("="*70)
        print("\nCUDA is not available on this system.")
        print("This script requires CUDA to run.")
        print("\nSOLUTION:")
        print("  Please deploy with CUDA 12.8 when selecting your GPU")
        print("  This template requires CUDA 12.8")
        print("\n" + "="*70)
        sys.exit(1)
except RuntimeError as e:
    error_msg = str(e).lower()
    if "no kernel image" in error_msg or "cuda error" in error_msg:
        print("\n" + "="*70)
        print("CUDA KERNEL COMPATIBILITY ERROR")
        print("="*70)
        print("\nThis error occurs when your GPU architecture is not supported")
        print("by the installed CUDA kernels. This typically happens when:")
        print("  • Your GPU model is older or different from what was expected")
        print("  • The PyTorch/CUDA build doesn't include kernels for your GPU")
        print("\nSOLUTIONS:")
        print("  1. Use a newer GPU model (recommended):")
        print("     • H100 or H200 GPUs are recommended for best compatibility")
        print("  2. Ensure correct CUDA version:")
        print("     • Filter for CUDA 12.8 when selecting your GPU")
        print("     • This template requires CUDA 12.8")
        print("\n" + "="*70)
        sys.exit(1)
    else:
        raise
PYTHON_EOF
    if [ $? -ne 0 ]; then
        exit 1
    fi
}

print_header "Checking CUDA Compatibility"
check_cuda_compatibility
echo ""

# Model download logic - start in background
print_header "Starting Model Download"
echo ""

mkdir -p "$NETWORK_VOLUME/models"

# Initialize MODEL_DOWNLOAD_PID to ensure it's always set
MODEL_DOWNLOAD_PID=""

case $MODEL_TYPE in
    "flux" | "sdxl" | "wan13" | "wan14b_t2v" | "wan14b_i2v" | "qwen" | "z_image_turbo")

        # Determine file names and output folders per model
        case $MODEL_TYPE in
            "flux")
                TOML_FILE="flux.toml"
                OUTPUT_DIR="$NETWORK_VOLUME/output_folder/flux_lora"
                ;;
            "sdxl")
                TOML_FILE="sdxl.toml"
                OUTPUT_DIR="$NETWORK_VOLUME/output_folder/sdxl_lora"
                ;;
            "wan13")
                TOML_FILE="wan13_video.toml"
                OUTPUT_DIR="$NETWORK_VOLUME/output_folder/wan13_lora"
                ;;
            "wan14b_t2v")
                TOML_FILE="wan14b_t2v.toml"
                OUTPUT_DIR="$NETWORK_VOLUME/output_folder/wan14b_t2v_lora"
                ;;
            "wan14b_i2v")
                TOML_FILE="wan14b_i2v.toml"
                OUTPUT_DIR="$NETWORK_VOLUME/output_folder/wan14b_i2v_lora"
                ;;
            "qwen")
                TOML_FILE="qwen.toml"
                OUTPUT_DIR="$NETWORK_VOLUME/output_folder/qwen_lora"
                ;;
            "z_image_turbo")
                TOML_FILE="z_image_turbo.toml"
                OUTPUT_DIR="$NETWORK_VOLUME/output_folder/z_image_turbo_lora"
                ;;
        esac

        # Ensure examples directory exists
        mkdir -p "$NETWORK_VOLUME/diffusion_pipe/examples"

        # Set MODEL_TOML path
        MODEL_TOML="$NETWORK_VOLUME/diffusion_pipe/examples/$TOML_FILE"
        SOURCE_TOML="$NETWORK_VOLUME/lora-trainer/toml_files/$TOML_FILE"

        # Copy or update TOML
        if [ -f "$MODEL_TOML" ]; then
            print_info "$TOML_FILE already exists in examples directory"
        elif [ -f "$SOURCE_TOML" ]; then
            cp "$SOURCE_TOML" "$MODEL_TOML"
            print_success "Moved $TOML_FILE to examples directory"
        else
            print_warning "$TOML_FILE not found at expected location: $SOURCE_TOML"
            print_warning "Please ensure the file exists or manually copy it to: $MODEL_TOML"
        fi

        # Update output_dir in TOML
        if [ -f "$MODEL_TOML" ]; then
            sed -i "s|^[[:space:]]*output_dir[[:space:]]*=.*|output_dir = '$OUTPUT_DIR'|" "$MODEL_TOML"
        fi

        # Model-specific background downloads
        case $MODEL_TYPE in
            "flux")
                if [[ -z "${HF_TOKEN:-}" || "$HF_TOKEN" == "token_here" ]]; then
                    print_error "HF_TOKEN is not set properly."
                    exit 1
                fi
                print_info "Starting Flux model download in background..."
                mkdir -p "$NETWORK_VOLUME/models/flux"
                hf download black-forest-labs/FLUX.1-dev --local-dir "$NETWORK_VOLUME/models/flux" --repo-type model --token "$HF_TOKEN" > "$NETWORK_VOLUME/logs/model_download.log" 2>&1 &
                MODEL_DOWNLOAD_PID=$!
                ;;
            "sdxl")
                print_info "Starting Base SDXL model download in background..."
                hf download timoshishi/sdXL_v10VAEFix sdXL_v10VAEFix.safetensors --local-dir "$NETWORK_VOLUME/models/" > "$NETWORK_VOLUME/logs/model_download.log" 2>&1 &
                MODEL_DOWNLOAD_PID=$!
                ;;
            "wan13")
                print_info "Starting Wan 1.3B model download in background..."
                mkdir -p "$NETWORK_VOLUME/models/Wan/Wan2.1-T2V-1.3B"
                hf download Wan-AI/Wan2.1-T2V-1.3B --local-dir "$NETWORK_VOLUME/models/Wan/Wan2.1-T2V-1.3B" > "$NETWORK_VOLUME/logs/model_download.log" 2>&1 &
                MODEL_DOWNLOAD_PID=$!
                ;;
            "wan14b_t2v")
                print_info "Starting Wan 14B T2V model download in background..."
                mkdir -p "$NETWORK_VOLUME/models/Wan/Wan2.1-T2V-14B"
                hf download Wan-AI/Wan2.1-T2V-14B --local-dir "$NETWORK_VOLUME/models/Wan/Wan2.1-T2V-14B" > "$NETWORK_VOLUME/logs/model_download.log" 2>&1 &
                MODEL_DOWNLOAD_PID=$!
                ;;
            "wan14b_i2v")
                print_info "Starting Wan 14B I2V model download in background..."
                mkdir -p "$NETWORK_VOLUME/models/Wan/Wan2.1-I2V-14B-480P"
                hf download Wan-AI/Wan2.1-I2V-14B-480P --local-dir "$NETWORK_VOLUME/models/Wan/Wan2.1-I2V-14B-480P" > "$NETWORK_VOLUME/logs/model_download.log" 2>&1 &
                MODEL_DOWNLOAD_PID=$!
                ;;
            "qwen")
                print_info "Starting Qwen Image model download in background..."
                mkdir -p "$NETWORK_VOLUME/models/Qwen-Image"
                hf download Qwen/Qwen-Image --local-dir "$NETWORK_VOLUME/models/Qwen-Image" > "$NETWORK_VOLUME/logs/model_download.log" 2>&1 &
                MODEL_DOWNLOAD_PID=$!
                ;;
            "z_image_turbo")
                print_info "Starting Z Image Turbo model download in background..."
                mkdir -p "$NETWORK_VOLUME/models/z_image"
                (
                    hf download Comfy-Org/z_image_turbo --local-dir "$NETWORK_VOLUME/models/z_image_turbo_temp"
                    mv "$NETWORK_VOLUME/models/z_image_turbo_temp/split_files/diffusion_models/z_image_turbo_bf16.safetensors" "$NETWORK_VOLUME/models/z_image/"
                    mv "$NETWORK_VOLUME/models/z_image_turbo_temp/split_files/vae/ae.safetensors" "$NETWORK_VOLUME/models/z_image/"
                    mv "$NETWORK_VOLUME/models/z_image_turbo_temp/split_files/text_encoders/qwen_3_4b.safetensors" "$NETWORK_VOLUME/models/z_image/"
                    rm -rf "$NETWORK_VOLUME/models/z_image_turbo_temp"
                    wget -q --show-progress -O "$NETWORK_VOLUME/models/z_image/zimage_turbo_training_adapter_v2.safetensors" \
                        "https://huggingface.co/ostris/zimage_turbo_training_adapter/resolve/main/zimage_turbo_training_adapter_v2.safetensors"
                ) > "$NETWORK_VOLUME/logs/model_download.log" 2>&1 &
                MODEL_DOWNLOAD_PID=$!
                ;;
        esac
        ;;
    *)
        print_error "Unknown MODEL_TYPE: $MODEL_TYPE"
        exit 1
        ;;
esac

echo ""

# Start captioning processes if needed
if [ "$CAPTION_MODE" != "skip" ]; then
    print_header "Starting Captioning Process"
    echo ""

    # Clear any existing subfolders in dataset directories before captioning
    if [ "$CAPTION_MODE" = "images" ] || [ "$CAPTION_MODE" = "both" ]; then
        print_info "Cleaning up image dataset directory..."
        # Remove any subdirectories but keep files
        find "$NETWORK_VOLUME/image_dataset_here" -mindepth 1 -maxdepth 1 -type d -exec rm -rf {} +
        print_success "Image dataset directory cleaned"
    fi

    if [ "$CAPTION_MODE" = "videos" ] || [ "$CAPTION_MODE" = "both" ]; then
        print_info "Cleaning up video dataset directory..."
        # Remove any subdirectories but keep files
        find "$NETWORK_VOLUME/video_dataset_here" -mindepth 1 -maxdepth 1 -type d -exec rm -rf {} +
        print_success "Video dataset directory cleaned"
    fi

    echo ""

    # Start image captioning in background if needed
    if [ "$CAPTION_MODE" = "images" ] || [ "$CAPTION_MODE" = "both" ]; then
        print_info "Starting image captioning process..."
        JOY_CAPTION_SCRIPT="$NETWORK_VOLUME/Captioning/JoyCaption/JoyCaptionRunner.sh"

        if [ -f "$JOY_CAPTION_SCRIPT" ]; then
            if [ -n "$TRIGGER_WORD" ]; then
                bash "$JOY_CAPTION_SCRIPT" --trigger-word "$TRIGGER_WORD" > "$NETWORK_VOLUME/logs/image_captioning.log" 2>&1 &
            else
                bash "$JOY_CAPTION_SCRIPT" > "$NETWORK_VOLUME/logs/image_captioning.log" 2>&1 &
            fi
            IMAGE_CAPTION_PID=$!
            print_success "Image captioning started in background (PID: $IMAGE_CAPTION_PID)"

            # Wait for image captioning with progress indicator
            print_info "Waiting for image captioning to complete..., initial run can take 5-20 minutes."
            timeout_counter=0
            max_timeout=3600 # 1 hour timeout
            while kill -0 "$IMAGE_CAPTION_PID" 2> /dev/null; do
                # Check for completion first
                if tail -n 1 "$NETWORK_VOLUME/logs/image_captioning.log" 2> /dev/null | grep -q "All done!"; then
                    break
                fi
                # Check for actual errors (more specific patterns to avoid false positives)
                # Look for actual error patterns: [ERROR], Error:, Traceback, Exception:, or failed with exit code
                if tail -n 20 "$NETWORK_VOLUME/logs/image_captioning.log" 2> /dev/null | grep -qiE "(^\[ERROR\]|^Error:|^Traceback|Exception:|failed with exit)"; then
                    print_error "Image captioning encountered errors. Check log: $NETWORK_VOLUME/logs/image_captioning.log"
                    exit 1
                fi
                echo -n "."
                sleep 5
                timeout_counter=$((timeout_counter + 5))
                if [ $timeout_counter -ge $max_timeout ]; then
                    print_error "Image captioning timed out after 1 hour. Check log: $NETWORK_VOLUME/logs/image_captioning.log"
                    kill "$IMAGE_CAPTION_PID"
                    exit 1
                fi
            done
            echo ""
            # Verify captioning actually completed successfully
            wait "$IMAGE_CAPTION_PID"
            if [ $? -ne 0 ]; then
                print_error "Image captioning failed. Check log: $NETWORK_VOLUME/logs/image_captioning.log"
                exit 1
            fi
            print_success "Image captioning completed!"
        else
            print_error "JoyCaption script not found at: $JOY_CAPTION_SCRIPT"
            exit 1
        fi
    fi

    # Start video captioning if needed
    if [ "$CAPTION_MODE" = "videos" ] || [ "$CAPTION_MODE" = "both" ]; then
        print_info "Starting video captioning process..."

        if [ "$VIDEO_CAPTIONER" = "qwen" ]; then
            # ==========================================
            # QWEN2.5-VL VIDEO CAPTIONER
            # ==========================================
            QWEN_CAPTION_SCRIPT="$NETWORK_VOLUME/Captioning/qwen_captioner.sh"

            if [ -f "$QWEN_CAPTION_SCRIPT" ]; then
                if [ -n "$TRIGGER_WORD" ]; then
                    bash "$QWEN_CAPTION_SCRIPT" --trigger-word "$TRIGGER_WORD" \
                        > "$NETWORK_VOLUME/logs/video_captioning.log" 2>&1 &
                else
                    bash "$QWEN_CAPTION_SCRIPT" \
                        > "$NETWORK_VOLUME/logs/video_captioning.log" 2>&1 &
                fi
                VIDEO_CAPTION_PID=$!

                print_info "Waiting for Qwen video captioning to complete..."
                print_info "To monitor: tail -f $NETWORK_VOLUME/logs/video_captioning.log"
                timeout_counter=0
                max_timeout=7200 # 2 hour timeout
                while kill -0 "$VIDEO_CAPTION_PID" 2> /dev/null; do
                    if tail -n 1 "$NETWORK_VOLUME/logs/video_captioning.log" 2> /dev/null | grep -q "All done!"; then
                        break
                    fi
                    if tail -n 20 "$NETWORK_VOLUME/logs/video_captioning.log" 2> /dev/null | grep -qiE "(^\[ERROR\]|^Error:|^Traceback|Exception:|failed with exit)"; then
                        print_error "Qwen video captioning encountered errors. Check log: $NETWORK_VOLUME/logs/video_captioning.log"
                        exit 1
                    fi
                    echo -n "."
                    sleep 5
                    timeout_counter=$((timeout_counter + 5))
                    if [ $timeout_counter -ge $max_timeout ]; then
                        print_error "Qwen video captioning timed out after 2 hours. Check log: $NETWORK_VOLUME/logs/video_captioning.log"
                        kill "$VIDEO_CAPTION_PID"
                        exit 1
                    fi
                done
                echo ""

                wait "$VIDEO_CAPTION_PID"
                if [ $? -eq 0 ]; then
                    print_success "Qwen video captioning completed successfully"
                else
                    print_error "Qwen video captioning failed. Check log: $NETWORK_VOLUME/logs/video_captioning.log"
                    exit 1
                fi
            else
                print_error "Qwen captioning script not found at: $QWEN_CAPTION_SCRIPT"
                exit 1
            fi

        else
            # ==========================================
            # GEMINI VIDEO CAPTIONER
            # ==========================================
            VIDEO_CAPTION_SCRIPT="$NETWORK_VOLUME/Captioning/gemini_captioner.sh"

            ENV_FILE="/etc/environment"

            if [ -z "$GEMINI_API_KEY" ]; then
                if [ -f "$ENV_FILE" ]; then
                    export $(grep '^GEMINI_API_KEY=' "$ENV_FILE" | xargs 2> /dev/null || true)
                fi
            fi

            if [ -z "$GEMINI_API_KEY" ] || [ "$GEMINI_API_KEY" == "token_here" ]; then
                echo "------------------------------------------------"
                echo " GEMINI API KEY NOT FOUND"
                echo "------------------------------------------------"

                read -p "Paste your Gemini API Key: " USER_KEY

                if [ -z "$USER_KEY" ]; then
                    echo "Error: No API key provided."
                    exit 1
                fi

                export GEMINI_API_KEY="$USER_KEY"

                echo "[OK] Saving GEMINI_API_KEY for future runs..."
                sed -i '/^GEMINI_API_KEY=/d' "$ENV_FILE" 2> /dev/null || true
                echo "GEMINI_API_KEY=$USER_KEY" | tee -a "$ENV_FILE" > /dev/null
            fi

            echo "Using GEMINI_API_KEY: ${GEMINI_API_KEY:0:4}****${GEMINI_API_KEY: -4}"

            if [ -f "$VIDEO_CAPTION_SCRIPT" ]; then
                bash "$VIDEO_CAPTION_SCRIPT" \
                    --mode videos \
                    --trigger-word "$TRIGGER_WORD" \
                    > "$NETWORK_VOLUME/logs/video_captioning.log" 2>&1 &
                VIDEO_CAPTION_PID=$!

                print_info "Waiting for Gemini video captioning to complete..."
                timeout_counter=0
                max_timeout=7200
                while kill -0 "$VIDEO_CAPTION_PID" 2> /dev/null; do
                    if tail -n 1 "$NETWORK_VOLUME/logs/video_captioning.log" 2> /dev/null | grep -q "\[GEMINI_DONE\]"; then
                        break
                    fi
                    if tail -n 20 "$NETWORK_VOLUME/logs/video_captioning.log" 2> /dev/null | grep -qiE "(^\[ERROR\]|^Error:|^Traceback|Exception:|failed with exit)"; then
                        print_error "Video captioning encountered errors. Check log: $NETWORK_VOLUME/logs/video_captioning.log"
                        exit 1
                    fi
                    echo -n "."
                    sleep 2
                    timeout_counter=$((timeout_counter + 2))
                    if [ $timeout_counter -ge $max_timeout ]; then
                        print_error "Video captioning timed out after 2 hours. Check log: $NETWORK_VOLUME/logs/video_captioning.log"
                        exit 1
                    fi
                done
                echo ""

                wait "$VIDEO_CAPTION_PID"
                if [ $? -eq 0 ]; then
                    print_success "Gemini video captioning completed successfully"
                else
                    print_error "Gemini video captioning failed. Check log: $NETWORK_VOLUME/logs/video_captioning.log"
                    exit 1
                fi
            else
                print_error "Gemini captioning script not found at: $VIDEO_CAPTION_SCRIPT"
                exit 1
            fi
        fi
    fi

    echo ""
fi

# Wait for model download to complete
if [ -n "$MODEL_DOWNLOAD_PID" ]; then
    print_header "Finalizing Model Download"
    echo ""
    print_info "Waiting for model download to complete..."
    print_info "To view model download progress, open a new terminal window and paste:"
    echo "  tail -f $NETWORK_VOLUME/logs/model_download.log"
    echo ""
    timeout_counter=0
    max_timeout=10800 # 3 hour timeout for large models
    while kill -0 "$MODEL_DOWNLOAD_PID" 2> /dev/null; do
        # Check for errors in log
        if tail -n 20 "$NETWORK_VOLUME/logs/model_download.log" 2> /dev/null | grep -qi "error\|failed\|exception\|unauthorized\|403\|404"; then
            print_error "Model download encountered errors. Check log: $NETWORK_VOLUME/logs/model_download.log"
            kill "$MODEL_DOWNLOAD_PID" 2> /dev/null || true
            exit 1
        fi
        echo -n "."
        sleep 3
        timeout_counter=$((timeout_counter + 3))
        if [ $timeout_counter -ge $max_timeout ]; then
            print_error "Model download timed out after 3 hours. Check log: $NETWORK_VOLUME/logs/model_download.log"
            kill "$MODEL_DOWNLOAD_PID" 2> /dev/null || true
            exit 1
        fi
    done
    echo ""
    wait "$MODEL_DOWNLOAD_PID"
    download_exit_code=$?

    if [ $download_exit_code -ne 0 ]; then
        print_error "Model download failed with exit code $download_exit_code. Check log: $NETWORK_VOLUME/logs/model_download.log"
        exit 1
    fi

    # Verify model files actually exist based on MODEL_TYPE
    print_info "Verifying model download..."
    case $MODEL_TYPE in
        "flux")
            if [ ! -f "$NETWORK_VOLUME/models/flux/flux1-dev.safetensors" ] && [ ! -d "$NETWORK_VOLUME/models/flux" ]; then
                print_error "Flux model files not found after download. Check log: $NETWORK_VOLUME/logs/model_download.log"
                exit 1
            fi
            ;;
        "sdxl")
            if [ ! -f "$NETWORK_VOLUME/models/sdXL_v10VAEFix.safetensors" ]; then
                print_error "SDXL model file not found after download. Check log: $NETWORK_VOLUME/logs/model_download.log"
                exit 1
            fi
            ;;
        "wan13")
            if [ ! -d "$NETWORK_VOLUME/models/Wan/Wan2.1-T2V-1.3B" ] || [ -z "$(ls -A "$NETWORK_VOLUME/models/Wan/Wan2.1-T2V-1.3B" 2> /dev/null)" ]; then
                print_error "Wan 1.3B model files not found after download. Check log: $NETWORK_VOLUME/logs/model_download.log"
                exit 1
            fi
            ;;
        "wan14b_t2v")
            if [ ! -d "$NETWORK_VOLUME/models/Wan/Wan2.1-T2V-14B" ] || [ -z "$(ls -A "$NETWORK_VOLUME/models/Wan/Wan2.1-T2V-14B" 2> /dev/null)" ]; then
                print_error "Wan 14B T2V model files not found after download. Check log: $NETWORK_VOLUME/logs/model_download.log"
                exit 1
            fi
            ;;
        "wan14b_i2v")
            if [ ! -d "$NETWORK_VOLUME/models/Wan/Wan2.1-I2V-14B-480P" ] || [ -z "$(ls -A "$NETWORK_VOLUME/models/Wan/Wan2.1-I2V-14B-480P" 2> /dev/null)" ]; then
                print_error "Wan 14B I2V model files not found after download. Check log: $NETWORK_VOLUME/logs/model_download.log"
                exit 1
            fi
            ;;
        "qwen")
            if [ ! -d "$NETWORK_VOLUME/models/Qwen-Image" ] || [ -z "$(ls -A "$NETWORK_VOLUME/models/Qwen-Image" 2> /dev/null)" ]; then
                print_error "Qwen Image model files not found after download. Check log: $NETWORK_VOLUME/logs/model_download.log"
                exit 1
            fi
            ;;
        "z_image_turbo")
            missing_files=""
            if [ ! -f "$NETWORK_VOLUME/models/z_image/z_image_turbo_bf16.safetensors" ]; then
                missing_files="$missing_files z_image_turbo_bf16.safetensors"
            fi
            if [ ! -f "$NETWORK_VOLUME/models/z_image/ae.safetensors" ]; then
                missing_files="$missing_files ae.safetensors"
            fi
            if [ ! -f "$NETWORK_VOLUME/models/z_image/qwen_3_4b.safetensors" ]; then
                missing_files="$missing_files qwen_3_4b.safetensors"
            fi
            if [ ! -f "$NETWORK_VOLUME/models/z_image/zimage_turbo_training_adapter_v2.safetensors" ]; then
                missing_files="$missing_files zimage_turbo_training_adapter_v2.safetensors"
            fi
            if [ -n "$missing_files" ]; then
                print_error "Z Image Turbo model files missing after download:$missing_files"
                print_error "Check log: $NETWORK_VOLUME/logs/model_download.log"
                exit 1
            fi
            ;;
    esac
    print_success "Model download completed and verified!"
    echo ""
fi

# Update dataset.toml file with actual paths and video config
print_header "Configuring Dataset"
echo ""

# Defining toml variables
DATASET_TOML="$NETWORK_VOLUME/diffusion_pipe/examples/dataset.toml"
MODEL_TOML="$NETWORK_VOLUME/diffusion_pipe/examples/$TOML_FILE"

if [ -f "$DATASET_TOML" ]; then
    print_info "Updating dataset.toml with actual paths..."

    # Replace $NETWORK_VOLUME with actual path in image directory
    sed -i "s|\$NETWORK_VOLUME/image_dataset_here|$NETWORK_VOLUME/image_dataset_here|g" "$DATASET_TOML" 2> /dev/null || print_warning "Failed to update image directory path in dataset.toml"

    # Replace $NETWORK_VOLUME with actual path in video directory (even if commented)
    sed -i "s|\$NETWORK_VOLUME/video_dataset_here|$NETWORK_VOLUME/video_dataset_here|g" "$DATASET_TOML" 2> /dev/null || print_warning "Failed to update video directory path in dataset.toml"

    # Uncomment video dataset section if user wants to caption videos
    if [ "$CAPTION_MODE" = "videos" ] || [ "$CAPTION_MODE" = "both" ]; then
        print_info "Enabling video dataset in configuration..."
        # Uncomment the video directory section
        sed -i '/# \[\[directory\]\]/,/# num_repeats = 5/ s/^# //' "$DATASET_TOML" 2> /dev/null
        # Verify uncommenting worked by checking if video directory section exists uncommented
        if ! grep -q "^\[\[directory\]\]" "$DATASET_TOML" || [ -z "$(grep -A2 "^\[\[directory\]\]" "$DATASET_TOML" | grep -m1 "video_dataset_here")" ]; then
            # Check if there's a commented video section that wasn't uncommented
            if grep -q "# path = '\$NETWORK_VOLUME/video_dataset_here'" "$DATASET_TOML"; then
                print_warning "Video dataset section may not have been uncommented correctly. Please check dataset.toml manually."
            fi
        fi
    fi

    print_success "Dataset configuration updated"
else
    print_warning "dataset.toml not found at $DATASET_TOML"
fi

echo ""

# ---------------------------------------
# Helper functions
# ---------------------------------------

get_toml_value() {
    local key="$1"
    local file="$2"

    grep -E "^${key}[[:space:]]*=" "$file" \
        | head -n1 \
        | sed -E "s/^${key}[[:space:]]*=[[:space:]]*//;s/[\"']//g"
}

default_if_empty() {
    local value="$1"
    local default="$2"
    [ -z "$value" ] && echo "$default" || echo "$value"
}

read_training_config() {
    RESOLUTION=$(grep -E "^[[:space:]]*resolutions" "$DATASET_TOML" | sed -E 's/.*\[([0-9]+).*/\1/')
    [ -z "$RESOLUTION" ] && RESOLUTION="1024 (default)"

    EPOCHS=$(get_toml_value "epochs" "$MODEL_TOML")
    SAVE_EVERY=$(get_toml_value "save_every_n_epochs" "$MODEL_TOML")
    RANK=$(get_toml_value "rank" "$MODEL_TOML")
    LR=$(get_toml_value "lr" "$MODEL_TOML")

    OPTIMIZER_TYPE=$(awk '
        /\[optimizer\]/ {found=1; next}
        /^\[/ {found=0}
        found && /type[[:space:]]*=/ {
            gsub(/["'\'' ]/, "", $3)
            print $3
            exit
        }
    ' "$MODEL_TOML")

    EPOCHS=$(default_if_empty "$EPOCHS" "1000 (default)")
    SAVE_EVERY=$(default_if_empty "$SAVE_EVERY" "2 (default)")
    RANK=$(default_if_empty "$RANK" "32 (default)")
    LR=$(default_if_empty "$LR" "2e-5 (default)")
    OPTIMIZER_TYPE=$(default_if_empty "$OPTIMIZER_TYPE" "adamw_optimi (default)")
}

display_training_config() {
    print_header "Training Configuration Summary"
    echo ""
    echo -e "${BOLD}Model:${NC} $MODEL_NAME"
    if [[ "$RESOLUTION" =~ ^[0-9]+$ ]]; then
        echo -e "${BOLD}Resolution:${NC} ${RESOLUTION}x${RESOLUTION}"
    else
        echo -e "${BOLD}Resolution:${NC} ${RESOLUTION}"
    fi
    echo ""
    echo -e "${BOLD}Updated Training Parameters:${NC}"
    echo "  📊 Epochs: $EPOCHS"
    echo "  💾 Save Every: $SAVE_EVERY epochs"
    echo "  🎛️  LoRA Rank: $RANK"
    echo "  📈 Learning Rate: $LR"
    echo "  ⚙️  Optimizer: $OPTIMIZER_TYPE"
    echo ""
}

# Display training configuration summary
read_training_config
display_training_config

# Show dataset paths and repeats
if [ "$CAPTION_MODE" != "skip" ]; then
    echo -e "${BOLD}Dataset Configuration:${NC}"

    # Always show image dataset info
    if [ "$CAPTION_MODE" = "images" ] || [ "$CAPTION_MODE" = "both" ]; then
        IMAGE_COUNT=$(find "$NETWORK_VOLUME/image_dataset_here" -maxdepth 1 -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.bmp" -o -iname "*.gif" -o -iname "*.tiff" -o -iname "*.webp" \) | wc -l)
        echo "  📷 Images: $NETWORK_VOLUME/image_dataset_here ($IMAGE_COUNT files)"
        echo "     Repeats: 1 per epoch"
    fi

    # Show video dataset info if applicable
    if [ "$CAPTION_MODE" = "videos" ] || [ "$CAPTION_MODE" = "both" ]; then
        VIDEO_COUNT=$(find "$NETWORK_VOLUME/video_dataset_here" -maxdepth 1 -type f \( -iname "*.mp4" -o -iname "*.avi" -o -iname "*.mov" -o -iname "*.mkv" -o -iname "*.webm" \) | wc -l)
        echo "  🎬 Videos: $NETWORK_VOLUME/video_dataset_here ($VIDEO_COUNT files)"
        echo "     Repeats: 5 per epoch"
    fi
else
    echo -e "${BOLD}Dataset:${NC} Using existing captions"
fi

if [ "$MODEL_TYPE" = "flux" ]; then
    echo -e "${BOLD}Hugging Face Token:${NC} Set ✓"
fi

if [ "$CAPTION_MODE" = "videos" ] || [ "$CAPTION_MODE" = "both" ]; then
    if [ "$VIDEO_CAPTIONER" = "gemini" ]; then
        echo -e "${BOLD}Video Captioner:${NC} Gemini | API Key: Set ✓"
    else
        echo -e "${BOLD}Video Captioner:${NC} Qwen2.5-VL (local)"
    fi
fi

echo ""

# Prompt user about configuration files
print_header "Training Configuration"
echo ""

print_info "Before starting training, you can modify the default training parameters in these files:"
echo ""
echo -e "${BOLD}1. Model Configuration:${NC}"
echo "   $NETWORK_VOLUME/diffusion_pipe/examples/$TOML_FILE"
echo ""
echo -e "${BOLD}2. Dataset Configuration:${NC}"
echo "   $NETWORK_VOLUME/diffusion_pipe/examples/dataset.toml"
echo ""

print_warning "These files contain important settings like:"
echo "  • Learning rate, batch size, epochs"
echo "  • Dataset paths and image/video resolutions"
echo "  • LoRA rank and other adapter settings"
echo ""

echo -e "${YELLOW}Would you like to modify these files before starting training?${NC}"
echo "1) Continue with default settings"
echo "2) Pause here - I'll modify the files manually"
echo ""

while true; do
    read -p "Enter your choice (1-2): " config_choice
    case $config_choice in
        1)
            print_success "Continuing with default training settings..."
            break
            ;;
        2)
            print_info "Training paused for manual configuration."
            echo ""
            echo -e "${BOLD}Configuration Files:${NC}"
            echo "1. Model settings: $NETWORK_VOLUME/diffusion_pipe/examples/$TOML_FILE"
            echo "2. Dataset settings: $NETWORK_VOLUME/diffusion_pipe/examples/dataset.toml"
            echo ""
            print_warning "Please modify these files as needed, then return here to continue."
            echo ""

            while true; do
                read -p "Have you finished configuring the settings? (yes/no): " config_done
                case $config_done in
                    yes | YES | y | Y)
                        print_success "Configuration completed. Reading updated settings..."
                        echo ""
                        read_training_config

                        # Display updated configuration for confirmation
                        display_training_config

                        while true; do
                            read -p "Do these updated settings look correct? (yes/no): " settings_confirm
                            case $settings_confirm in
                                yes | YES | y | Y)
                                    print_success "Settings confirmed. Proceeding with training..."
                                    break 2 # Break out of both loops
                                    ;;
                                no | NO | n | N)
                                    print_info "Please modify the configuration files again."
                                    echo ""
                                    break # Go back to configuration loop
                                    ;;
                                *)
                                    print_error "Please enter 'yes' or 'no'."
                                    ;;
                            esac
                        done
                        ;;
                    no | NO | n | N)
                        print_info "Take your time configuring the settings."
                        ;;
                    *)
                        print_error "Please enter 'yes' or 'no'."
                        ;;
                esac
            done
            break
            ;;
        *)
            print_error "Invalid choice. Please enter 1 or 2."
            ;;
    esac
done

echo ""

# Check if image captioning is still running
if [ "$CAPTION_MODE" = "images" ] || [ "$CAPTION_MODE" = "both" ]; then
    # Image captioning was already handled in the captioning section above
    # No need to check again here

    # Prompt user to inspect image captions
    print_header "Caption Inspection"
    echo ""
    print_info "Please manually inspect the generated captions in:"
    echo "  $NETWORK_VOLUME/image_dataset_here"
    echo ""
    print_warning "Check that the captions are accurate and appropriate for your training data."
    echo ""

    while true; do
        read -p "Have you reviewed the image captions and are ready to proceed? (yes/no): " inspect_choice
        case $inspect_choice in
            yes | YES | y | Y)
                print_success "Image captions approved. Proceeding to training..."
                break
                ;;
            no | NO | n | N)
                print_info "Please review the captions and run this script again when ready."
                exit 0
                ;;
            *)
                print_error "Please enter 'yes' or 'no'."
                ;;
        esac
    done
    echo ""
fi

# Check video captions if applicable
if [ "$CAPTION_MODE" = "videos" ] || [ "$CAPTION_MODE" = "both" ]; then
    # Video captioning was already handled in the captioning section above
    # No need to check again here

    print_header "Video Caption Inspection"
    echo ""
    print_info "Please manually inspect the generated video captions in:"
    echo "  $NETWORK_VOLUME/video_dataset_here"
    echo ""
    print_warning "Check that the video captions are accurate and appropriate for your training data."
    echo ""

    while true; do
        read -p "Have you reviewed the video captions and are ready to proceed? (yes/no): " video_inspect_choice
        case $video_inspect_choice in
            yes | YES | y | Y)
                print_success "Video captions approved. Proceeding to training..."
                break
                ;;
            no | NO | n | N)
                print_info "Please review the captions and run this script again when ready."
                exit 0
                ;;
            *)
                print_error "Please enter 'yes' or 'no'."
                ;;
        esac
    done
    echo ""
fi

# Start training
print_header "Starting Training"
echo ""

print_info "Changing to diffusion_pipe directory..."
cd "$NETWORK_VOLUME/diffusion_pipe"

echo ""

print_info "Starting LoRA training with $MODEL_NAME..."
print_info "Using configuration: examples/$TOML_FILE"
echo ""

# Add special warning for Qwen Image model initialization
if [ "$MODEL_TYPE" = "qwen" ]; then
    print_warning "⚠️  IMPORTANT: Qwen Image model initialization can take several minutes."
    print_warning "⚠️  The script may appear to hang during initialization - this is NORMAL."
    print_warning "⚠️  As long as the script doesn't exit with an error, let it run."
    echo ""
    print_info "Waiting 10 seconds for you to read this message..."
    sleep 10
    echo ""
fi

# Add special warning for Z Image Turbo model initialization
if [ "$MODEL_TYPE" = "z_image_turbo" ]; then
    print_warning "⚠️  IMPORTANT: Z Image Turbo model initialization can take several minutes."
    print_warning "⚠️  The script may appear to hang during initialization - this is NORMAL."
    print_warning "⚠️  As long as the script doesn't exit with an error, let it run."
    echo ""
    print_info "Waiting 10 seconds for you to read this message..."
    sleep 10
    echo ""
fi

# TensorBoard
TENSORBOARD_FOLDER="$NETWORK_VOLUME/output_folder"
print_info "TensorBoard logs for this run are located at:\n$TENSORBOARD_FOLDER\n"

echo "Access TensorBoard:"
echo "  Local:  http://localhost:6006"
echo ""

echo "If using SSH tunneling:"
echo "  ssh -p <SSH_PORT> hostname@<SERVER_IP> -L 6006:localhost:6006"
echo "  Then open: http://localhost:6006"
echo ""

print_warning "Training is starting. This may take several hours depending on your dataset size and model."
print_info "You can monitor progress in the console output below."
echo ""

# Start training with the appropriate TOML file
NCCL_P2P_DISABLE="1" NCCL_IB_DISABLE="1" deepspeed --num_gpus=1 train.py --deepspeed --config "examples/$TOML_FILE"

print_success "Training completed!"
