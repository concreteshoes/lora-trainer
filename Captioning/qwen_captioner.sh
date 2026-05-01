#!/bin/bash
set -e

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/qwen_caption_env"
REQUIREMENTS_FILE="$SCRIPT_DIR/qwen_caption_requirements.txt"

# Repo is copied into the image — no cloning needed
CAPTIONER_REPO_DIR="$SCRIPT_DIR/Qwen2.5-VL-Video-Captioning"
CAPTIONER_SCRIPT="$CAPTIONER_REPO_DIR/Qwen2.5-vl-captioner_v3.py"
REFINEMENT_SCRIPT="$CAPTIONER_REPO_DIR/qwen2.5_caption_refinement.py"
DEFAULT_CAPTION_CONFIG="$CAPTIONER_REPO_DIR/config/captioning-config.toml"
DEFAULT_REFINEMENT_CONFIG="$CAPTIONER_REPO_DIR/config/refinement-config.toml"

# Video dataset directory — relies solely on $NETWORK_VOLUME (set by start.sh)
detect_default_video_dir() {
    if [[ -n "$NETWORK_VOLUME" && -d "$NETWORK_VOLUME/video_dataset_here" ]]; then
        echo "$NETWORK_VOLUME/video_dataset_here"
    fi
}

DEFAULT_VIDEO_DIR=$(detect_default_video_dir)

# Inject $NETWORK_VOLUME into the captioning config at runtime.
# Mirrors the pattern used by the diffusion-pipe wrapper — keeps start.sh clean.
QWEN_CAPTION_CONFIG="/Captioning/Qwen2.5-VL-Video-Captioning/config/captioning-config.toml"
QWEN_REFINEMENT_CONFIG="/Captioning/Qwen2.5-VL-Video-Captioning/config/refinement-config.toml"

if [[ -f "$QWEN_CAPTION_CONFIG" ]]; then
    # input_path — where videos are read from
    sed -i "s|input_path = \"/path/to/input_folder\"|input_path = \"${NETWORK_VOLUME}/video_dataset_here\"|" "$QWEN_CAPTION_CONFIG"

    # output_dir — where caption files are written (alongside the videos)
    sed -i "s|output_dir = \"/path/to/output/folder\"|output_dir = \"${NETWORK_VOLUME}/video_dataset_here\"|" "$QWEN_CAPTION_CONFIG"

    # HF_TOKEN — required for gated Qwen models on HuggingFace
    if [[ -n "$HF_TOKEN" ]]; then
        sed -i "s|hf_token = \"HF_Token\"|hf_token = \"${HF_TOKEN}\"|" "$QWEN_CAPTION_CONFIG"
        echo "Qwen caption config patched: NETWORK_VOLUME and HF_TOKEN injected"
    else
        echo "WARNING: HF_TOKEN is not set — model download may fail for gated models"
        echo "Qwen caption config patched: NETWORK_VOLUME injected (HF_TOKEN skipped)"
    fi
fi

if [[ -f "$QWEN_REFINEMENT_CONFIG" ]]; then
    # txt file processing paths
    sed -i "s|input_dir = \"/path/to/input_folder\"|input_dir = \"${NETWORK_VOLUME}/video_dataset_here\"|" "$QWEN_REFINEMENT_CONFIG"
    sed -i "s|output_dir = \"/path/to/output_folder/refined\"|output_dir = \"${NETWORK_VOLUME}/video_dataset_here/refined\"|" "$QWEN_REFINEMENT_CONFIG"

    # CSV processing paths (if using csv mode instead)
    sed -i "s|input_csv = \"/path/to/video_captions.csv\"|input_csv = \"${NETWORK_VOLUME}/video_dataset_here/video_captions.csv\"|" "$QWEN_REFINEMENT_CONFIG"
    sed -i "s|output_csv = \"/path/to/video_captions_refined.csv\"|output_csv = \"${NETWORK_VOLUME}/video_dataset_here/video_captions_refined.csv\"|" "$QWEN_REFINEMENT_CONFIG"

    echo "Qwen refinement config patched: NETWORK_VOLUME injected"
fi

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

log_info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $1"; }
log_step()    { echo -e "${CYAN}[STEP]${NC} $1"; }

show_help() {
    echo "Qwen2.5-VL Video Captioning Wrapper"
    echo ""
    echo "Mode is detected automatically:"
    echo "  No .txt files in video_dataset_here  →  runs captioner"
    echo "  .txt files already present            →  runs refinement"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --trigger-word WORD      Prepend trigger word to each .txt caption after captioning"
    echo "                           Only applies when captioning runs (no prior .txt files)"
    echo "                           Only applies to output_format = \"individual\" in the config"
    echo "  --config FILE            Override the TOML config path for whichever mode runs"
    echo "  --setup-only             Setup venv only, do not run"
    echo "  --force-reinstall        Force reinstall of venv packages"
    echo "  -h, --help               Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                                       # Auto-detect and run"
    echo "  $0 --trigger-word mytoken                # Caption + prepend trigger word"
    echo "  $0 --config /path/to/custom.toml         # Override config"
    echo "  $0 --setup-only                          # Env setup only"
    echo ""
    if [[ -n "$DEFAULT_VIDEO_DIR" ]]; then
        echo "Video dataset: $DEFAULT_VIDEO_DIR"
    else
        echo "Warning: \$NETWORK_VOLUME is not set or video_dataset_here does not exist."
        echo "         Verify input_path in the captioning config."
    fi
}

check_python() {
    if command -v python3 &>/dev/null; then
        PYTHON_CMD="python3"
    elif command -v python &>/dev/null; then
        PYTHON_CMD="python"
    else
        log_error "Python not found in PATH"
        exit 1
    fi

    if ! $PYTHON_CMD -c "import sys; exit(0 if sys.version_info >= (3, 9) else 1)" 2>/dev/null; then
        PYTHON_VERSION=$($PYTHON_CMD -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
        log_error "Python 3.9+ required. Found: $PYTHON_VERSION"
        exit 1
    fi

    log_info "Python: $($PYTHON_CMD --version)"
}

create_requirements() {
    log_info "Writing requirements (system packages reused via --system-site-packages)..."
    cat > "$REQUIREMENTS_FILE" << 'EOF'
# -----------------------------------------------------------------------
# Inherited from system image via --system-site-packages (NOT re-downloaded):
#   torch, torchvision               — Dockerfile Layer 2
#   transformers, accelerate         — Dockerfile Layer 3
#   bitsandbytes, safetensors        — Dockerfile Layer 3
#   sentencepiece, protobuf          — Dockerfile Layer 3
#   pillow, tqdm, opencv-python      — Dockerfile Layer 3 / Layer 6
#   sageattention                    — compiled at runtime in start.sh
#
# Only packages absent from the system image:
# -----------------------------------------------------------------------

# Qwen VL utilities + decord for video frame extraction
qwen-vl-utils[decord]==0.0.8

# pandas — used for CSV result logging, not present in system image
pandas

# tomli — TOML parser; Python 3.12 has tomllib built-in but the script
# may import tomli by name. Tiny install, safe to include.
tomli
EOF
    log_success "Requirements file written"
}

setup_venv() {
    log_step "Setting up virtual environment..."

    if [[ "$FORCE_REINSTALL" == "true" && -d "$VENV_DIR" ]]; then
        log_warning "Removing existing venv for fresh install..."
        rm -rf "$VENV_DIR"
    fi

    if [[ -d "$VENV_DIR" && ! -f "$VENV_DIR/bin/activate" ]]; then
        log_warning "Venv appears corrupted, removing..."
        rm -rf "$VENV_DIR"
    fi

    if [[ ! -d "$VENV_DIR" ]]; then
        log_info "Creating venv with --system-site-packages at $VENV_DIR"
        # --system-site-packages inherits torch, transformers, bitsandbytes,
        # sageattention, and all other system packages from the image.
        $PYTHON_CMD -m venv "$VENV_DIR" --system-site-packages

        if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
            log_error "Venv creation failed — activate script missing"
            exit 1
        fi
        log_success "Venv created"
    else
        log_info "Venv already exists"
    fi

    source "$VENV_DIR/bin/activate"

    log_info "Upgrading pip..."
    pip install --upgrade pip --quiet

    if [[ "$FORCE_REINSTALL" == "true" ]]; then
        log_info "Force reinstalling requirements..."
        pip install --force-reinstall -r "$REQUIREMENTS_FILE"
    else
        log_info "Installing requirements (skipping already-satisfied)..."
        pip install -r "$REQUIREMENTS_FILE"
    fi

    log_success "Venv setup complete"
}

check_cuda() {
    log_info "Verifying CUDA availability..."
    "$VENV_DIR/bin/python" << 'PYTHON_EOF'
import sys, torch
if not torch.cuda.is_available():
    print("ERROR: CUDA not available to torch")
    sys.exit(1)
x = torch.randn(1, device='cuda') * 2
print(f"CUDA OK: {torch.cuda.get_device_name(0)} | torch {torch.__version__}")
PYTHON_EOF
}

check_sageattention() {
    log_info "Checking SageAttention availability..."
    if "$VENV_DIR/bin/python" -c "import sageattention" 2>/dev/null; then
        log_success "SageAttention available via system-site-packages"
    else
        log_warning "SageAttention not found — ensure start.sh has completed before running"
        log_warning "Captioner will fall back to standard attention"
    fi
}

check_network_volume() {
    if [[ -z "$NETWORK_VOLUME" ]]; then
        log_warning "\$NETWORK_VOLUME is not set — config input_path was not patched by start.sh"
        log_warning "Verify input_path in: $DEFAULT_CAPTION_CONFIG"
    elif [[ ! -d "$NETWORK_VOLUME/video_dataset_here" ]]; then
        log_warning "Directory not found: $NETWORK_VOLUME/video_dataset_here"
        log_warning "Create it or update input_path in: $DEFAULT_CAPTION_CONFIG"
    else
        log_info "Video dataset: $NETWORK_VOLUME/video_dataset_here"
    fi
}

run_captioner() {
    local config_file="${CONFIG_FILE:-$DEFAULT_CAPTION_CONFIG}"

    if [[ ! -f "$CAPTIONER_SCRIPT" ]]; then
        log_error "Captioner script not found: $CAPTIONER_SCRIPT"
        exit 1
    fi

    if [[ ! -f "$config_file" ]]; then
        log_error "Config not found: $config_file"
        exit 1
    fi

    log_step "Running Qwen2.5-VL video captioner..."
    log_info "Config: $config_file"

    source "$VENV_DIR/bin/activate"
    cd "$CAPTIONER_REPO_DIR"
    python "$CAPTIONER_SCRIPT" --config "$config_file"

    log_success "Captioning complete"
}

run_refinement() {
    local config_file="${CONFIG_FILE:-$DEFAULT_REFINEMENT_CONFIG}"

    if [[ ! -f "$REFINEMENT_SCRIPT" ]]; then
        log_error "Refinement script not found: $REFINEMENT_SCRIPT"
        exit 1
    fi

    if [[ ! -f "$config_file" ]]; then
        log_error "Config not found: $config_file"
        exit 1
    fi

    log_step "Running Qwen2.5 caption refinement..."
    log_info "Config: $config_file"

    source "$VENV_DIR/bin/activate"
    cd "$CAPTIONER_REPO_DIR"
    python "$REFINEMENT_SCRIPT" --config "$config_file"

    log_success "Refinement complete"
}

prepend_trigger_word() {
    local trigger_word="$1"
    local output_dir="${NETWORK_VOLUME}/video_dataset_here"

    log_step "Prepending trigger word '$trigger_word' to caption files..."

    local count=0
    local skipped=0

    while IFS= read -r -d '' txt_file; do
        local content
        content=$(cat "$txt_file")

        # Skip if trigger word is already prepended (idempotent)
        if [[ "$content" == "${trigger_word},"* ]]; then
            (( skipped++ )) || true
            continue
        fi

        echo "${trigger_word}, ${content}" > "$txt_file"
        (( count++ )) || true
    done < <(find "$output_dir" -maxdepth 1 -name "*.txt" -print0)

    if [[ $count -gt 0 ]]; then
        log_success "Trigger word prepended to $count file(s)"
    fi
    if [[ $skipped -gt 0 ]]; then
        log_info "$skipped file(s) already had trigger word — skipped"
    fi
    if [[ $count -eq 0 && $skipped -eq 0 ]]; then
        log_warning "No .txt caption files found in $output_dir"
        log_warning "Trigger word only applies to output_format = \"individual\" in the config"
    fi
}

# --- Argument parsing ---
SETUP_ONLY=false
FORCE_REINSTALL=false
CONFIG_FILE=""
TRIGGER_WORD=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            shift
            CONFIG_FILE="$1"
            shift
            ;;
        --trigger-word)
            shift
            TRIGGER_WORD="$1"
            shift
            ;;
        --setup-only)
            SETUP_ONLY=true
            shift
            ;;
        --force-reinstall)
            FORCE_REINSTALL=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            log_error "Unknown argument: $1"
            show_help
            exit 1
            ;;
    esac
done

# --- Main ---
main() {
    log_info "Qwen2.5-VL Video Captioning Wrapper"

    check_python
    check_network_volume
    create_requirements
    setup_venv
    check_cuda
    check_sageattention

    if [[ "$SETUP_ONLY" == "true" ]]; then
        log_success "Setup complete."
        log_info "Caption config:    $DEFAULT_CAPTION_CONFIG"
        log_info "Refinement config: $DEFAULT_REFINEMENT_CONFIG"
        exit 0
    fi

    # Auto-detect mode based on whether .txt captions already exist in the dataset folder
    local video_dir="${NETWORK_VOLUME}/video_dataset_here"
    local txt_count
    txt_count=$(find "$video_dir" -maxdepth 1 -name "*.txt" 2>/dev/null | wc -l)

    if [[ "$txt_count" -eq 0 ]]; then
        log_info "No .txt files found in $video_dir — starting captioning pass"
        run_captioner

        if [[ -n "$TRIGGER_WORD" ]]; then
            prepend_trigger_word "$TRIGGER_WORD"
        fi
    else
        log_info "$txt_count .txt file(s) found in $video_dir — starting refinement pass"
        log_warning "Refinement rewrites captions using a text-only model (no video access)."
        log_warning "Review outputs carefully before training."
        run_refinement

        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "  Refined captions written to:"
        echo "    ${video_dir}/refined/"
        echo ""
        echo "  Next steps:"
        echo "  1. Open JupyterLab and inspect the refined .txt files"
        echo "     alongside the originals in video_dataset_here/"
        echo ""
        echo "  2. For files you want to keep, move them with drag-and-drop"
        echo "     in JupyterLab, or use:"
        echo "       mv ${video_dir}/refined/filename.txt ${video_dir}/filename.txt"
        echo ""
        echo "  3. To promote all refined captions at once:"
        echo "       mv ${video_dir}/refined/*.txt ${video_dir}/"
        echo ""
        echo "  4. Remove the refined/ subdirectory before training:"
        echo "       rm -rf ${video_dir}/refined/"
        echo "     (subdirectories in the dataset folder will break training)"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo ""

        if [[ -n "$TRIGGER_WORD" ]]; then
            log_warning "--trigger-word is ignored during refinement — apply it after promoting refined captions by re-running the wrapper with your trigger word on a fresh caption run, or prepend manually."
        fi
    fi

    fuser -v /dev/nvidia* 2>/dev/null || log_info "GPU VRAM is clear"
    log_success "All done!"
}

main "$@"
