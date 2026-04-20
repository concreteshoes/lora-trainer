#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/joy_caption_env"
PYTHON_SCRIPT="$SCRIPT_DIR/joy_caption_batch.py"
REQUIREMENTS_FILE="$SCRIPT_DIR/requirements.txt"

# Default image directory detection
detect_default_image_dir() {
    # First check if NETWORK_VOLUME is set
    if [[ -n "$NETWORK_VOLUME" && -d "$NETWORK_VOLUME/image_dataset_here" ]]; then
        echo "$NETWORK_VOLUME/image_dataset_here"
    # Check for workspace volume
    elif [[ -d "/workspace/diffusion_pipe_working_folder/image_dataset_here" ]]; then
        echo "/workspace/diffusion_pipe_working_folder/image_dataset_here"
    # Check for local volume
    elif [[ -d "/diffusion_pipe_working_folder/image_dataset_here" ]]; then
        echo "/diffusion_pipe_working_folder/image_dataset_here"
    # Fallback to current directory
    else
        echo "."
    fi
}

DEFAULT_IMAGE_DIR=$(detect_default_image_dir)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to show usage
show_help() {
    echo "Joy Caption Batch Processing Wrapper"
    echo ""
    echo "This script sets up a virtual environment and runs the Joy Caption batch processor."
    echo ""
    echo "Usage: $0 [INPUT_DIR] [OPTIONS]"
    echo ""
    echo "Optional:"
    echo "  INPUT_DIR                    Directory containing images to process"
    echo "                              (default: $DEFAULT_IMAGE_DIR)"
    echo ""
    echo "Options:"
    echo "  --output-dir DIR             Directory to save caption files (default: same as input)"
    echo "  --prompt TEXT                Caption generation prompt (default: Write a descriptive caption for this image in a casual tone within 50 words. Do NOT mention any text that is in the image."
    echo "  --trigger-word WORD          Trigger word to prepend to captions (e.g., 'Alice' -> 'Alice, <caption>')"
    echo "  --no-skip-existing           Process all images even if caption files already exist"
    echo "  --setup-only                 Only setup the environment, don't run captioning"
    echo "  --force-reinstall            Force reinstall of all requirements"
    echo "  -h, --help                   Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                           # Use default image directory: $DEFAULT_IMAGE_DIR"
    echo "  $0 /path/to/images"
    echo "  $0 --trigger-word 'claude' --output-dir /path/to/captions"
    echo "  $0 /path/to/images --prompt 'Describe this image in detail.' --timeout 10"
    echo "  $0 --setup-only              # Just setup the environment"
    echo ""
    echo "Default image directory detection:"
    echo "  1. \$NETWORK_VOLUME/image_dataset_here (if NETWORK_VOLUME is set)"
    echo "  2. /workspace/diffusion_pipe_working_folder/image_dataset_here"
    echo "  3. /diffusion_pipe_working_folder/image_dataset_here"
    echo "  4. Current directory (.)"
}

# Function to install system dependencies
install_system_deps() {
    log_info "Checking system dependencies..."

    # Check if we need to install python3-venv
    if ! dpkg -l | grep -q python3-venv 2> /dev/null; then
        log_info "Installing required system packages..."

        # Update package list
        if command -v apt &> /dev/null; then
            log_info "Updating package list..."
            apt update || {
                log_warning "Failed to update package list. Continuing anyway..."
            }

            log_info "Installing python3-venv..."
            apt install -y python3-venv || {
                log_error "Failed to install python3-venv. You may need to run this script as root or with sudo."
                exit 1
            }
            log_success "System dependencies installed"
        else
            log_warning "apt package manager not found. Assuming dependencies are available."
        fi
    else
        log_info "System dependencies already installed"
    fi
}

# Function to check if Python 3.8+ is available
check_python() {
    if command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
    elif command -v python &> /dev/null; then
        PYTHON_CMD="python"
    else
        log_error "Python is not installed or not in PATH"
        exit 1
    fi

    # Check Python version
    PYTHON_VERSION=$($PYTHON_CMD -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    REQUIRED_VERSION="3.8"

    if ! $PYTHON_CMD -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)" 2> /dev/null; then
        log_error "Python 3.8 or higher is required. Found: $PYTHON_VERSION"
        exit 1
    fi

    log_info "Using Python $PYTHON_VERSION: $($PYTHON_CMD --version)"
}

create_requirements() {
    log_info "Creating requirements.txt for CUDA 12.8 Isolation..."
    cat > "$REQUIREMENTS_FILE" << 'EOF'
--extra-index-url https://download.pytorch.org/whl/cu128
torch==2.9.0+cu128
torchvision==0.24.0+cu128
transformers>=4.44.0
accelerate>=0.33.0
Pillow>=10.0.0
numpy
safetensors
sentencepiece
protobuf
bitsandbytes>=0.43.0
EOF
    log_success "Created requirements.txt optimized for your Docker Host"
}

# Function to setup virtual environment
setup_venv() {
    log_info "Setting up virtual environment..."

    # Remove existing venv if force reinstall is requested
    if [[ "$FORCE_REINSTALL" == "true" && -d "$VENV_DIR" ]]; then
        log_warning "Removing existing virtual environment for fresh install..."
        rm -rf "$VENV_DIR"
    fi

    # Check if venv exists and is valid
    if [[ -d "$VENV_DIR" ]]; then
        if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
            log_warning "Virtual environment appears corrupted (missing activate script)"
            log_info "Removing corrupted virtual environment..."
            rm -rf "$VENV_DIR"
        else
            log_info "Virtual environment already exists and appears valid"
        fi
    fi

    # Create virtual environment if it doesn't exist or was removed
    if [[ ! -d "$VENV_DIR" ]]; then
        log_info "Creating virtual environment at $VENV_DIR"
        $PYTHON_CMD -m venv "$VENV_DIR"

        # Verify creation was successful
        if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
            log_error "Failed to create virtual environment properly"
            log_error "The activate script was not created"
            exit 1
        fi

        log_success "Virtual environment created successfully"
    fi

    # Activate virtual environment
    source "$VENV_DIR/bin/activate"

    # Upgrade pip
    log_info "Upgrading pip..."
    pip install --upgrade pip

    # Install or upgrade requirements
    if [[ "$FORCE_REINSTALL" == "true" ]]; then
        log_info "Force reinstalling requirements..."
        pip install --force-reinstall -r "$REQUIREMENTS_FILE"
    else
        log_info "Installing requirements..."
        pip install -r "$REQUIREMENTS_FILE"
    fi

    log_success "Virtual environment setup complete"
}

# Function to check if the Python script exists
check_python_script() {
    if [[ ! -f "$PYTHON_SCRIPT" ]]; then
        log_error "Python script not found: $PYTHON_SCRIPT"
        log_error "Please ensure joy_caption_batch.py is in the same directory as this script"
        exit 1
    fi
}

# Parse command line arguments
SETUP_ONLY=false
FORCE_REINSTALL=false
SCRIPT_ARGS=()
INPUT_DIR_SPECIFIED=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -h | --help)
            show_help
            exit 0
            ;;
        --setup-only)
            SETUP_ONLY=true
            shift
            ;;
        --force-reinstall)
            FORCE_REINSTALL=true
            shift
            ;;
        --output-dir | --prompt | --trigger-word | --timeout)
            # These are options that take values, add both the option and its value
            SCRIPT_ARGS+=("$1")
            shift
            if [[ $# -gt 0 ]]; then
                SCRIPT_ARGS+=("$1")
                shift
            fi
            ;;
        --no-skip-existing)
            # This is a flag option
            SCRIPT_ARGS+=("$1")
            shift
            ;;
        *)
            # Check if this looks like a directory path (first positional argument)
            if [[ ! "$INPUT_DIR_SPECIFIED" == "true" && ! "$1" =~ ^-- ]]; then
                INPUT_DIR_SPECIFIED=true
                SCRIPT_ARGS+=("$1")
            else
                SCRIPT_ARGS+=("$1")
            fi
            shift
            ;;
    esac
done

# Function to run the captioning script
run_captioning() {
    log_info "Activating virtual environment and running Joy Caption..."
    source "$VENV_DIR/bin/activate"

    # Use an array to handle arguments safely
    local cmd_args=("$PYTHON_SCRIPT" "$@")

    log_info "Running: python ${cmd_args[*]}"
    python "${cmd_args[@]}" # This is safer than eval

    log_success "Joy Caption processing completed"
}

# --- MODIFIED CUDA CHECK (Uses venv python) ---
check_cuda_compatibility() {
    log_info "Verifying CUDA health..."
    # Use the python inside the venv to ensure torch is present
    "$VENV_DIR/bin/python" << 'PYTHON_EOF'
import sys, torch
try:
    if not torch.cuda.is_available():
        print("ERROR: CUDA not found by Torch.")
        sys.exit(1)
    # Test a small tensor op to check driver/kernel match
    x = torch.randn(1, device='cuda') * 2
    print(f"CUDA Check Passed: Using {torch.cuda.get_device_name(0)}")
except Exception as e:
    print(f"CUDA KERNEL ERROR: {e}")
    sys.exit(1)
PYTHON_EOF
}

# Main execution
main() {
    log_info "Starting Joy Caption Batch Processing Setup"

    install_system_deps
    check_python
    create_requirements
    setup_venv # This installs Torch

    # NOW we check compatibility, after Torch is actually installed
    check_cuda_compatibility

    if [[ "$SETUP_ONLY" == "true" ]]; then
        log_success "Environment setup complete."
        exit 0
    fi

    check_python_script

    if [[ "$INPUT_DIR_SPECIFIED" == "false" ]]; then
        log_info "Using default: $DEFAULT_IMAGE_DIR"
        mkdir -p "$DEFAULT_IMAGE_DIR"
        # Prepend default dir to arguments
        SCRIPT_ARGS=("$DEFAULT_IMAGE_DIR" "${SCRIPT_ARGS[@]}")
    fi

    run_captioning "${SCRIPT_ARGS[@]}"

    # Check if VRAM is still leaked/held
    fuser -v /dev/nvidia* 2> /dev/null || echo "GPU VRAM is clear."
    log_success "All done! 🎉"
}

main "$@"
