#!/bin/bash

# --- CONFIGURATION ---
# Adjust this path to where train.py actually lives
REPO_DIR="$NETWORK_VOLUME/diffusion-pipe"
# ---------------------

print_info() { echo -e "\033[1;34m[INFO]\033[0m $*"; }
print_success() { echo -e "\033[1;32m[SUCCESS]\033[0m $*"; }
print_warning() { echo -e "\033[1;33m[WARNING]\033[0m $*"; }
print_error() { echo -e "\033[1;31m[ERROR]\033[0m $*"; }

if [ -z "$1" ]; then
    print_error "Usage: $0 <model_toml_file>"
    exit 1
fi

# Get the absolute path of the TOML before we change directories
MODEL_TOML=$(realpath "$1")

if [ ! -f "$MODEL_TOML" ]; then
    print_error "Config file not found: $MODEL_TOML"
    exit 1
fi

# 1. Parse Config (doing this before cd in case paths are relative to script)
OUTPUT_DIR_RAW=$(grep -E "^output_dir[[:space:]]*=" "$MODEL_TOML" | sed -E "s/.*=[[:space:]]*//;s/['\"]//g")

# If output_dir is relative in the TOML, we need to be careful.
# Usually, it's relative to the repo root during training.
print_info "Searching for past runs in: $OUTPUT_DIR_RAW"

# 2. Enter the Repo
if [ -d "$REPO_DIR" ]; then
    cd "$REPO_DIR"
    print_info "Moved to directory: $(pwd)"
else
    print_error "Repo directory not found at $REPO_DIR"
    exit 1
fi

# 3. Identify Past Runs
# Check if OUTPUT_DIR is absolute; if not, assume it's relative to REPO_DIR
if [[ "$OUTPUT_DIR_RAW" = /* ]]; then
    SEARCH_DIR="$OUTPUT_DIR_RAW"
else
    SEARCH_DIR="$(pwd)/$OUTPUT_DIR_RAW"
fi

runs=()
if [ -d "$SEARCH_DIR" ]; then
    for d in "$SEARCH_DIR"/*/; do
        dirname=$(basename "$d")
        if [[ "$dirname" =~ ^[0-9]{8}_[0-9]{2}-[0-9]{2}-[0-9]{2}$ ]]; then
            runs+=("$dirname")
        fi
    done
fi

# 4. Selection Logic
RESUME_ARG=""
if [ ${#runs[@]} -eq 0 ]; then
    print_warning "No existing training runs found in $SEARCH_DIR. Starting fresh."
else
    print_info "Found ${#runs[@]} previous run(s)."
    select choice in "${runs[@]}" "Start Fresh (New Run)"; do
        if [[ -n "$choice" && "$choice" == "Start Fresh (New Run)" ]]; then
            break
        elif [[ -n "$choice" ]]; then
            RESUME_ARG="$choice"
            print_success "Selected run: $RESUME_ARG"
            break
        else
            print_warning "Invalid selection."
        fi
    done
fi

# 5. Epoch Check
NUM_EPOCHS=$(grep -E "^epochs[[:space:]]*=" "$MODEL_TOML" | grep -o "[0-9]\+")
if [ -n "$RESUME_ARG" ]; then
    print_info "Current epochs in config: $NUM_EPOCHS"
    read -p "Increase total epochs? [y/N]: " EXTEND
    if [[ "$EXTEND" =~ ^[Yy]$ ]]; then
        read -p "Enter new total: " NEW_EPOCHS
        if [[ "$NEW_EPOCHS" =~ ^[0-9]+$ ]] && [ "$NEW_EPOCHS" -gt "$NUM_EPOCHS" ]; then
            sed -i -E "s/^epochs[[:space:]]*=.*/epochs = $NEW_EPOCHS/" "$MODEL_TOML"
            print_success "Updated TOML to $NEW_EPOCHS epochs."
        fi
    fi
fi

# 5b. Learning Rate Check for Resume
if [ -n "$RESUME_ARG" ]; then
    # Check if a linear scheduler is active
    if grep -qE "^lr_scheduler[[:space:]]*=[[:space:]]*['\"]linear['\"]" "$MODEL_TOML"; then
        print_warning "Linear scheduler detected. Because epochs are extending, the original LR has decayed to 0."
        read -p "Do you want to set a forced constant learning rate for the resumed epochs? [y/N]: " FORCE_LR
        if [[ "$FORCE_LR" =~ ^[Yy]$ ]]; then
            read -p "Enter forced LR (e.g., 1e-5): " NEW_LR
            if [[ -n "$NEW_LR" ]]; then
                # This sed command finds force_constant_lr (even if commented out with a #) and activates it
                sed -i -E "s/^[#[:space:]]*force_constant_lr[[:space:]]*=.*/force_constant_lr = $NEW_LR/" "$MODEL_TOML"
                print_success "Updated TOML: forced learning rate set to $NEW_LR"
            fi
        fi
    fi
fi

# 6. Execution
CMD=(deepspeed --num_gpus=1 train.py --deepspeed --config "$MODEL_TOML")

if [ -n "$RESUME_ARG" ]; then
    # Combine the base search directory with the selected timestamp folder
    FULL_RESUME_PATH="$SEARCH_DIR/$RESUME_ARG"
    CMD+=(--resume_from_checkpoint "$FULL_RESUME_PATH")
fi

export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"

print_info "Launching Training..."
"${CMD[@]}"
