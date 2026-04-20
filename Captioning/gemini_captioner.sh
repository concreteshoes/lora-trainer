#!/bin/bash
set -e

# --- ARGUMENT PARSING ---
MODE=""
USER_TRIGGER=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)
            MODE="$2"
            shift 2
            ;;
        --trigger-word)
            USER_TRIGGER="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# --- 1. API KEY HANDLING (PERSISTENT) ---

ENV_FILE="/etc/environment"

# 1. If already set (wrapper or environment), use it
if [ -n "$GEMINI_API_KEY" ] && [ "$GEMINI_API_KEY" != "token_here" ]; then
    echo "Using GEMINI_API_KEY from environment"
else
    # 2. Try loading from /etc/environment
    if [ -f "$ENV_FILE" ]; then
        export $(grep '^GEMINI_API_KEY=' "$ENV_FILE" | xargs 2> /dev/null || true)
    fi

    # 3. If still missing → ONLY prompt if interactive
    if [ -z "$GEMINI_API_KEY" ] || [ "$GEMINI_API_KEY" == "token_here" ]; then

        # Detect if script is running interactively
        if [ -t 0 ]; then
            echo "---"
            echo "GEMINI_API_KEY not detected."

            read -p "Please paste your Gemini API Key: " USER_KEY

            if [ -z "$USER_KEY" ]; then
                echo "Error: No key provided."
                exit 1
            fi

            export GEMINI_API_KEY="$USER_KEY"

            echo "[OK] Saving GEMINI_API_KEY for future runs..."

            sed -i '/^GEMINI_API_KEY=/d' "$ENV_FILE" 2> /dev/null || true
            echo "GEMINI_API_KEY=$USER_KEY" | tee -a "$ENV_FILE" > /dev/null

        else
            echo "Error: GEMINI_API_KEY not set and no interactive input available."
            exit 1
        fi
    fi
fi

echo "Using GEMINI_API_KEY: ${GEMINI_API_KEY:0:4}****${GEMINI_API_KEY: -4}"

# --- 2. MEDIA TYPE & PATH SELECTION ---
if [ -n "$MODE" ]; then
    if [ "$MODE" = "images" ]; then
        MEDIA_CHOICE=1
    elif [ "$MODE" = "videos" ]; then
        MEDIA_CHOICE=2
    else
        echo "Error: --mode must be 'images' or 'videos'"
        exit 1
    fi
    echo "Mode set via CLI: $MODE"
else
    echo "------------------------------------------------"
    echo " Select Media Type to Caption:"
    echo " 1) Images"
    echo " 2) Videos"
    echo "------------------------------------------------"
    read -p "Selection [1 or 2]: " MEDIA_CHOICE
fi

if [ "$MEDIA_CHOICE" == "1" ]; then
    DATASET_FOLDER="image_dataset_here"
    MODE="Images"
else
    DATASET_FOLDER="video_dataset_here"
    MODE="Videos"
fi

# Set the final working directory
WORKING_DIR="${NETWORK_VOLUME:-/workspace}/$DATASET_FOLDER"
echo "Mode set to: $MODE"
echo "Target Directory: $WORKING_DIR"

# --- 3. TRIGGER WORD HANDLING ---
echo "------------------------------------------------"

if [ -z "$USER_TRIGGER" ]; then
    if [ -t 0 ]; then
        read -p "Enter Trigger Word (Leave blank for none): " USER_TRIGGER
    else
        USER_TRIGGER=""
    fi
else
    echo "Using trigger word from CLI: $USER_TRIGGER"
fi

echo "------------------------------------------------"

# Define variables for Conda/Repo
REPO_DIR="/TripleX"
REPO_URL="https://github.com/Hearmeman24/TripleX.git"
CONDA_DIR="/tmp/TripleX_miniconda"
CONDA_ENV_NAME="TripleX"
CONDA_ENV_PATH="$CONDA_DIR/envs/$CONDA_ENV_NAME"
SCRIPT_PATH="/TripleX/captioners/gemini.py"

# --- 4. REPO & CONDA SETUP (Fast Check) ---
if [ ! -d "$REPO_DIR" ]; then
    git clone "$REPO_URL" "$REPO_DIR"
fi

if [ ! -d "$CONDA_DIR" ]; then
    mkdir -p "/tmp/triplex"
    curl -sSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/triplex/miniconda.sh
    bash /tmp/triplex/miniconda.sh -b -p $CONDA_DIR
    rm /tmp/triplex/miniconda.sh
fi

export PATH="$CONDA_DIR/bin:$PATH"
eval "$($CONDA_DIR/bin/conda shell.bash hook)"

if [ ! -d "$CONDA_ENV_PATH" ]; then
    conda create -y -n $CONDA_ENV_NAME python=3.12
    conda activate $CONDA_ENV_NAME
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
    pip install google-generativeai opencv-python-headless python-dotenv
else
    conda activate $CONDA_ENV_NAME
fi

# --- MODEL PATCH ---
if ! grep -q 'gemini-2.5-flash' "$SCRIPT_PATH"; then
    echo "[PATCH] Updating model lists → gemini-2.5-flash"

    sed -i '/INDIVIDUAL_FALLBACK_MODELS = \[/,/]/c\
INDIVIDUAL_FALLBACK_MODELS = [\
    "gemini-2.5-flash"\
]' "$SCRIPT_PATH"

    sed -i '/COMPOSITE_FALLBACK_MODELS = \[/,/]/c\
COMPOSITE_FALLBACK_MODELS = [\
    "gemini-2.5-flash"\
]' "$SCRIPT_PATH"
fi

# --- CONCURRENCY PATCH (safe replace) ---
if ! grep -q 'max_workers=1' "$SCRIPT_PATH"; then
    echo "[PATCH] Limiting concurrency"

    sed -i 's/ThreadPoolExecutor(max_workers=[0-9]\+)/ThreadPoolExecutor(max_workers=1)/g' "$SCRIPT_PATH"
    sed -i 's/ThreadPoolExecutor()/ThreadPoolExecutor(max_workers=1)/g' "$SCRIPT_PATH"
fi

# --- RATE LIMIT DELAY ---
if ! grep -q 'time.sleep(1.5)' "$SCRIPT_PATH"; then
    echo "[PATCH] Adding rate limit delay"

    sed -i '/for model_name in model_list:/a\ \ \ \ \ \ \ \ time.sleep(1.5)' "$SCRIPT_PATH"
fi

# --- DISABLE REWRITE (ALL OCCURRENCES) ---
if ! grep -q 'SKIP_REWRITE' "$SCRIPT_PATH"; then
    echo "[PATCH] Disabling rewrite step"

    sed -i 's/final_caption = rewrite_composite_caption(composite)/# SKIP_REWRITE\n# &/g' "$SCRIPT_PATH"
fi

# --- VERIFY ---
echo "----- VERIFY MODELS -----"
grep -A5 "FALLBACK_MODELS" "$SCRIPT_PATH"

echo "----- VERIFY THREADS -----"
grep -n "ThreadPoolExecutor" "$SCRIPT_PATH"

echo "----- VERIFY DELAY -----"
grep -n "sleep" "$SCRIPT_PATH"

# --- 5. EXECUTION LOGIC ---
# We structure the prompt to force the trigger word as the FIRST token.
if [ -n "$USER_TRIGGER" ]; then
    # The 'STRICT' prompt ensures no introductory filler words.
    EXTRA_PROMPT="CRITICAL: Your output MUST begin with exactly '$USER_TRIGGER, ' as the very first token. Do not use phrases like 'In this image' or 'This video shows'. Start directly with '$USER_TRIGGER, ' followed by the visual description."
else
    EXTRA_PROMPT=""
fi

# Create directory if it doesn't exist to avoid python crashes
mkdir -p "$WORKING_DIR"

echo "Launching Gemini Captioning for $MODE..."

# Run the python script
if [ "$MEDIA_CHOICE" == "2" ]; then
    # Video mode: sample frames
    python "$SCRIPT_PATH" --dir "$WORKING_DIR" --custom_prompt "$EXTRA_PROMPT" --max_frames 10 --fps 1.0
else
    # Image mode: treat each file as a single-frame process
    python "$SCRIPT_PATH" --dir "$WORKING_DIR" --custom_prompt "$EXTRA_PROMPT"
fi

echo "[GEMINI_DONE]"
