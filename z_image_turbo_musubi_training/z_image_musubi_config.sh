# ====== Z-Image Turbo Musubi Config File ======

# ---- [1] DATASET PATHS ----
DATASET_DIR="$NETWORK_VOLUME/image_dataset_here"
OUTPUT_NAME="my_zimage_lora"
CAPTION_EXT=".txt"

# ---- [2] TRAINING DYNAMICS ----
# BATCH_SIZE: 1 is standard for 24GB/48GB. H100 can push 4-8.
BATCH_SIZE=1

# NUM_REPEATS: Controls how often each image is seen per epoch.
# Z-Image trains efficiently; use low repeats to avoid overfitting.
# Recommended:
#   3–5 → ideal for ~50–100 images (balanced learning)
#   5–7 → for high-variation datasets
#   >8  → risk of overfitting (identity/clothing lock)
NUM_REPEATS=4

# GRAD_ACCUM_STEPS: Effectively multiplies batch size without VRAM cost.
# Recommendation: Set to 4 if BATCH_SIZE is 1.
GRAD_ACCUM_STEPS=4

# RESOLUTION_LIST: Native target for Z-Image.
RESOLUTION_LIST="1024, 1024"

# ---- [3] HARDWARE & VRAM OPTIMIZATION ----
# TE_CACHE_BATCH_SIZE: Batch size for pre-caching the Qwen LLM.
# Recommendation: 8 for 24GB cards, 32+ for H100.
TE_CACHE_BATCH_SIZE=32

# FP8_BASE & FP8_SCALED: Runs the DiT in 8-bit.
# Recommendation: Set to 1 for 24GB cards. Keep at 0 for H100/A100.
FP8_BASE=0
FP8_SCALED=0

# ---- [4] LORA ARCHITECTURE ----
# NETWORK_DIM / ALPHA: Z-Image typically handles higher ranks well.
LORA_RANK=32
LORA_ALPHA=16

# ---- [5] SCHEDULE & OPTIMIZER ----
MAX_TRAIN_EPOCHS=12
SAVE_EVERY_N_EPOCHS=1
LEARNING_RATE=1.0

# ---- OPTIMIZER CONFIGURATION ----
# Choices: "adamw" "adamw8bit", "adafactor", "prodigyopt.Prodigy"
OPTIMIZER_TYPE="adamw"

# Base arguments that work everywhere
OPTIMIZER_ARGS=(
    "weight_decay=0.01"
)

# Arguments used by adamw and adamw8bit
if [ "$OPTIMIZER_TYPE" == "adamw" ] || [ "$OPTIMIZER_TYPE" == "adamw8bit" ]; then
    OPTIMIZER_ARGS+=(
        "eps=1e-8"
    )
fi

# Arguments used by Adafactor
if [ "$OPTIMIZER_TYPE" == "adafactor" ]; then
    OPTIMIZER_ARGS+=(
        "scale_parameter=False"
        "relative_step=False"
        "warmup_init=False"
    )
fi

# Arguments used by Prodigy
if [ "$OPTIMIZER_TYPE" == "prodigyopt.Prodigy" ]; then
    OPTIMIZER_ARGS+=(
        "decouple=True"
        "d_coef=0.8"
        "use_bias_correction=True"
        "safeguard_warmup=True"
        "betas=0.9,0.99"
    )
fi

# ---- [6] ADVANCED ----
# Enables Post-Hoc EMA for merging of snapshots after training, useful for achieving 'perfect' LoRAs, adds to storage req.
USE_EMA=0

# Reduces overfitting and correlation locking, improving generalization and composability of the LoRA (0 - 0.09)
NETWORK_DROPOUT=0.01

# Massive boost to training speed if set to 0, make sure you have enough VRAM, minimum 48GB with batch_size 1
GRADIENT_CHECKPOINTING=1

# Set to 1 for 24GB/48GB cards to prevent OOM. Set to 0 for 80GB+ cards for max speed.
SPLIT_ATTN=0

# NUM_CPU_THREADS_PER_PROCESS: Controls the CPU threads used by the main training process.
NUM_CPU_THREADS_PER_PROCESS=1

# MAX_DATA_LOADER_N_WORKERS: Number of subprocesses dedicated to loading and augmenting images.
MAX_DATA_LOADER_N_WORKERS=2

# Shift (2.0–3.1) is mainly useful during inference for extra detail.
DISCRETE_FLOW_SHIFT=2.5

# Set to True to prevent upscaling of small images, ensuring the model learns from real pixels rather than blurred artifacts
BUCKET_NO_UPSCALE=true

KEEP_DATASET=0
SKIP_CACHE=0
