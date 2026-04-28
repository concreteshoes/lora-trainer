# ====== Qwen 2512 Musubi Config File ======

# ---- [1] DATASET PATHS ----
# Path to your images and .txt caption files.
DATASET_DIR="$NETWORK_VOLUME/image_dataset_here"

# The name used for the saved .safetensors files.
OUTPUT_NAME="my_qwen_lora"

# Matches the output of your captioning script
CAPTION_EXT=".txt"

# ---- [2] TRAINING DYNAMICS ----
# BATCH_SIZE: Images processed per GPU step.
# Recommendation: 1 for 24GB VRAM, 2-4 for 80GB VRAM.
BATCH_SIZE=1

# NUM_REPEATS: How many times the model sees each image per epoch.
# Recommendation: 1-10 depending on dataset size. Use higher for small datasets (<20 images).
NUM_REPEATS=4

# GRAD_ACCUM_STEPS: Simulates a larger batch size without using more VRAM.
# Calculation: Total Batch = BATCH_SIZE * GRAD_ACCUM_STEPS.
# Recommendation: Set to 4 if BATCH_SIZE is 1 to stabilize gradients.
GRAD_ACCUM_STEPS=4

# ---- [3] HARDWARE & VRAM OPTIMIZATION ----
# FP8_BASE: Reduces DiT (transformer) weights to 8-bit.
# Recommendation: Set to 1 for GPUs with <40GB VRAM. Set to 0 for H100/A100 for max precision.
FP8_BASE=0

# FP8_SCALED: Advanced 8-bit scaling.
# Recommendation: Only set to 1 if FP8_BASE is 1.
FP8_SCALED=0

# RESOLUTION_LIST: Standard bucket size.
RESOLUTION_LIST="1024, 1024"

# ---- [4] LORA ARCHITECTURE ----
# LORA_RANK (Network Dim): The "capacity" of the LoRA.
# Recommendation: 16 is great for faces; 32 is better for complex outfits/styles.
# Note: Higher rank = higher VRAM usage and larger file size.
LORA_RANK=32

# --- LoRA Alpha (Scaling Factor) ---
# High Alpha (e.g., matching LORA_RANK) = Stronger effect, faster learning, higher risk of "crunchy" artifacts.
# Low Alpha (e.g., 50% of LORA_RANK) = Smoother gradients, better flexibility, more natural textures.
LORA_ALPHA=16

# ---- [5] SCHEDULE & OPTIMIZATION ----
# MAX_TRAIN_EPOCHS: Total training length.
# Recommendation: 10-20 for characters. Watch TensorBoard for over-fitting (burn).
MAX_TRAIN_EPOCHS=16

# SAVE_EVERY: Frequency of checkpoint saves.
SAVE_EVERY_N_EPOCHS=1

# LEARNING_RATE: The "speed" of learning.
LEARNING_RATE=5e-5

# ---- OPTIMIZER CONFIGURATION ----
# Choices: "adamw" "adamw8bit", "adafactor", "prodigyopt.Prodigy"
OPTIMIZER_TYPE="adamw8bit"

# Choices: "cosine", "constant"
LR_SCHEDULER="cosine"

# Choices: "shift", "sigmoid", "qwen_shift"
TIMESTEP_SAMPLING="shift"

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
        "clip_threshold=1.0"
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
NETWORK_DROPOUT=0

# Massive boost to training speed if set to 0, make sure you have enough VRAM, minimum 48GB with batch_size 1
GRADIENT_CHECKPOINTING=1

# Attention - "flash", "sdpa"
ATTN="flash"

# Set to 1 for 24GB/48GB cards to prevent OOM. Set to 0 for 80GB+ cards for max speed. (Recommended to use with Flash Attention)
SPLIT_ATTN=1

# NUM_CPU_THREADS_PER_PROCESS: Controls the CPU threads used by the main training process.
NUM_CPU_THREADS_PER_PROCESS=1

# MAX_DATA_LOADER_N_WORKERS: Number of subprocesses dedicated to loading and augmenting images.
MAX_DATA_LOADER_N_WORKERS=2

# Shift 2.5–3.2 for improved detail and photorealism
DISCRETE_FLOW_SHIFT=2.5

# Set to True to prevent upscaling of small images, ensuring the model learns from real pixels rather than blurred artifacts
BUCKET_NO_UPSCALE=true

KEEP_DATASET=0
SKIP_CACHE=0
