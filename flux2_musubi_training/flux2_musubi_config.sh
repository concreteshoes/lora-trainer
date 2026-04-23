# ====== Flux.2-Klein-9B Musubi Config File ======

# ---- [1] DATASET PATHS ----
# Path to your images and .txt caption files.
DATASET_DIR="$NETWORK_VOLUME/image_dataset_here"

# Prefix for your saved .safetensors.
OUTPUT_NAME="my_flux2_lora"

# Matches your Gemini/JoyCaption output.
CAPTION_EXT=".txt"

# ---- [2] TRAINING DYNAMICS ----
# BATCH_SIZE: 1 is the standard for 9B models on almost all hardware.
# Recommendation: Only increase to 2+ if using an H100 (80GB) AND FP8_BASE is enabled.
BATCH_SIZE=1

# NUM_REPEATS: Flux requires more "reps" to learn identity than smaller models.
# Recommendation: 10-15 repeats for a small high-quality dataset (15-30 images).
NUM_REPEATS=10

# GRAD_ACCUM_STEPS: Increases effective batch size without OOM.
# Recommendation: 4 is highly recommended for Flux to smooth out the training curve.
GRAD_ACCUM_STEPS=4

# MAX_TRAIN_EPOCHS:
# Recommendation: 12-16 is usually enough given the high NUM_REPEATS.
MAX_TRAIN_EPOCHS=16

# SAVE_EVERY_N_EPOCHS: Flux checkpoints are large (~300MB+); save every 2 to save disk space.
SAVE_EVERY_N_EPOCHS=2

# ---- [3] HARDWARE & VRAM OPTIMIZATION ----
# FP8_BASE & FP8_SCALED: Runs the DiT (Transformer) in 8-bit.
# Recommendation: Set BOTH to 1 for 24GB cards (3090/4090).
# Set to 0 for H100 to maintain the highest BF16 quality for skin/textures.
FP8_BASE=0
FP8_SCALED=0

# RESOLUTION_LIST:
# Recommendation: Flux.2 is optimized for 1024x1024.
# Lowering this to 512x512 is NOT recommended for Flux as it destroys the "9B" detail advantage.
RESOLUTION_LIST="1024, 1024"

# ---- [4] LORA ARCHITECTURE ----
# NETWORK_DIM (Rank):
# Recommendation: 32 is the "sweet spot" for Flux.
# Go to 64 if training a very complex art style or hyper-detailed clothing.
LORA_RANK=32

# NETWORK_ALPHA:
# Recommendation: Always set to 1/2 of DIM (e.g., 16) to prevent weight explosion/frying.
LORA_ALPHA=16

# ---- [5] LEARNING RATE & OPTIMIZER ----
LEARNING_RATE=1.0

# ---- OPTIMIZER CONFIGURATION ----
# Choices: "adamw" "adamw8bit", "adafactor", "prodigyopt.Prodigy"
OPTIMIZER_TYPE="prodigyopt.Prodigy"
# Choices: "cosine", "constant"
LR_SCHEDULER="cosine"
# Choices: "flux2_shift", "sigmoid"
TIMESTEP_SAMPLING="flux2_shift"

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

# NUM_CPU_THREADS_PER_PROCESS: Controls the CPU threads used by the main training process.
NUM_CPU_THREADS_PER_PROCESS=1

# MAX_DATA_LOADER_N_WORKERS: Number of subprocesses dedicated to loading and augmenting images.
MAX_DATA_LOADER_N_WORKERS=2

# Shift of 2.0–2.8 for detail without artifacts
DISCRETE_FLOW_SHIFT=2.0

# Set to True to prevent upscaling of small images, ensuring the model learns from real pixels rather than blurred artifacts
BUCKET_NO_UPSCALE=true

KEEP_DATASET=0
SKIP_CACHE=0
