# ====== Wan 2.2 Config File ======

# ---- [1] DATASET PATHS ----
# Toggle these depending on your current training phase.
DATASET_DIR="$NETWORK_VOLUME/image_dataset_here"
# DATASET_DIR="$NETWORK_VOLUME/video_dataset_here"

# Set type: "image" (for character likeness) or "video" (for motion/I2V).
DATASET_TYPE="image"

# Matches your Gemini captioning output.
CAPTION_EXT=".txt"

# ---- [2] LORA ARCHITECTURE & OUTPUT ----
# Capacity (The "Resolution" of the LoRA)
LORA_RANK=64
# Strength (The "Volume" of the LoRA)
LORA_ALPHA=32

# TITLE_HIGH: Focuses on composition/coarse features (Time > 70%).
# TITLE_LOW: Focuses on fine details/texture (Time < 30%).
TITLE_HIGH="wan2.2_lora_high"
TITLE_LOW="wan2.2_lora_low"

# ---- [3] TRAINING DYNAMICS ----
# BATCH_SIZE: 1 is safest for Wan 2.2 (1.3B/14B) on most hardware.
BATCH_SIZE=1

# Increases effective batch size.
# Recommendation: Set to 2-4 to stabilize video training.
GRAD_ACCUM_STEPS=2

# NUM_REPEATS: Standard for image-based Likeness.
NUM_REPEATS=1

# RESOLUTION_LIST: Wan 2.2 is flexible, but 1024x1024 (Image) or 480p/720p (Video) is standard.
RESOLUTION_LIST="1024, 1024"

# ---- [4] VIDEO SPECIFIC OPTIONS ----
# TARGET_FRAMES: Number of frames per sample. [1] for images, [81] for 5s at 16fps.
TARGET_FRAMES="1, 57, 117"

# FRAME_EXTRACTION: Where to start the clip.
# Recommendation: "head" is best for I2V (Image-to-Video) training.
FRAME_EXTRACTION="head"

# ---- [5] HARDWARE & VRAM ----
# FP8_BASE: Crucial for Wan 2.2 (14B model) on 24GB/40GB cards.
# Recommendation: Set to 1 for consumer cards. 0 for H100.
FP8_BASE=0

# FP8_T5: Runs the T5 Text Encoder in 8-bit.
# Recommendation: Always 1 unless you have 80GB VRAM to spare.
FP8_T5=1

# ---- [6] SCHEDULE & OPTIMIZER ----
MAX_TRAIN_EPOCHS=100
SAVE_EVERY_N_EPOCHS=20
LEARNING_RATE=1e-4

# SEED Selection: Pick one if running a single GPU.
SEED_HIGH=41
SEED_LOW=42

# ---- OPTIMIZER CONFIGURATION ----
# Choices: "adamw" "adamw8bit", "adafactor", "prodigyopt.Prodigy"
OPTIMIZER_TYPE="adamw"
# Choices: "cosine", "constant"
LR_SCHEDULER="cosine"
# Choices: "shift", "sigmoid"
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

# Shift of 2.5–3.5 for sharper, more detailed outputs
DISCRETE_FLOW_SHIFT=3.0

# Set to True to prevent upscaling of small images, ensuring the model learns from real pixels rather than blurred artifacts
BUCKET_NO_UPSCALE=true

KEEP_DATASET=0
SKIP_CACHE=0
