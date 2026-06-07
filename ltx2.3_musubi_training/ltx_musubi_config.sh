#!/usr/bin/env bash

# ====== LTX 2.3 Config File ======

# ---- [1] LTX VERSION & SYSTEM MODALITY ----
# Choices: "2.0", "2.3" (Locks dual embedding setups)
LTX_VERSION="2.3"
# Choices: "video" (T2V/I2V), "av" (Joint Audio-Video), "audio"
LTX_MODE="video"

# ---- [2] DATASET PATHS & SPECIFICATIONS ----
DATASET_DIR="$NETWORK_VOLUME/video_dataset_here"
#DATASET_DIR="$NETWORK_VOLUME/image_dataset_here"
CAPTION_EXT=".txt"

# ---- [3] LORA ARCHITECTURE & CAPACITY ----
# Structural width of adjustment layer
LORA_RANK=32
# Mathematical layer scaling strength
LORA_ALPHA=32
# Choices: "t2v" (Text-to-Video), "v2v" (For I2V / IC-LoRA)
LORA_TARGET_PRESET="t2v"
# Set to 1 to train text-connector embeddings (+3.8GB VRAM)
TRAIN_CONNECTORS=0
# The base label applied to final exported checkpoints
OUTPUT_NAME="my_ltx23_lora"

# ---- [4] TRAINING DYNAMICS & BATCHING ----
BATCH_SIZE=1
# Multiplies batch size to stabilize video gradients
GRAD_ACCUM_STEPS=2
MAX_TRAIN_EPOCHS=13
SAVE_EVERY_N_EPOCHS=1
# Number of times to repeat the dataset per epoch
NUM_REPEATS=5

# Handles bucket counts that do not divide evenly into the accumulation window
# Choices: "drop" (skips remainders), "pad" (repeats samples), "allow_mixed"
ACCUMULATION_GROUP_REMAINDER="pad"

# Native Aspect Ratio Bucketing resolutions
RESOLUTION_LIST="720, 1280"
# Prevents blurring small source files
BUCKET_NO_UPSCALE=true

# ---- [5] VIDEO & FRAME EXTRACTION OPTIONS ----
# LTX-2 uses an 8-frame temporal VAE step. Clean blocks follow (8N + 1).
# "1, 41, 81" is a perfect mathematical bridge that works cleanly for both Wan and LTX-2.
TARGET_FRAMES="1, 41, 81"
# Options: "head", "chunk", "full", "slide", "uniform"
FRAME_EXTRACTION="full"
# Hard ceiling to truncate ultra-long clips and prevent VAE Out-of-Memory errors
MAX_FRAMES=81

# Active only if FRAME_EXTRACTION is set to "slide"
#FRAME_STRIDE=1
# Active only if FRAME_EXTRACTION is set to "uniform"
#FRAME_SAMPLE=4

# ---- [6] REGULARIZATION & GUIDANCE PREP ----
# Probability of dropping text captions per step to teach the model unconditional generation.
# Enables clean execution of Classifier-Free Guidance (CFG) during inference.
CAPTION_DROPOUT_RATE=0.1
# Controls target layer drop-out rates (0 - 0.09)
NETWORK_DROPOUT=0

# ---- [7] ADVANCED / MULTI-MODAL OPTIONS (DISABLED) ----
# REFERENCE_FRAMES=1
# REFERENCE_DOWNSCALE=1
# FIRST_FRAME_COND_P=0.1
# VIDEO_CAPTION_DROPOUT_RATE=0.0
# AUDIO_CAPTION_DROPOUT_RATE=0.0

# ---- [8] SCHEDULE & OPTIMIZER CONFIG ----
LEARNING_RATE=1e-4

# Choices: "cosine", "constant", "constant_with_warmup"
LR_SCHEDULER="cosine"

# Choices: "adamw", "adamw8bit", "adafactor", "prodigyopt.Prodigy"
OPTIMIZER_TYPE="adamw"

# Shifted Logit Normal is the optimized default sampling setup for LTX-2 architectures.
# Choices: "shift", "sigmoid" "shifted_logit_normal"
TIMESTEP_SAMPLING="shifted_logit_normal"

# Shift of 2.5–3.5 for sharper, more detailed outputs.
# If timestep is 'shifted_logit_normal' this value is not being used.
DISCRETE_FLOW_SHIFT=3.0

# Standard optimization rules applied across matrices
OPTIMIZER_ARGS=(
    "weight_decay=0.01"
)

# Arguments used by adamw and adamw8bit
if [ "$OPTIMIZER_TYPE" == "adamw" ] || [ "$OPTIMIZER_TYPE" == "adamw8bit" ]; then
    OPTIMIZER_ARGS+=("eps=1e-8")
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

# Some blocks can be offloaded to CPU for memory savings
#BLOCKS_TO_SWAP=

# Reduces overfitting and correlation locking, improving generalization and composability of the LoRA (0 - 0.09)
NETWORK_DROPOUT=0

# Massive boost to training speed if set to 0, make sure you have enough VRAM, minimum 48GB with batch_size 1
GRADIENT_CHECKPOINTING=1

# Attention - "flash", "sdpa"
ATTN="flash"

# NUM_CPU_THREADS_PER_PROCESS: Controls the CPU threads used by the main training process.
NUM_CPU_THREADS_PER_PROCESS=1

# MAX_DATA_LOADER_N_WORKERS: Number of subprocesses dedicated to loading and augmenting images.
MAX_DATA_LOADER_N_WORKERS=2

# Set to True to prevent upscaling of small images, ensuring the model learns from real pixels rather than blurred artifacts
BUCKET_NO_UPSCALE=true

KEEP_DATASET=0
SKIP_CACHE=0
