# Comprehensive LoRA Trainer using Diffusion Pipe & Musubi Tuner w/ Flash & Sage Attn for CUDA 12.8
## Quick Start Guide

This is a thorough LoRA trainer template featuring Flux, SDXL, Wan, Qwen and Z-Image models using both a diffusion-pipe wrapper and Musubi training scripts.

The diffusion-pipe `interactive_start_training.sh` will allow you to train Flux1-dev, Wan 2.1, SDXL, Qwen Image, Qwen 2512 and Z-Image Turbo models.

The provided Musubi scripts will allow you to train with Qwen Edit-2511, Z-Image Base & ostris' De-Turbo, Wan 2.2 and FLUX.2 [klein] 9B. You will find instructions in
their respective folders. It is recommended to choose the Musubi script for Z-Image Turbo as it provides a more granular approach to training.

Use JoyCaption for auto-captioning of images and the Gemini script for videos (paid tier required!). 

You can resume training from the last checkpoint for either trainer pipeline. Use TensorBoard for graph eval and if you are training with Musubi you have
the ability to evaluate your LoRAs by running visual inference for specified checkpoints.

Exclusive to the Musubi scripts, you can apply Post-Hoc EMA merge for a range of trained steps to get the 'perfect' LoRA model by injecting a beta value.

This template has provisions for deployment to ephemeral and persistent storage environments. An OpenSSH server is included for secure transfer of data.
The image comes with installed `rclone` for transfers to and from Google Drive. Aside from the obvious benefit, this approach is recommended if you are deploying
on a community cloud service like vast.ai to an instance that throttles ssh transfers. Check the configuration script in the root directory on how to set it up.


⚠️ Hardware Requirements

Take notice of the size of the models you want to train with and allocate your storage capacity accordingly.

GPU: NVIDIA Ampere architecture or newer is required (RTX 30-series, 40-series, A100, H100, etc.).

Precision: This template uses bf16. Older GPUs (Turing/20-series and below) do not support native bf16 and will fail or perform poorly.

VRAM: 24GB minimum (RTX 3090/4090) for 9B models; 40GB+ recommended for Wan 2.2 (A14B).

##### Note: If you run into bugs, report them to me on discord: bytesizelife

---

### Environmental Variables

| Variable | Description |
|---|---|
| `HUGGING_FACE_TOKEN=""` | Hugging Face API key (required for Flux models) |
| `GEMINI_API_KEY=""` | Gemini API key (required for video processing) |
| `SSH_PUBLIC_KEY=""` | Add your public key if you want SSH transfers |



### Ports

- `8888` - Jupyter
- `22`   - SSH
- `6006` - TensorBoard


### Accessing the Instance

```bash 
If you are using custom SSH key location you might want to create a config file in
~/.ssh/config for Linux or $HOME\.ssh\config for Windows.
```
Linux:
```bash
Host *
    IdentityFile PATH/.ssh/id_ed25519
    IdentitiesOnly yes
```
Windows:
```bash 
Host *
    IdentityFile PATH\.ssh\id_ed25519
    IdentitiesOnly yes
```

You can transfer files using `rsync` and connect via SSH:

### Example: sync local dataset to remote
```bash
rsync -avP -e "ssh -p <SSH_PORT>" /path/to/local/dataset/ hostname@<SERVER_IP>:/path/to/remote/dataset/
```
#### SSH with port forwarding for JupyterLab:
```bash
ssh -p <SSH_PORT> hostname@<SERVER_IP> -L 8888:localhost:8888
```
Then open your browser to:
http://localhost:8888/lab

#### SSH with port forwarding for TensorBoard:
```bash
ssh -p <SSH_PORT> hostname@<SERVER_IP> -L 6006:localhost:6006
```
Then open your browser to:
http://localhost:6006

---

## Getting started with the diffusion-pipe training wrapper

### Step 1: Initialize Terminal Environment (if on Runpod)

1. Click the **Terminal** button to open a command prompt
2. Type: `bash`
3. Press **ENTER** to enter bash shell

### Step 2: Start Interactive Training Process

Type the following command and press **ENTER**:

```bash
bash interactive_start_training.sh
```

### Step 3: Follow the Interactive Setup

The script will guide you through configuration options:

**Model Selection:**
- Flux models
- SDXL models
- Wan models
- Qwen Image model
- Z-Image Turbo model

**API key configuration:**
- `HUGGING_FACE_TOKEN=""` — Hugging Face API key (required for Flux models)
- `GEMINI_API_KEY=""` — Gemini API key (required for video processing)

**Dataset Processing:**
- Image captioning options
- Video captioning options
- Training configuration review

### Step 4: Wait for Completion

The automated process will:
- Download required models
- Generate captions for your dataset
- Start the training process
- Save results when complete


## For Musubi-tuner training consult the readme files in the <model>_musubi_training folders

---

## Results Location

Your trained LoRA files will be saved in: `output_folder/` and `output_folder_musubi/`

---

## Preparation Checklist

- [ ] Training data placed in correct folder:
  - Images: `image_dataset_here/` folder
  - Videos: `video_dataset_here/` folder
- [ ] API keys ready:
  - Hugging Face API key (for Flux models)
  - Gemini API key (for video processing)

---

**Happy training!**
