# LoRA Trainer using Diffusion-pipe, Musubi-Tuner & OneTrainer w/ Flash & Sage Attn for CUDA 12.8
## Quick Start Guide

This is a thorough LoRA trainer template featuring Flux, SDXL, Wan, Qwen and Z-Image models using diffusion-pipe, Musubi training scripts and OneTrainer.

The diffusion-pipe `interactive_start_training.sh` allows you to train Flux1-dev, Wan 2.1, SDXL, Qwen Image, Qwen 2512 and Z-Image Turbo & Base models.

The provided Musubi-tuner scripts will allow you to train with Qwen Edit-2511, Qwen 2512, Z-Image Base & ostris' De-Turbo, Wan 2.2 and FLUX.2 [klein] 9B. 
OneTrainer has been added to this image specifically for the use of Prodigy_ADV for stochastic rounding in the training of Z-Image Base loras.

Instructions on how to run each pipeline is in the following folders: <model>_musubi_training, OneTrainer_config and in the root image folder for Diffusion-pipe.

You can use JoyCaption for auto-captioning of images and for videos you can use Qwen2.5-VL. Gemini is also available but requires a tier above the free one. 
OneTrainer's captioner is also available. 

Resume training from the last checkpoint irrespective of the pipeline. Use TensorBoard for graph eval and if you are training with Musubi or OneTrainer
you have the ability to evaluate your outputs by running visual inference for specified checkpoints.

Exclusive to the Musubi scripts, you can apply Post-Hoc EMA merge for a range of trained steps to get the 'perfect' LoRA model by injecting a beta value.

This template has provisions for deployment to ephemeral and persistent storage environments. An OpenSSH server is included for secure transfer of data.
The image comes with installed `rclone` for transfers to and from Google Drive. Check the configuration script in the root directory on how to set it up.

Pro tip: If you are not initializing locally, it is highly recommended you run all training through `tmux` sessions. Navigate to the script folder then:<br>
`tmux new-session "bash -c 'bash setup_and_train_qwen.sh; exec bash'"`


#### Deploy:
  RunPod  - https://bit.ly/4tCnool
  Vast.ai - https://bit.ly/4c8eSb6


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
| `HF_TOKEN=""`       | Hugging Face API key (required for Flux models) |
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


**Happy training!**
