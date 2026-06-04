# LoRA Trainer using Diffusion-pipe, Musubi-Tuner & OneTrainer w/ Flash & Sage Attn for CUDA 12.8
## Quick Start Guide

This is a LoRA trainer template featuring Flux, Chroma1 HD, SDXL, Wan, LTX 2.3, Qwen and Z-Image models using diffusion-pipe, Musubi training scripts and OneTrainer.

The script `diffusion_pipe_training.sh` allows you to train models: Flux1-dev, Wan 2.1, SDXL, Qwen Image, Z-Image Turbo v2 adaptor, LTX 2.3 (no audio).

The Musubi-tuner scripts will allow you to train with Qwen Edit-2511, Qwen 2512, Z-Image Base & ostris' De-Turbo, Wan 2.2, LTX 2.3 and FLUX.2 [klein] 9B. 
With the supplied OneTrainer configs you can train with Z-Image Base (for the use of Prodigy_ADV with stochastic rounding), and with Chroma1 HD. Default config
exists in case you want to try some other models.

Instructions on how to run each pipeline is in the following folders: <model>_musubi_training, OneTrainer_config and for diffusion-pipe - in the root image folder.

You can use JoyCaption for auto-captioning of images and for videos you can use Qwen2.5-VL. Gemini is also available but requires a tier above the free one. 
OneTrainer's captioner is also available. 

The provided scripts will let you resume training from a checkpoint irrespective of the pipeline you choose. Use TensorBoard for graph eval and if you are training with Musubi, OneTrainer and diffusion-pipe (LTX 2.3 only) you have the ability to evaluate your lora by running visual inference.

Exclusive to the Musubi scripts, you can apply Post-Hoc EMA merge for a range of trained steps to get the 'perfect' LoRA model by injecting beta and sigrel values.

This template has provisions for deployment to ephemeral and persistent storage environments. An OpenSSH server is included for secure transfer of data.
The image comes with installed rclone with a setup script for transfers to and from Google Drive. Check the configuration script in the root directory on how to set it up.
Other notable packages are JupyterLab, Filebrowser and tmux.

Pro tip: If you are not initializing locally, it is highly recommended you run all training through `tmux` sessions. Navigate to the script folder then:<br>
`tmux new-session "bash -c 'bash setup_and_train_<model>.sh; exec bash'"`


### Deploy:
- RunPod  - https://tinyurl.com/y4xn6v8u
- Vast.ai - https://tinyurl.com/2btvp8y7


⚠️ Hardware Requirements

Take notice of the size of the models you want to train with and allocate your storage capacity accordingly.

GPU: NVIDIA Ampere architecture or newer is required (RTX 30-series, 40-series, A100, H100, etc.).


##### Note: If you run into bugs, report them to me on discord: bytesizelife

### ⚖️ License & Usage

This project is licensed under AGPL-3.0. Additionally, commercial redistribution — 
including paywalling access to this image or derivative works — is not permitted 
without explicit written permission from the author.

See [LICENSE](LICENSE) for full terms.

---

### Environmental Variables

| Variable | Description |
|---|---|
| `HF_TOKEN=""`       | Hugging Face API key (required for Flux models) |
| `GEMINI_API_KEY=""` | Gemini API key (required for video processing) |
| `SSH_PUBLIC_KEY=""` | Add your public key if you want SSH transfers |
| `FB_PASSWORD=""`    | Choose a Filebrowser pass (optional) |


### Ports

- `8080` - Filebrowser
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

#### Example: sync local dataset to remote
```bash
rsync -avP -e "ssh -p <SSH_PORT>" /path/to/local/dataset/ hostname@<SERVER_IP>:/path/to/remote/dataset/
```
#### SSH with port forwarding for Filebrowser:
```bash
ssh -p <SSH_PORT> hostname@<SERVER_IP> -L 8080:localhost:8080
```
Then open your browser to:
http://localhost:8080

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
