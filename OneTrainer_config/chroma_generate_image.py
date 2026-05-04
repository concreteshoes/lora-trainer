import argparse
import gc
import logging
import os
import sys
import time
import threading
import itertools
import torch
from PIL import Image
from tqdm import tqdm
from diffusers import ChromaPipeline

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class Spinner:
    """Context manager for a threaded console spinner."""

    def __init__(self, message="Loading"):
        self.message = message
        # Smooth braille spinner characters
        self.spinner_cycle = itertools.cycle(
            ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'])
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._spin, daemon=True)

    def _spin(self):
        while not self.stop_event.is_set():
            sys.stdout.write(f"\r{next(self.spinner_cycle)} {self.message}")
            sys.stdout.flush()
            time.sleep(0.1)
        # Clear the line when done
        sys.stdout.write(f"\r{' ' * (len(self.message) + 2)}\r")
        sys.stdout.flush()

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_event.set()
        self.thread.join()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chroma batch inference script for OneTrainer LoRAs")

    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to Chroma1-HD HuggingFace snapshot directory")
    parser.add_argument("--lora_weight", type=str, default=None,
                        help="Path to LoRA .safetensors file")
    parser.add_argument("--lora_multiplier", type=float, default=1.0,
                        help="LoRA scale / multiplier. Default is 1.0")

    # Batch mode — prompt file with lines of: prompt text|seed
    parser.add_argument("--prompt_file", type=str, default=None,
                        help="Path to file containing prompts, one per line as 'prompt|seed'")
    # Single prompt mode (kept for manual use)
    parser.add_argument("--prompt", type=str, default=None,
                        help="Single prompt for generation")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed for single prompt mode")

    parser.add_argument("--image_size", type=int, nargs=2, default=[1024, 1024],
                        help="Image size as height width. Default is 1024 1024")
    parser.add_argument("--infer_steps", type=int, default=30,
                        help="Number of inference steps. Default is 30")
    parser.add_argument("--save_path", type=str, required=True,
                        help="Directory to save generated images")
    parser.add_argument("--output_prefix", type=str, default="chroma",
                        help="Prefix for output filenames. Saves as prefix_seed_N.png")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use. Defaults to cuda if available")

    args = parser.parse_args()

    if args.prompt_file is None and args.prompt is None:
        raise ValueError("Either --prompt_file or --prompt must be specified")

    return args


def load_pipeline(args: argparse.Namespace) -> ChromaPipeline:
    """Load ChromaPipeline by constructing from individual component directories."""
    from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler
    from diffusers.models import ChromaTransformer2DModel
    from transformers import T5EncoderModel, T5TokenizerFast

    dtype = torch.bfloat16
    model_path = args.model_path

    logger.info(f"Loading Chroma components from: {model_path}")

    logger.info("Loading tokenizer...")
    tokenizer = T5TokenizerFast.from_pretrained(
        model_path, subfolder="tokenizer")

    logger.info("Loading T5 text encoder...")
    # Note: fp8 is not supported for T5 inference — layer norm mixes fp8/float32
    # T5 always loads in bf16.
    text_encoder = T5EncoderModel.from_pretrained(
        model_path, subfolder="text_encoder", torch_dtype=dtype
    )

    logger.info("Loading VAE...")
    vae = AutoencoderKL.from_pretrained(
        model_path, subfolder="vae", torch_dtype=dtype)

    logger.info("Loading transformer...")
    transformer = ChromaTransformer2DModel.from_pretrained(
        model_path, subfolder="transformer", torch_dtype=dtype
    )

    logger.info("Loading scheduler...")
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        model_path, subfolder="scheduler"
    )

    logger.info("Assembling ChromaPipeline...")
    pipe = ChromaPipeline(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        vae=vae,
        transformer=transformer,
        scheduler=scheduler,
    )

    return pipe


def load_lora(pipe: ChromaPipeline, lora_path: str, multiplier: float) -> ChromaPipeline:
    """Load and apply LoRA weights manually using underscore-to-module reverse lookup."""
    from safetensors.torch import load_file
    from tqdm import tqdm  # Make sure tqdm is imported

    logger.info(f"Loading LoRA from: {lora_path}")
    logger.info(f"LoRA multiplier: {multiplier}")

    state_dict = load_file(lora_path)
    transformer = pipe.transformer

    # Build reverse lookup: underscore_path -> (dot_path, module)
    module_lookup = {}
    for dot_path, module in transformer.named_modules():
        if dot_path == "":
            continue
        underscore_path = dot_path.replace(".", "_")
        module_lookup[underscore_path] = (dot_path, module)

    # Collect lora pairs keyed by their module underscore path
    lora_data = {}
    prefix = "lora_transformer_"
    for key, tensor in state_dict.items():
        for suffix in (".lora_down.weight", ".lora_up.weight", ".alpha"):
            if key.endswith(suffix):
                module_underscore = key[len(prefix):-len(suffix)]
                lora_data.setdefault(module_underscore, {})[
                    suffix.lstrip(".")] = tensor
                break

    applied = 0
    skipped = 0

    # ---> ADDED TQDM HERE <---
    for module_underscore, weights in tqdm(lora_data.items(), desc="Patching LoRA weights", unit="module"):
        if "lora_down.weight" not in weights or "lora_up.weight" not in weights:
            skipped += 1
            continue

        if module_underscore not in module_lookup:
            skipped += 1
            continue

        dot_path, module = module_lookup[module_underscore]

        if not hasattr(module, "weight") or module.weight is None:
            skipped += 1
            continue

        lora_down = weights["lora_down.weight"].to(
            dtype=module.weight.dtype, device=module.weight.device
        )
        lora_up = weights["lora_up.weight"].to(
            dtype=module.weight.dtype, device=module.weight.device
        )

        if "alpha" in weights:
            rank = lora_down.shape[0]
            alpha = weights["alpha"].item()
            scale = (alpha / rank) * multiplier
        else:
            scale = multiplier

        if lora_down.dim() == 2 and lora_up.dim() == 2:
            delta = lora_up @ lora_down
        elif lora_down.dim() == 4:
            delta = (lora_up.flatten(1) @ lora_down.flatten(1)
                     ).view(module.weight.shape)
        else:
            skipped += 1
            continue

        with torch.no_grad():
            module.weight.data += (scale * delta).to(module.weight.dtype)

        applied += 1

    logger.info(
        f"\nLoRA applied: {applied} modules patched, {skipped} skipped")
    if applied == 0:
        logger.error(
            "No modules were patched — LoRA had no effect. Check key format.")

    return pipe


def generate_image(pipe: ChromaPipeline, prompt: str, seed: int,
                   height: int, width: int, infer_steps: int,
                   device: torch.device) -> Image.Image:
    """Run inference for a single prompt and return PIL image."""
    generator = torch.Generator(device=device).manual_seed(seed)

    with torch.inference_mode():
        result = pipe(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=infer_steps,
            generator=generator,
        )

    return result.images[0]


def main():
    args = parse_args()

    device_str = args.device if args.device is not None else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    device = torch.device(device_str)
    logger.info(f"Using device: {device}")

    os.makedirs(args.save_path, exist_ok=True)

    # Build prompt list
    if args.prompt_file is not None:
        with open(args.prompt_file, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
        prompts = []
        for line in lines:
            if "|" in line:
                text, seed_str = line.rsplit("|", 1)
                prompts.append((text.strip(), int(seed_str.strip())))
            else:
                prompts.append((line, 42))
    else:
        prompts = [(args.prompt, args.seed)]

    height, width = args.image_size
    total = len(prompts)
    logger.info(f"Total prompts to generate: {total}")

    # Load and Move
    pipe = load_pipeline(args)
    with Spinner(f"Moving 8.9B parameters to {device}..."):
        pipe = pipe.to(device)

    # Apply LoRA (Now nearly instant)
    if args.lora_weight is not None:
        pipe = load_lora(pipe, args.lora_weight, args.lora_multiplier)

    # Standardize the progress bars
    pipe.set_progress_bar_config(disable=False, leave=False)

    pbar = tqdm(prompts, desc="Total Batch Progress", unit="img")
    for idx, (prompt, seed) in enumerate(pbar, 1):
        multiplier_str = f"{args.lora_multiplier:.1f}".replace(".", "x")
        filename = f"{args.output_prefix}_m{multiplier_str}_seed_{seed}.png"
        save_path = os.path.join(args.save_path, filename)

        if os.path.exists(save_path):
            continue

        pbar.set_description(f"Image {idx}/{total} (Seed {seed})")

        # Generation happens here using the best available kernel (FA2 via SDPA)
        image = generate_image(pipe, prompt, seed, height, width,
                               args.infer_steps, device)
        image.save(save_path)

    # Cleanup
    del pipe
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info("All done!")


if __name__ == "__main__":
    main()
