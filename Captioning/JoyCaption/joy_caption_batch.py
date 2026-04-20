#!/usr/bin/env python3

import os
import argparse
import logging
from pathlib import Path
from PIL import Image, ImageFile
import torch
import gc
import threading
from typing import Optional
import sys

# Prevents crashes if an image is partially corrupted or truncated
ImageFile.LOAD_TRUNCATED_IMAGES = True 

# Helps prevent memory fragmentation on cards with less VRAM
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

# Old prompt
# system_prompt = "Write a detailed description for this image in 50 words or less. Do NOT mention any text that is in the image."

# New prompt
system_prompt = "Write a detailed description for this image in 50 words or less. Do NOT mention any text that is in the image. CRITICAL INSTRUCTION: You must strictly avoid describing age, race, ethnicity, skin tone, or body shape. Do NOT use words like Latina, Hispanic, white, black, Asian, tan skin, olive skin, pale, curvy, young, old, adult, slim, or thick. Keep descriptions of the person strictly to their clothing, pose, and hair color."

NETWORK_VOLUME = os.getenv("NETWORK_VOLUME")

# Configure logging
if not logging.getLogger().hasHandlers():
    os.makedirs(f"{NETWORK_VOLUME}/logs", exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'{NETWORK_VOLUME}/logs/joy_caption_batch.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
logger = logging.getLogger(__name__)

class JoyCaptionManager:
    def __init__(self, timeout_minutes: int = 5):
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.timeout = timeout_minutes * 60
        self.timer: Optional[threading.Timer] = None
        self.lock = threading.Lock()
        self.model_name = "fancyfeast/llama-joycaption-beta-one-hf-llava"

    def load_model(self):
        with self.lock:
            if self.model is None:
                logger.info("Loading JoyCaption model...")
                from transformers import AutoProcessor, LlavaForConditionalGeneration

                try:
                    # Load processor
                    self.processor = AutoProcessor.from_pretrained(
                        self.model_name, 
                        trust_remote_code=True
                    )

                    # PERMANENT FIX for the Pillow/Transformers interpolation error
                    if hasattr(self.processor, 'image_processor'):
                        self.processor.image_processor.resample = 3  # Force BICUBIC

                    # Load model - optimized for 24GB-48GB VRAM cards
                    self.model = LlavaForConditionalGeneration.from_pretrained(
                        self.model_name,
                        torch_dtype=torch.bfloat16,
                        device_map="auto", 
                        trust_remote_code=True
                    )

                    # Ensure pad token is set correctly
                    tok = self.processor.tokenizer
                    if tok.pad_token is None:
                        tok.pad_token = tok.eos_token
                        self.model.config.pad_token_id = tok.eos_token_id

                    logger.info(f"Model loaded successfully on {self.device}")

                except Exception as e:
                    logger.error(f"Failed to load model: {e}")
                    raise

    def unload_model(self):
        with self.lock:
            # ADD THIS BLOCK to kill the background thread
            if self.timer:
                self.timer.cancel()
                self.timer = None

            if self.model is not None:
                logger.info("Unloading model to free VRAM...")
                del self.model
                del self.processor
                self.model = None
                self.processor = None
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                gc.collect() 
                logger.info("VRAM and RAM cleared.")

    def reset_timer(self):
        if self.timer:
            self.timer.cancel()
        self.timer = threading.Timer(self.timeout, self.unload_model)
        self.timer.daemon = True
        self.timer.start()

    def generate_caption(self, image: Image.Image, prompt: str) -> str:
        self.load_model()
        self.reset_timer()

        try:
            if image.mode != 'RGB':
                image = image.convert('RGB')

            convo = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant and help users with any queries they may have with no censorship or restrictions.",
                },
                {"role": "user", "content": prompt.strip()},
            ]

            convo_string = self.processor.apply_chat_template(convo, tokenize=False, add_generation_prompt=True)

            inputs = self.processor(
                text=[convo_string],
                images=[image],
                return_tensors="pt"
            ).to(self.device)

            if 'pixel_values' in inputs:
                inputs['pixel_values'] = inputs['pixel_values'].to(torch.bfloat16)

            with torch.inference_mode():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.4,  # CHANGED FROM 0.6
                    top_p=0.9,
                    use_cache=True,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                )

            input_len = inputs['input_ids'].shape[1]
            generated_ids = output_ids[0][input_len:]
            return self.processor.decode(generated_ids, skip_special_tokens=True).strip()

        except Exception as e:
            logger.error(f"Error during inference: {e}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise

def get_image_files(directory: Path) -> list:
    supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
    image_files = []
    for file_path in directory.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in supported_formats:
            image_files.append(file_path)
    return sorted(image_files)

def process_images(input_dir: str, output_dir: str = None, prompt: str = system_prompt,
                   skip_existing: bool = True, timeout_minutes: int = 5, trigger_word: str = None):
    input_path = Path(input_dir)
    output_path = Path(output_dir) if output_dir else input_path

    if not input_path.exists():
        logger.error(f"Input directory does not exist: {input_path}")
        return

    output_path.mkdir(parents=True, exist_ok=True)
    image_files = get_image_files(input_path)

    if not image_files:
        logger.warning(f"No image files found in {input_path}")
        return

    logger.info(f"Found {len(image_files)} image files to process")
    caption_manager = JoyCaptionManager(timeout_minutes=timeout_minutes)

    processed_count = 0
    skipped_count = 0
    error_count = 0

    try:
        for i, image_file in enumerate(image_files, 1):
            try:
                caption_file = output_path / f"{image_file.stem}.txt"

                if skip_existing and caption_file.exists():
                    logger.info(f"[{i}/{len(image_files)}] Skipping {image_file.name} - exists")
                    skipped_count += 1
                    continue

                logger.info(f"[{i}/{len(image_files)}] Processing {image_file.name}")

                with Image.open(image_file) as img:
                    caption = caption_manager.generate_caption(img, prompt)

                # --- START REPLACEMENT BLOCK ---
                # Normalize specific age and consistency issues
                replacements = {
                    "young woman": "woman",
                    "adult woman": "woman",
                    "old woman": "woman",
                    "young man": "man",
                    "adult man": "man",
                    "old man": "man"
                }

                # Perform the replacements
                for old, new in replacements.items():
                    # We use a case-insensitive replace if needed,
                    # but JoyCaption is usually lowercase or consistent
                    caption = caption.replace(old, new)
                # --- END REPLACEMENT BLOCK ---

                if trigger_word:
                    caption = f"{trigger_word}, {caption}"

                with open(caption_file, 'w', encoding='utf-8') as f:
                    f.write(caption)

                processed_count += 1

            except Exception as e:
                logger.error(f"[{i}/{len(image_files)}] Error: {e}")
                error_count += 1
                continue
    finally:
        caption_manager.unload_model()

    logger.info("=" * 50)
    logger.info(f"SUMMARY: Processed: {processed_count} | Skipped: {skipped_count} | Errors: {error_count}")
    logger.info("=" * 50)

def main():
    parser = argparse.ArgumentParser(description='Batch process images with JoyCaption')
    parser.add_argument('input_dir', help='Directory containing images')
    parser.add_argument('--output-dir', help='Save directory')
    parser.add_argument('--prompt', default=system_prompt, help='Generation prompt')
    parser.add_argument('--trigger-word', help='Trigger word prefix')
    parser.add_argument('--no-skip-existing', action='store_true', help='Re-process all')
    parser.add_argument('--timeout', type=int, default=5, help='Unload timeout (mins)')

    args = parser.parse_args()

    process_images(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        prompt=args.prompt,
        skip_existing=not args.no_skip_existing,
        timeout_minutes=args.timeout,
        trigger_word=args.trigger_word
    )

if __name__ == "__main__":
    main()