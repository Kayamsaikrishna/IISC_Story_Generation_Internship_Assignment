# setup_models.py
"""
Script to download and setup required models for the story generator.
Run this script after installing requirements to download all necessary models.
"""

import os
import sys
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from diffusers import StableDiffusionPipeline
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_models():
    """Download and setup all required models"""
    
    # Create models directory
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # 1. Setup Language Model for story generation
    logger.info("Setting up language model...")
    try:
        model_name = "microsoft/DialoGPT-medium"
        
        # Download and save tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.save_pretrained(models_dir / "dialogpt-medium")
        
        # Download and save model
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        )
        model.save_pretrained(models_dir / "dialogpt-medium")
        
        logger.info("✓ Language model downloaded successfully")
        
    except Exception as e:
        logger.error(f"✗ Error downloading language model: {e}")
        logger.info("Falling back to GPT-2...")
        
        # Fallback to GPT-2
        try:
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
            model = AutoModelForCausalLM.from_pretrained("gpt2")
            
            tokenizer.save_pretrained(models_dir / "gpt2")
            model.save_pretrained(models_dir / "gpt2")
            
            logger.info("✓ GPT-2 model downloaded successfully")
        except Exception as e2:
            logger.error(f"✗ Error downloading GPT-2: {e2}")
    
    # 2. Setup Stable Diffusion for image generation
    logger.info("Setting up Stable Diffusion model...")
    try:
        sd_model_path = models_dir / "stable-diffusion-v1-5"
        
        if not sd_model_path.exists():
            # Download Stable Diffusion
            pipe = StableDiffusionPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                safety_checker=None,
                requires_safety_checker=False
            )
            
            # Save the model
            pipe.save_pretrained(sd_model_path)
            logger.info("✓ Stable Diffusion model downloaded successfully")
        else:
            logger.info("✓ Stable Diffusion model already exists")
            
    except Exception as e:
        logger.error(f"✗ Error downloading Stable Diffusion: {e}")
        logger.info("You can still run the application, but image generation may use fallback methods")
    
    # 3. Create necessary directories
    logger.info("Creating directory structure...")
    dirs_to_create = [
        "media/generated_images/characters",
        "media/generated_images/backgrounds", 
        "media/generated_images/combined",
        "media/audio_uploads"
    ]
    
    for dir_path in dirs_to_create:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    logger.info("✓ Directory structure created")
    
    # 4. Test GPU availability
    logger.info("Testing GPU availability...")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"✓ GPU available: {gpu_name} ({gpu_memory:.1f}GB)")
    else:
        logger.info("! No GPU available, using CPU (slower generation)")
    
    logger.info("\n" + "="*50)
    logger.info("Setup complete! You can now run:")
    logger.info("python manage.py migrate")
    logger.info("python manage.py runserver")
    logger.info("="*50)

if __name__ == "__main__":
    setup_models()