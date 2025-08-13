# generator/image_generator.py
import os
import logging
import torch
from PIL import Image, ImageEnhance, ImageFilter
import cv2
import numpy as np
from diffusers import StableDiffusionPipeline, DiffusionPipeline
from django.conf import settings
import time

logger = logging.getLogger(__name__)

class ImageGenerator:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipe = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the Stable Diffusion model"""
        try:
            model_path = settings.MODELS_DIR / "stable-diffusion-v1-5"
            
            if model_path.exists():
                # Load local model
                self.pipe = StableDiffusionPipeline.from_pretrained(
                    str(model_path),
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    safety_checker=None,
                    requires_safety_checker=False
                )
            else:
                # Download and cache model
                self.pipe = StableDiffusionPipeline.from_pretrained(
                    "runwayml/stable-diffusion-v1-5",
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    safety_checker=None,
                    requires_safety_checker=False,
                    cache_dir=str(settings.MODELS_DIR)
                )
                # Save for future use
                self.pipe.save_pretrained(str(model_path))
            
            self.pipe = self.pipe.to(self.device)
            
            # Enable memory efficient attention
            if self.device == "cuda":
                self.pipe.enable_attention_slicing()
                self.pipe.enable_xformers_memory_efficient_attention()
            
            logger.info(f"Image generation model initialized on {self.device}")
            
        except Exception as e:
            logger.error(f"Error initializing image model: {e}")
            self.pipe = None
    
    def generate_character_image(self, character_description):
        """Generate character image from description"""
        if not self.pipe:
            return self._create_placeholder_image("Character")
        
        try:
            # Enhanced prompt for character generation
            enhanced_prompt = f"portrait of {character_description}, high quality, detailed, digital art, character design, centered composition"
            negative_prompt = "blurry, low quality, distorted, deformed, multiple people, text, watermark"
            
            # Generate image
            with torch.autocast(self.device):
                result = self.pipe(
                    prompt=enhanced_prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=20,
                    guidance_scale=7.5,
                    width=512,
                    height=512,
                    generator=torch.Generator(device=self.device).manual_seed(42)
                )
            
            image = result.images[0]
            
            # Post-process image
            image = self._enhance_image(image)
            
            # Save image
            timestamp = int(time.time())
            filename = f"character_{timestamp}.png"
            filepath = settings.MEDIA_ROOT / "generated_images" / "characters" / filename
            filepath.parent.mkdir(parents=True, exist_ok=True)
            image.save(filepath)
            
            return f"generated_images/characters/{filename}"
            
        except Exception as e:
            logger.error(f"Error generating character image: {e}")
            return self._create_placeholder_image("Character")
    
    def generate_background_image(self, background_description):
        """Generate background image from description"""
        if not self.pipe:
            return self._create_placeholder_image("Background")
        
        try:
            # Enhanced prompt for background generation
            enhanced_prompt = f"landscape, {background_description}, high quality, detailed, digital art, environment design, wide shot"
            negative_prompt = "people, characters, humans, animals, blurry, low quality, text, watermark"
            
            # Generate image
            with torch.autocast(self.device):
                result = self.pipe(
                    prompt=enhanced_prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=20,
                    guidance_scale=7.5,
                    width=512,
                    height=512,
                    generator=torch.Generator(device=self.device).manual_seed(123)
                )
            
            image = result.images[0]
            
            # Post-process image
            image = self._enhance_image(image)
            
            # Save image
            timestamp = int(time.time())
            filename = f"background_{timestamp}.png"
            filepath = settings.MEDIA_ROOT / "generated_images" / "backgrounds" / filename
            filepath.parent.mkdir(parents=True, exist_ok=True)
            image.save(filepath)
            
            return f"generated_images/backgrounds/{filename}"
            
        except Exception as e:
            logger.error(f"Error generating background image: {e}")
            return self._create_placeholder_image("Background")
    
    def combine_images(self, character_path, background_path):
        """Combine character and background images"""
        try:
            char_img = Image.open(settings.MEDIA_ROOT / character_path)
            bg_img = Image.open(settings.MEDIA_ROOT / background_path)
            
            # Resize to same dimensions
            target_size = (1024, 1024)
            char_img = char_img.resize(target_size, Image.Resampling.LANCZOS)
            bg_img = bg_img.resize(target_size, Image.Resampling.LANCZOS)
            
            # Create mask for character (simple approach)
            char_array = np.array(char_img)
            bg_array = np.array(bg_img)
            
            # Create a blend mask (center focus)
            mask = self._create_blend_mask(target_size)
            
            # Blend images
            combined_array = self._blend_images(char_array, bg_array, mask)
            combined_img = Image.fromarray(combined_array)
            
            # Enhance final image
            combined_img = self._enhance_image(combined_img)
            
            # Save combined image
            timestamp = int(time.time())
            filename = f"combined_{timestamp}.png"
            filepath = settings.MEDIA_ROOT / "generated_images" / "combined" / filename
            filepath.parent.mkdir(parents=True, exist_ok=True)
            combined_img.save(filepath, quality=95)
            
            return f"generated_images/combined/{filename}"
            
        except Exception as e:
            logger.error(f"Error combining images: {e}")
            return self._create_placeholder_image("Combined")
    
    def _create_blend_mask(self, size):
        """Create a circular blend mask"""
        mask = np.zeros(size, dtype=np.float32)
        center = (size[0] // 2, size[1] // 2)
        radius = min(size) // 3
        
        y, x = np.ogrid[:size[1], :size[0]]
        mask_circle = ((x - center[0]) ** 2 + (y - center[1]) ** 2) <= radius ** 2
        
        # Create gradient mask
        for i in range(size[1]):
            for j in range(size[0]):
                dist = np.sqrt((j - center[0]) ** 2 + (i - center[1]) ** 2)
                if dist <= radius:
                    mask[j, i] = 1.0
                elif dist <= radius * 1.5:
                    mask[j, i] = 1.0 - (dist - radius) / (radius * 0.5)
        
        return mask
    
    def _blend_images(self, char_array, bg_array, mask):
        """Blend two images using mask"""
        mask_3d = np.stack([mask] * 3, axis=-1)
        blended = char_array * mask_3d + bg_array * (1 - mask_3d)
        return blended.astype(np.uint8)
    
    def _enhance_image(self, image):
        """Enhance image quality"""
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.1)
        
        # Enhance color
        enhancer = ImageEnhance.Color(image)
        image = enhancer.enhance(1.1)
        
        # Slight sharpening
        image = image.filter(ImageFilter.UnsharpMask(radius=1, percent=10, threshold=3))
        
        return image
    
    def _create_placeholder_image(self, image_type):
        """Create a placeholder image when generation fails"""
        try:
            # Create a simple colored placeholder
            colors = {
                "Character": (100, 150, 200),
                "Background": (150, 200, 100),
                "Combined": (200, 150, 100)
            }
            
            img = Image.new('RGB', (512, 512), colors.get(image_type, (128, 128, 128)))
            
            timestamp = int(time.time())
            filename = f"placeholder_{image_type.lower()}_{timestamp}.png"
            
            if image_type == "Character":
                folder = "characters"
            elif image_type == "Background":
                folder = "backgrounds"
            else:
                folder = "combined"
            
            filepath = settings.MEDIA_ROOT / "generated_images" / folder / filename
            filepath.parent.mkdir(parents=True, exist_ok=True)
            img.save(filepath)
            
            return f"generated_images/{folder}/{filename}"
            
        except Exception as e:
            logger.error(f"Error creating placeholder: {e}")
            return None