# generator/chains.py
import os
import logging
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

logger = logging.getLogger(__name__)

class StoryChains:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.story_llm = None
        self.image_prompt_llm = None
        self._initialize_models()
        
    def _initialize_models(self):
        """Initialize the language models"""
        try:
            # Use a smaller, efficient model for story generation
            model_name = "microsoft/DialoGPT-medium"
            
            # Load tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )
            
            # Create pipeline
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_length=512,
                temperature=0.7,
                do_sample=True,
                device=0 if self.device == "cuda" else -1
            )
            
            # Create HuggingFace pipeline wrapper
            self.story_llm = HuggingFacePipeline(pipeline=pipe)
            self.image_prompt_llm = self.story_llm  # Reuse same model
            
            logger.info("Models initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            # Fallback to a simpler approach
            self._initialize_fallback_models()
    
    def _initialize_fallback_models(self):
        """Fallback initialization with simpler models"""
        try:
            model_name = "gpt2"
            pipe = pipeline(
                "text-generation",
                model=model_name,
                max_length=256,
                temperature=0.8,
                device=0 if self.device == "cuda" else -1
            )
            self.story_llm = HuggingFacePipeline(pipeline=pipe)
            self.image_prompt_llm = self.story_llm
            logger.info("Fallback models initialized")
        except Exception as e:
            logger.error(f"Error with fallback models: {e}")
    
    def generate_story_and_descriptions(self, user_prompt):
        """Generate story, character description, and background description"""
        
        story_template = """
        Based on this prompt: {user_prompt}
        
        Write a short, engaging story (2-3 paragraphs):
        """
        
        character_template = """
        Based on this story prompt: {user_prompt}
        
        Describe the main character in detail for image generation. Include:
        - Physical appearance (age, gender, build, hair, eyes, skin)
        - Clothing and accessories
        - Facial expression and pose
        - Art style: digital art, realistic, fantasy art
        
        Character description:
        """
        
        background_template = """
        Based on this story prompt: {user_prompt}
        
        Describe the scene/background in detail for image generation. Include:
        - Location and environment
        - Time of day and lighting
        - Weather and atmosphere
        - Colors and mood
        - Art style: digital art, realistic, fantasy art
        
        Background description:
        """
        
        try:
            # Create chains
            story_chain = LLMChain(
                llm=self.story_llm,
                prompt=PromptTemplate(template=story_template, input_variables=["user_prompt"])
            )
            
            character_chain = LLMChain(
                llm=self.image_prompt_llm,
                prompt=PromptTemplate(template=character_template, input_variables=["user_prompt"])
            )
            
            background_chain = LLMChain(
                llm=self.image_prompt_llm,
                prompt=PromptTemplate(template=background_template, input_variables=["user_prompt"])
            )
            
            # Generate outputs
            story = story_chain.run(user_prompt=user_prompt).strip()
            character_desc = character_chain.run(user_prompt=user_prompt).strip()
            background_desc = background_chain.run(user_prompt=user_prompt).strip()
            
            return {
                'story': self._clean_text(story),
                'character_description': self._clean_text(character_desc),
                'background_description': self._clean_text(background_desc)
            }
            
        except Exception as e:
            logger.error(f"Error generating story and descriptions: {e}")
            return self._generate_fallback_content(user_prompt)
    
    def _clean_text(self, text):
        """Clean and format generated text"""
        # Remove unwanted tokens and clean up
        text = text.replace('<|endoftext|>', '').strip()
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        return '\n'.join(lines)
    
    def _generate_fallback_content(self, user_prompt):
        """Generate simple fallback content when models fail"""
        return {
            'story': f"In a world inspired by '{user_prompt}', our hero embarks on an extraordinary adventure. Through challenges and discoveries, they learn valuable lessons and grow stronger. The journey transforms them in ways they never expected, leading to a satisfying conclusion.",
            'character_description': f"A determined protagonist with expressive eyes and confident posture, wearing appropriate attire for their adventure. Digital art style, detailed character design, fantasy realistic.",
            'background_description': f"A vivid scene that matches '{user_prompt}' with rich colors and atmospheric lighting. Detailed environment, fantasy art style, cinematic composition."
        }