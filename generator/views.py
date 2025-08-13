# generator/views.py
import logging
import time
import speech_recognition as sr
from django.shortcuts import render, redirect
from django.contrib import messages
from django.http import JsonResponse
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from .forms import StoryPromptForm
from .models import StoryGeneration
from .chains import StoryChains
from .image_generator import ImageGenerator

logger = logging.getLogger(__name__)

def index(request):
    """Main page with form"""
    form = StoryPromptForm()
    recent_generations = StoryGeneration.objects.all()[:5]
    return render(request, 'index.html', {
        'form': form,
        'recent_generations': recent_generations
    })

def generate_story(request):
    """Process form and generate story with images"""
    if request.method != 'POST':
        return redirect('index')
    
    form = StoryPromptForm(request.POST, request.FILES)
    if not form.is_valid():
        messages.error(request, 'Please provide a valid prompt or audio file.')
        return redirect('index')
    
    start_time = time.time()
    
    try:
        # Get user prompt
        user_prompt = form.cleaned_data.get('text_prompt', '')
        audio_file = form.cleaned_data.get('audio_file')
        
        if not user_prompt and audio_file:
            user_prompt = process_audio_file(audio_file)
            if not user_prompt:
                messages.error(request, 'Could not process audio file.')
                return redirect('index')
        
        # Create story generation record
        story_gen = StoryGeneration.objects.create(user_prompt=user_prompt)
        
        # Initialize chains and generators
        story_chains = StoryChains()
        image_generator = ImageGenerator()
        
        # Generate story and descriptions
        logger.info("Generating story and descriptions...")
        story_data = story_chains.generate_story_and_descriptions(user_prompt)
        
        story_gen.story = story_data['story']
        story_gen.character_description = story_data['character_description']
        story_gen.background_description = story_data['background_description']
        story_gen.save()
        
        # Generate images
        logger.info("Generating character image...")
        char_image_path = image_generator.generate_character_image(story_data['character_description'])
        
        logger.info("Generating background image...")
        bg_image_path = image_generator.generate_background_image(story_data['background_description'])
        
        logger.info("Combining images...")
        combined_image_path = image_generator.combine_images(char_image_path, bg_image_path)
        
        # Update record with image paths
        story_gen.character_image = char_image_path
        story_gen.background_image = bg_image_path
        story_gen.combined_image = combined_image_path
        story_gen.processing_time = time.time() - start_time
        story_gen.save()
        
        messages.success(request, 'Story generated successfully!')
        return render(request, 'result.html', {'story_gen': story_gen})
        
    except Exception as e:
        logger.error(f"Error generating story: {e}")
        messages.error(request, 'An error occurred while generating the story.')
        return redirect('index')

def process_audio_file(audio_file):
    """Process uploaded audio file and convert to text"""
    try:
        # Save temporary file
        file_content = ContentFile(audio_file.read())
        temp_path = default_storage.save('temp_audio.wav', file_content)
        full_path = default_storage.path(temp_path)
        
        # Initialize speech recognition
        r = sr.Recognizer()
        
        with sr.AudioFile(full_path) as source:
            audio = r.record(source)
            text = r.recognize_google(audio)
        
        # Clean up temp file
        default_storage.delete(temp_path)
        
        return text
        
    except Exception as e:
        logger.error(f"Error processing audio: {e}")
        return None

def story_detail(request, story_id):
    """View individual story"""
    try:
        story_gen = StoryGeneration.objects.get(id=story_id)
        return render(request, 'result.html', {'story_gen': story_gen})
    except StoryGeneration.DoesNotExist:
        messages.error(request, 'Story not found.')
        return redirect('index')