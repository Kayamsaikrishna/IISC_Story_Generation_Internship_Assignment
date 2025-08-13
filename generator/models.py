# generator/models.py
from django.db import models
import uuid

class StoryGeneration(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user_prompt = models.TextField()
    story = models.TextField(blank=True)
    character_description = models.TextField(blank=True)
    background_description = models.TextField(blank=True)
    character_image = models.ImageField(upload_to='generated_images/characters/', blank=True)
    background_image = models.ImageField(upload_to='generated_images/backgrounds/', blank=True)
    combined_image = models.ImageField(upload_to='generated_images/combined/', blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    processing_time = models.FloatField(blank=True, null=True)
    
    class Meta:
        ordering = ['-created_at']