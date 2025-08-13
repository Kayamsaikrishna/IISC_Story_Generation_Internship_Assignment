# generator/forms.py
from django import forms

class StoryPromptForm(forms.Form):
    text_prompt = forms.CharField(
        widget=forms.Textarea(attrs={
            'class': 'form-control',
            'rows': 4,
            'placeholder': 'Enter your story prompt here... (e.g., "A brave knight in a mystical forest")'
        }),
        label='Story Prompt',
        required=False
    )
    
    audio_file = forms.FileField(
        widget=forms.FileInput(attrs={
            'class': 'form-control',
            'accept': '.wav,.mp3,.m4a,.ogg'
        }),
        label='Or upload an audio file',
        required=False
    )
    
    def clean(self):
        cleaned_data = super().clean()
        text_prompt = cleaned_data.get('text_prompt')
        audio_file = cleaned_data.get('audio_file')
        
        if not text_prompt and not audio_file:
            raise forms.ValidationError("Please provide either a text prompt or an audio file.")
        
        return cleaned_data