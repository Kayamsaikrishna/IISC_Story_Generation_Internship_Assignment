# README.md
# IISc Aerospace Engineering Internship - Story Generator

A Django web application powered by LangChain that generates creative stories with character and background images using AI models.

## Features

- **Story Generation**: Creates engaging short stories from user prompts
- **Character Creation**: Generates detailed character descriptions and images  
- **Scene Building**: Creates background descriptions and scene images
- **Image Composition**: Combines character and background into unified scenes
- **Audio Support**: Optional audio input with speech-to-text conversion
- **GPU Acceleration**: Optimized for NVIDIA RTX 4060 8GB GPU

## System Requirements

- **OS**: Windows 11 (tested), Linux, macOS
- **GPU**: NVIDIA RTX 4060 8GB (recommended) or CPU fallback
- **RAM**: 16GB+ recommended
- **Python**: 3.8+
- **Storage**: ~10GB for models

## Installation

### 1. Clone/Create Project
```bash
mkdir story_generator
cd story_generator
```

### 2. Create Virtual Environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/macOS  
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download Models
```bash
python setup_models.py
```

### 5. Setup Django
```bash
python manage.py migrate
python manage.py createsuperuser  # Optional
```

### 6. Run Server
```bash
python manage.py runserver
```

Visit `http://127.0.0.1:8000` to use the application.

## Project Structure

```
story_generator/
├── models/                 # Downloaded AI models (local storage)
├── media/                  # Generated images and uploads  
├── static/                 # CSS, JavaScript, assets
├── templates/              # HTML templates
├── generator/              # Main Django app
│   ├── chains.py          # LangChain orchestration
│   ├── image_generator.py # Image generation logic
│   ├── views.py           # Web interface logic
│   └── models.py          # Database models
├── requirements.txt       # Python dependencies
├── setup_models.py       # Model download script
└── README.md             # This file
```

## Architecture Overview

### LangChain Orchestration
- **Story Chain**: Generates narrative content from user prompts
- **Character Chain**: Creates detailed character descriptions
- **Background Chain**: Generates scene/environment descriptions
- **Modular Design**: Separate chains for different content types

### Image Generation Pipeline
1. **Character Image**: Generated from character description
2. **Background Image**: Generated from scene description  
3. **Image Composition**: Combines both using PIL/OpenCV
4. **Model**: Stable Diffusion v1.5 (local, no API costs)

### Prompt Engineering
- **Story Prompts**: Structured templates for narrative generation
- **Image Prompts**: Enhanced descriptions for visual consistency
- **Iterative Refinement**: Multiple prompt variations for quality
- **Style Consistency**: Matching art styles across components

## Key Features

### Web Interface
- Clean, responsive Bootstrap design
- Real-time form validation
- Progress indicators during generation
- Image gallery for results
- Mobile-friendly layout

### AI Models Used
- **Text Generation**: Microsoft DialoGPT-medium (fallback: GPT-2)
- **Image Generation**: Stable Diffusion v1.5
- **Speech Recognition**: Google Speech Recognition API
- **All models run locally** - no subscription APIs

### Error Handling & Robustness
- Graceful model loading fallbacks
- Comprehensive logging system
- Input validation and sanitization
- Placeholder generation on failures
- GPU/CPU automatic detection

## Generated Content Examples

### Input Prompt
"A brave knight in a mystical forest"

### Outputs
- **Story**: 2-3 paragraph narrative about the knight's adventure
- **Character**: "A noble knight in shining armor with determined eyes..."
- **Background**: "Ancient mystical forest with glowing trees and morning mist..."
- **Images**: Character image, background image, and combined scene

## Technical Details

### GPU Optimization
- Automatic CUDA detection and usage
- Memory-efficient attention mechanisms  
- Mixed precision for faster inference
- Batch processing optimization

### Model Storage
- All models stored in local `models/` directory
- No system-wide installation required
- Portable setup across different machines
- Automatic model downloading on first run

### Performance
- **Story Generation**: ~10-15 seconds
- **Image Generation**: ~20-30 seconds per image
- **Total Processing**: ~60-90 seconds per request
- **GPU vs CPU**: ~3-4x speed improvement with GPU

## Deployment Notes

### Development
```bash
python manage.py runserver
```

### Production Considerations
- Set `DEBUG = False` in settings
- Configure proper `SECRET_KEY`
- Use production database (PostgreSQL)
- Setup static file serving
- Configure proper logging

### Docker Support (Optional)
Create `Dockerfile` for containerized deployment:
```dockerfile
FROM python:3.9
COPY requirements.txt .
RUN pip install -r requirements.txt
# ... rest of Docker setup
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size in image generation
   - Enable memory efficient attention
   - Use CPU fallback

2. **Model Download Failures**
   - Check internet connection
   - Verify Hugging Face access
   - Use VPN if needed

3. **Audio Processing Issues**
   - Install additional audio codecs
   - Check microphone permissions
   - Verify audio file formats

### Performance Optimization
- Use SSD storage for models
- Increase system RAM if possible
- Close unnecessary applications during generation
- Use latest GPU drivers

## API Documentation

### Main Endpoints
- `GET /` - Main interface
- `POST /generate/` - Generate story and images
- `GET /story/<id>/` - View generated story

### Database Models
- `StoryGeneration` - Stores all generated content and metadata

## Contributing

1. Follow Django best practices
2. Add comprehensive logging
3. Include error handling
4. Test on both GPU and CPU
5. Document prompt engineering decisions

## License & Usage

This project is created for the IISc Aerospace Engineering Internship assignment. Follow the submission guidelines and contact `bsvivek2003@gmail.com` for queries.

## Acknowledgments

- LangChain for AI orchestration framework
- Hugging Face for model hosting and transformers
- Stability AI for Stable Diffusion model
- Django community for web framework

---

**Note**: This application uses only free and open-source AI models as per assignment requirements. No paid APIs or subscriptions are needed.