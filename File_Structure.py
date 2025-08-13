import os

# Base project directory
base_dir = "story_generator"

# Folder structure
folders = [
    f"{base_dir}/models",
    f"{base_dir}/media/generated_images",
    f"{base_dir}/media/audio_uploads",
    f"{base_dir}/static/css",
    f"{base_dir}/static/js",
    f"{base_dir}/templates",
    f"{base_dir}/story_generator",
    f"{base_dir}/generator",
    f"{base_dir}/generator/migrations"
]

# Files to create
files = [
    f"{base_dir}/manage.py",
    f"{base_dir}/requirements.txt",
    f"{base_dir}/README.md",
    f"{base_dir}/setup_models.py",
    f"{base_dir}/static/css/style.css",
    f"{base_dir}/static/js/main.js",
    f"{base_dir}/templates/base.html",
    f"{base_dir}/templates/index.html",
    f"{base_dir}/templates/result.html",
    f"{base_dir}/story_generator/__init__.py",
    f"{base_dir}/story_generator/settings.py",
    f"{base_dir}/story_generator/urls.py",
    f"{base_dir}/story_generator/wsgi.py",
    f"{base_dir}/generator/__init__.py",
    f"{base_dir}/generator/admin.py",
    f"{base_dir}/generator/apps.py",
    f"{base_dir}/generator/models.py",
    f"{base_dir}/generator/views.py",
    f"{base_dir}/generator/urls.py",
    f"{base_dir}/generator/forms.py",
    f"{base_dir}/generator/chains.py",
    f"{base_dir}/generator/image_generator.py",
    f"{base_dir}/generator/migrations/__init__.py"
]

# Create folders
for folder in folders:
    os.makedirs(folder, exist_ok=True)

# Create empty files
for file in files:
    with open(file, 'w') as f:
        pass

print("Django project structure created successfully!")
