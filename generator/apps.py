# generator/apps.py (CORRECTED)
from django.apps import AppConfig

class GeneratorConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'generator'
    
    def ready(self):
        # Import any signals here if needed in the future
        pass