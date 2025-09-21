from django.core.management.base import BaseCommand
import requests
import os
from django.conf import settings


class Command(BaseCommand):
    help = 'Download ML models from S3'

    def add_arguments(self, parser):
        parser.add_argument(
            '--force',
            action='store_true',
            help='Force re-download even if files exist',
        )

    def handle(self, *args, **options):
        base = settings.BASE_DIR
        
        models_to_download = {
            # Diabetes prediction models
            'diabetes_risk_model.pkl': "https://team3thesis.s3.us-east-1.amazonaws.com/models/backend/core%5Cdiabetes_risk_model.pkl",
            'diabetes_risk_model_columns.pkl': "https://team3thesis.s3.us-east-1.amazonaws.com/models/backend/core%5Cdiabetes_risk_model_columns.pkl",
            
            # Blood group classification model
            'bloodgroup_model_20250823-140933.h5': "https://team3thesis.s3.us-east-1.amazonaws.com/models/backend/core%5Cbloodgroup_model_20250823-140933.h5",
            
            # Fingerprint pattern classification model
            'improved_pattern_cnn_model.h5': "https://team3thesis.s3.us-east-1.amazonaws.com/models/backend/core%5Cimproved_pattern_cnn_model.h5"
        }
        
        for filename, url in models_to_download.items():
            local_path = os.path.join(base, "core", filename)
            
            if os.path.exists(local_path) and not options['force']:
                self.stdout.write(
                    self.style.SUCCESS(f'{filename} already exists. Use --force to re-download.')
                )
                continue
                
            self.stdout.write(f'Downloading {filename}...')
            
            try:
                response = requests.get(url, stream=True)
                response.raise_for_status()
                
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                with open(local_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            
                self.stdout.write(
                    self.style.SUCCESS(f'Successfully downloaded {filename}')
                )
                
            except Exception as e:
                self.stdout.write(
                    self.style.ERROR(f'Failed to download {filename}: {e}')
                )