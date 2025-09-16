#!/usr/bin/env python
"""
Pre-deployment script to download and cache models
Run this during deployment to avoid slow first startup
"""
import os
import django
from django.conf import settings

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.settings')
django.setup()

def download_models():
    """Download all models during deployment"""
    print("=== Pre-downloading models for production deployment ===")
    
    try:
        # Download fingerprint model
        print("1. Loading fingerprint classifier...")
        from core.fingerprint_classifier_utils import model
        print("✅ Fingerprint model loaded and cached")
        
        # Download blood group model
        print("2. Loading blood group classifier...")
        from core.bloodgroup_classifier import BloodGroupClassifier
        classifier = BloodGroupClassifier()
        print("✅ Blood group model loaded and cached")
        
        print("🎉 All models downloaded and ready for production!")
        
    except Exception as e:
        print(f"❌ Error downloading models: {e}")
        raise

if __name__ == "__main__":
    download_models()
