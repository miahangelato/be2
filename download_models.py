#!/usr/bin/env python
"""
Pre-deployment script to download and cache models
Run this during deployment to avoid slow first startup
"""
import os
import sys

# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set Django settings module
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.settings')

def download_models():
    """Download all models during deployment"""
    print("=== Pre-downloading models for production deployment ===")
    
    try:
        # Setup Django
        import django
        django.setup()
        
        # Download fingerprint model
        print("1. Loading fingerprint classifier...")
        try:
            from core.fingerprint_classifier_utils import model
            print("✅ Fingerprint model loaded and cached")
        except Exception as e:
            print(f"⚠️  Fingerprint model failed: {e}")
        
        # Download blood group model
        print("2. Loading blood group classifier...")
        try:
            from core.bloodgroup_classifier import BloodGroupClassifier
            classifier = BloodGroupClassifier()
            print("✅ Blood group model loaded and cached")
        except Exception as e:
            print(f"⚠️  Blood group model failed: {e}")
        
        print("🎉 Model download process completed!")
        
    except Exception as e:
        print(f"❌ Error during model download: {e}")
        # Don't fail deployment if models can't be downloaded
        print("Continuing with deployment - models will be downloaded on first use")

if __name__ == "__main__":
    download_models()
