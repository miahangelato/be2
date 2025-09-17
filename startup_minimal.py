#!/usr/bin/env python3
"""
Railway startup script for Django application
Optimized for Railway deployment with S3 and PostgreSQL
"""
import os
import sys
import logging

# Configure minimal logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Railway startup process"""
    try:
        logger.info("🚀 Starting Railway deployment...")
        
        # Set Django settings
        os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.settings')
        
        # Railway environment variables
        os.environ['PYTHONUNBUFFERED'] = '1'
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logs
        
        # Ensure production mode
        os.environ['DEBUG'] = 'False'
        
        # Import Django after environment setup
        import django
        from django.core.management import call_command
        
        # Setup Django
        django.setup()
        logger.info("✅ Django setup complete")
        
        # Preload ML models from S3 (cache them locally)
        logger.info("🤖 Preloading ML models from S3...")
        try:
            from core.fingerprint_classifier_utils import get_model
            from core.bloodgroup_classifier import BloodGroupClassifier
            
            # This will download and cache the fingerprint model
            fingerprint_model = get_model()
            logger.info("✅ Fingerprint classification model loaded")
            
            # This will download and cache the blood group model  
            blood_classifier = BloodGroupClassifier()
            logger.info("✅ Blood group classification model loaded")
            
            # Try to load diabetes model (requires pandas)
            try:
                from core.diabetes_predictor import DiabetesPredictor
                diabetes_predictor = DiabetesPredictor()
                logger.info("✅ Diabetes prediction model loaded")
            except ImportError as e:
                logger.warning(f"⚠️ Diabetes model skipped (missing pandas): {e}")
            
        except Exception as e:
            logger.warning(f"⚠️ Model preloading warning: {e}")
            logger.info("📝 Models will be downloaded on first request instead")
        
        # Run migrations
        logger.info("📋 Running database migrations...")
        try:
            call_command('migrate', verbosity=1)
            logger.info("✅ Migrations complete")
        except Exception as e:
            logger.error(f"❌ Migration error: {e}")
            # Continue anyway for Railway
        
        # Collect static files (if using S3, this may not be needed)
        logger.info("📁 Collecting static files...")
        try:
            call_command('collectstatic', '--noinput', verbosity=1)
            logger.info("✅ Static files collected")
        except Exception as e:
            logger.warning(f"⚠️ Static files warning: {e}")
            # Continue anyway
        
        # Start Gunicorn server
        port = os.getenv('PORT', '8000')
        logger.info(f"🌐 Starting Gunicorn on port {port}")
        
        # Execute Gunicorn with optimized settings for Railway
        os.execvp('gunicorn', [
            'gunicorn',
            'backend.wsgi:application',
            '--bind', f'0.0.0.0:{port}',
            '--workers', '2',  # Railway has limited memory
            '--worker-class', 'sync',
            '--timeout', '120',
            '--keep-alive', '5',
            '--max-requests', '1000',
            '--max-requests-jitter', '100',
            '--log-level', 'info',
            '--access-logfile', '-',
            '--error-logfile', '-',
            '--capture-output'
        ])
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()