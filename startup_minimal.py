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
        
        # Skip model preloading for now - focus on getting basic app running
        logger.info("🤖 Skipping model preloading - models will download on first request")
        logger.info("📝 This reduces startup time and memory usage during deployment")
        
        # Run migrations
        logger.info("📋 Running database migrations...")
        try:
            call_command('migrate', verbosity=1)
            logger.info("✅ Migrations complete")
        except Exception as e:
            logger.error(f"❌ Migration error: {e}")
            # Continue anyway for Railway
        
        # Skip static files collection for Railway (using S3)
        logger.info("📁 Skipping static files collection (using S3 for static files)")
        logger.info("📝 This prevents startup delays and uses S3 directly")
        
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