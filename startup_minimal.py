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
        
        # Skip model preloading for Railway deployment
        logger.info("📝 Skipping ML model preloading for faster startup")
        logger.info("🤖 Models will be downloaded on first request")
        
        # Run migrations with timeout protection
        logger.info("📋 Running database migrations...")
        try:
            # Set a timeout for migrations
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError("Migration timeout")
            
            # Windows doesn't support SIGALRM, so just run without timeout on Windows
            if os.name != 'nt':
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(30)  # 30 second timeout
            
            call_command('migrate', verbosity=0, run_syncdb=True)
            
            if os.name != 'nt':
                signal.alarm(0)  # Cancel timeout
                
            logger.info("✅ Migrations complete")
        except TimeoutError:
            logger.error("❌ Migration timeout - continuing anyway")
        except Exception as e:
            logger.warning(f"⚠️ Migration warning: {e}")
            # Continue anyway for Railway
        
        # Skip collectstatic for Railway (using S3)
        logger.info("📁 Skipping static files collection (using S3)")
        
        # Start Gunicorn server
        port = os.getenv('PORT', '8000')
        logger.info(f"🌐 Starting Gunicorn on port {port}")
        
        # Execute Gunicorn with Railway-optimized settings
        gunicorn_args = [
            'gunicorn',
            'backend.wsgi:application',
            '--bind', f'0.0.0.0:{port}',
            '--workers', '1',  # Single worker for free tier
            '--worker-class', 'sync',
            '--timeout', '60',  # Shorter timeout
            '--keep-alive', '2',
            '--max-requests', '500',  # Lower for stability
            '--max-requests-jitter', '50',
            '--log-level', 'info',
            '--access-logfile', '-',
            '--error-logfile', '-'
        ]
        
        logger.info(f"🚀 Executing: {' '.join(gunicorn_args)}")
        os.execvp('gunicorn', gunicorn_args)
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()