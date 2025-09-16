#!/usr/bin/env python
"""
Simple Railway Startup Script
Starts Django without immediately loading ML models
"""
import os
import sys
import time
import django
from django.core.management import execute_from_command_line

def main():
    """Run administrative tasks and start the server."""
    
    # Set the default settings module
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.settings')
    
    # Detect Railway environment
    is_railway = bool(os.environ.get('RAILWAY_ENVIRONMENT') or 
                     os.environ.get('RAILWAY_PROJECT_ID') or
                     os.environ.get('RAILWAY_SERVICE_NAME'))
    
    # Set deployment flag for settings
    if is_railway:
        os.environ['RAILWAY_DEPLOYMENT'] = 'True'
        # Suppress TensorFlow warnings in production
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
        print("🚀 Railway Environment Detected")
    
    # Get port from environment (Railway provides this)
    port = os.environ.get('PORT', '8000')
    
    print(f"🌟 Starting Django server on port {port}")
    print(f"🔧 Debug Mode: {os.environ.get('DEBUG', 'False')}")
    
    # Check database configuration
    db_url = os.environ.get('DATABASE_URL')
    if db_url:
        print(f"📊 Database URL configured: {db_url[:20]}...")
    else:
        print("⚠️ No DATABASE_URL found - check Railway environment variables")
    
    # Setup Django
    try:
        django.setup()
        print("✅ Django setup completed")
    except Exception as e:
        print(f"❌ Django setup failed: {e}")
        sys.exit(1)
    
    # Run database migrations first
    print("📊 Running database migrations...")
    try:
        execute_from_command_line(['manage.py', 'migrate', '--noinput'])
        print("✅ Database migrations completed")
    except Exception as e:
        print(f"⚠️ Migration warning: {e}")
    
    # Test database connection
    try:
        from django.db import connection
        connection.ensure_connection()
        print("✅ Database connection successful")
    except Exception as db_error:
        print(f"⚠️ WARNING: Database connection test failed: {db_error}")
    
    # Create static directories if they don't exist
    try:
        from django.conf import settings
        static_dirs = getattr(settings, 'STATICFILES_DIRS', [])
        for static_dir in static_dirs:
            if not os.path.exists(static_dir):
                os.makedirs(static_dir, exist_ok=True)
                print(f"✅ Created static directory: {static_dir}")
    except Exception as e:
        print(f"⚠️ Could not create static directories: {e}")
    
    # Collect static files (if needed)
    try:
        execute_from_command_line(['manage.py', 'collectstatic', '--noinput'])
        print("✅ Static files collected")
    except Exception as e:
        print(f"⚠️ Static files warning: {e}")
    
    # Start the server
    if is_railway:
        # Production: Use gunicorn
        import subprocess
        print("🔥 Starting Gunicorn server...")
        try:
            # Use simpler gunicorn configuration
            subprocess.run([
                'gunicorn', 
                '--bind', f'0.0.0.0:{port}',
                '--workers', '1',  # Reduced workers to save memory
                '--timeout', '300',  # Increased timeout for model loading
                '--max-requests', '1000',
                '--max-requests-jitter', '100',
                '--preload',  # Preload app to avoid repeated model loading
                'backend.wsgi:application'
            ], check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ ERROR starting Gunicorn: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            sys.exit(1)
    else:
        # Development: Use Django dev server
        execute_from_command_line(['manage.py', 'runserver', f'0.0.0.0:{port}'])

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Shutting down gracefully...")
        sys.exit(0)
    except Exception as e:
        print(f"💥 Fatal error: {e}")
        sys.exit(1)