#!/usr/bin/env python
"""
Railway Startup Script for Django Application
This script handles the startup process for Railway deployment
"""
import os
import sys
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
    
    # Setup Django
    django.setup()
    
    # Import after Django setup
    from django.core.management.commands.runserver import Command as RunServerCommand
    from django.core.wsgi import get_wsgi_application
    
    # Get port from environment (Railway provides this)
    port = os.environ.get('PORT', '8000')
    
    print(f"🚀 Starting Django server on port {port}")
    print(f"🌍 Railway Deployment: {is_railway}")
    print(f"🔧 Debug Mode: {os.environ.get('DEBUG', 'False')}")
    print(f"� Database URL exists: {bool(os.environ.get('DATABASE_URL'))}")
    
    # Add some debugging info
    if os.environ.get('DATABASE_URL'):
        db_url = os.environ.get('DATABASE_URL')
        # Don't print the full URL for security, just the start
        print(f"📊 Database: {db_url[:20]}...")
    else:
        print("⚠️  No DATABASE_URL found - using local database config")
    
    # Run database migrations first
    print("📊 Running database migrations...")
    try:
        execute_from_command_line(['manage.py', 'migrate', '--noinput'])
        print("✅ Database migrations completed")
    except Exception as e:
        print(f"⚠️ Migration warning: {e}")
    
    # Collect static files (if needed)
    try:
        execute_from_command_line(['manage.py', 'collectstatic', '--noinput'])
        print("✅ Static files collected")
    except Exception as e:
        print(f"⚠️ Static files warning: {e}")
    
    # Start the server
    if os.environ.get('RAILWAY_DEPLOYMENT') == 'True':
        # Production: Use gunicorn
        import subprocess
        print("🔥 Starting Gunicorn server...")
        subprocess.run([
            'gunicorn', 
            '--bind', f'0.0.0.0:{port}',
            '--workers', '2',
            '--timeout', '120',
            'backend.wsgi:application'
        ])
    else:
        # Development: Use Django dev server
        execute_from_command_line(['manage.py', 'runserver', f'0.0.0.0:{port}'])

if __name__ == '__main__':
    main()
