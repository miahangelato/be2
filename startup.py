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
        # Suppress TensorFlow warnings in production
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    else:
        # For local development, try to load .env file
        try:
            from dotenv import load_dotenv
            env_path = os.path.join(os.path.dirname(__file__), '.env')
            if os.path.exists(env_path):
                load_dotenv(env_path)
                print(f"✅ Loaded environment variables from .env file")
            else:
                print(f"⚠️ No .env file found at {env_path}")
        except ImportError:
            print("⚠️ python-dotenv not installed, skipping .env loading")
            pass
    
    # Detect DB type for better error messages
    db_url = os.environ.get('DATABASE_URL', '')
    if db_url.startswith('postgres'):
        print("� Using PostgreSQL database")
        
        # Check for psycopg2 installation
        try:
            import psycopg2
            print("✅ psycopg2 is installed")
        except ImportError:
            print("❌ ERROR: psycopg2 is not installed! Install with: pip install psycopg2-binary")
            
        # Parse PostgreSQL connection details for debugging
        from urllib.parse import urlparse
        try:
            parsed = urlparse(db_url)
            host = parsed.hostname or os.environ.get('DB_HOST', 'unknown')
            port = parsed.port or os.environ.get('DB_PORT', '5432')
            dbname = parsed.path[1:] if parsed.path else os.environ.get('DB_NAME', 'unknown')
            
            print(f"📊 PostgreSQL connection details:")
            print(f"   - Host: {host}")
            print(f"   - Port: {port}")
            print(f"   - Database: {dbname}")
            print(f"   - SSL Mode: {'Required' if not bool(os.environ.get('RAILWAY_ENVIRONMENT')) else 'Not required for Railway'}")
        except Exception as e:
            print(f"⚠️ Could not parse DATABASE_URL: {e}")
    
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
    
    # Check database connection
    try:
        from django.db import connection
        connection.ensure_connection()
        
        # Get PostgreSQL version if connected
        if connection.vendor == 'postgresql':
            with connection.cursor() as cursor:
                cursor.execute("SELECT version();")
                version = cursor.fetchone()[0]
                print(f"✅ PostgreSQL connection successful: {version.split(',')[0]}")
                
                # Test a simple query to verify full functionality
                cursor.execute("SELECT COUNT(*) FROM pg_catalog.pg_tables;")
                table_count = cursor.fetchone()[0]
                print(f"✅ Database contains {table_count} tables")
        else:
            print(f"✅ Database connection successful: {connection.vendor}")
            
    except Exception as db_error:
        print(f"⚠️ WARNING: Database connection test failed: {db_error}")
        print("🔍 Common PostgreSQL issues:")
        print("   - Check that PostgreSQL service is running")
        print("   - Verify database name, username and password are correct")
        print("   - Confirm host and port are accessible from this environment")
        print("   - On Railway: check that the DATABASE_URL is correctly set")
        print("🔄 Will still attempt to start server...")
    
    # Start the server
    if os.environ.get('RAILWAY_DEPLOYMENT') == 'True':
        # Production: Use gunicorn
        import subprocess
        print("🔥 Starting Gunicorn server...")
        try:
            process = subprocess.run([
                'gunicorn', 
                '--bind', f'0.0.0.0:{port}',
                '--workers', '2',
                '--timeout', '120',
                '--log-level', 'debug',  # Increased log level for troubleshooting
                'backend.wsgi:application'
            ], check=True)
            if process.returncode != 0:
                print(f"⚠️ WARNING: Gunicorn exited with code {process.returncode}")
        except Exception as e:
            print(f"❌ ERROR starting Gunicorn: {e}")
    else:
        # Development: Use Django dev server
        execute_from_command_line(['manage.py', 'runserver', f'0.0.0.0:{port}'])

if __name__ == '__main__':
    main()
