#!/usr/bin/env python3
"""
Minimal Railway startup script - Django only
"""
import os
import sys
import django
from django.core.management import execute_from_command_line
from django.core.wsgi import get_wsgi_application

def setup_django():
    """Initialize Django settings"""
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.settings')
    django.setup()

def run_migrations():
    """Apply database migrations"""
    print("Applying database migrations...")
    try:
        execute_from_command_line(['manage.py', 'migrate'])
        print("✅ Migrations completed successfully")
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        raise

def start_server():
    """Start the Gunicorn server"""
    print("Starting Gunicorn server...")
    
    # Get port from environment (Railway sets this)
    port = os.environ.get('PORT', '8000')
    
    # Import the WSGI application
    application = get_wsgi_application()
    
    # Start Gunicorn programmatically
    from gunicorn.app.wsgiapp import WSGIApplication
    from gunicorn.config import Config
    
    # Gunicorn configuration
    options = {
        'bind': f'0.0.0.0:{port}',
        'workers': 2,
        'worker_class': 'sync',
        'timeout': 120,
        'keepalive': 2,
        'max_requests': 1000,
        'max_requests_jitter': 100,
        'access_logfile': '-',
        'error_logfile': '-',
        'capture_output': True,
        'enable_stdio_inheritance': True,
    }
    
    class StandaloneApplication(WSGIApplication):
        def init(self, parser, opts, args):
            return self.cfg
            
        def load_config(self):
            for key, value in options.items():
                self.cfg.set(key.lower(), value)
                
        def load_wsgiapp(self):
            return application
    
    app = StandaloneApplication()
    app.run()

def main():
    """Main entry point"""
    print("🚀 Starting Railway deployment...")
    
    # Add current directory to Python path
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    try:
        # Setup Django
        setup_django()
        
        # Run migrations
        run_migrations()
        
        # Start the server
        start_server()
        
    except Exception as e:
        print(f"❌ Startup failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()