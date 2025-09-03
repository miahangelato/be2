#!/usr/bin/env python
"""
Railway startup script with better error handling
"""
import os
import sys
import subprocess

def run_command(command, description):
    """Run a command and handle errors gracefully"""
    print(f"=== {description} ===")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        print(f"Command: {command}")
        print(f"Return code: {result.returncode}")
        
        if result.stdout:
            print(f"STDOUT:\n{result.stdout}")
        if result.stderr:
            print(f"STDERR:\n{result.stderr}")
            
        if result.returncode != 0:
            print(f"⚠️  {description} failed but continuing...")
            return False
        else:
            print(f"✅ {description} completed successfully")
            return True
            
    except Exception as e:
        print(f"❌ Error running {description}: {e}")
        return False

def main():
    print("🚀 Starting Railway deployment...")
    
    # Set environment
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.settings')
    
    # 1. Download models (optional - continue if fails)
    run_command("python download_models.py", "Model Download")
    
    # 2. Run migrations (required)
    if not run_command("python manage.py migrate", "Database Migration"):
        print("❌ Database migration failed - this is critical!")
        sys.exit(1)
    
    # 3. Collect static files (if needed)
    run_command("python manage.py collectstatic --noinput", "Static Files Collection")
    
    # 4. Start server
    port = os.getenv('PORT', '8000')
    server_command = f"gunicorn backend.wsgi:application --bind 0.0.0.0:{port} --timeout 120 --workers 1 --log-level info"
    
    print(f"🌐 Starting server on port {port}...")
    print(f"Command: {server_command}")
    
    # Don't capture output for server - let it run normally
    os.system(server_command)

if __name__ == "__main__":
    main()
