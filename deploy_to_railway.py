#!/usr/bin/env python3
"""
Railway deployment helper script
Run this to deploy the optimized version to Railway
"""
import subprocess
import sys
import os

def run_command(cmd, description):
    """Run a command and return success status"""
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        if result.stdout:
            print(f"Output: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed")
        print(f"Error: {e.stderr}")
        return False

def main():
    """Main deployment function"""
    print("🚀 Starting Railway deployment process...")
    
    # Check if we're in the right directory
    if not os.path.exists('manage.py'):
        print("❌ Please run this script from the backend directory (where manage.py is located)")
        sys.exit(1)
    
    # Check if Railway CLI is installed
    if not run_command("railway --version", "Checking Railway CLI"):
        print("💡 Please install Railway CLI: npm install -g @railway/cli")
        sys.exit(1)
    
    # Login to Railway (if not already logged in)
    print("\n🔐 Checking Railway authentication...")
    result = subprocess.run("railway whoami", shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print("Please log in to Railway:")
        if not run_command("railway login", "Railway login"):
            sys.exit(1)
    
    # Deploy to Railway
    if not run_command("railway up", "Deploying to Railway"):
        print("\n❌ Deployment failed. Common issues:")
        print("1. Check that all environment variables are set in Railway dashboard")
        print("2. Ensure DATABASE_URL is configured correctly")
        print("3. Verify AWS credentials (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)")
        print("4. Check Railway logs: railway logs")
        sys.exit(1)
    
    print("\n🎉 Deployment initiated successfully!")
    print("\n📋 Next steps:")
    print("1. Check deployment status: railway status")
    print("2. View logs: railway logs")
    print("3. Open app: railway open")
    print("4. Test health endpoint: <your-app-url>/api/ping/")
    print("5. Warm up models: <your-app-url>/api/health/")

if __name__ == "__main__":
    main()