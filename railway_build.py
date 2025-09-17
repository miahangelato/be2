#!/usr/bin/env python3
"""
Railway-specific build optimization script
Handles memory-efficient model downloads and build preparation
"""
import os
import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def optimize_build():
    """Optimize the build process for Railway deployment"""
    try:
        logger.info("🚀 Starting Railway build optimization...")
        
        # Set memory-friendly environment variables
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging
        os.environ['PYTHONUNBUFFERED'] = '1'      # Ensure output is not buffered
        os.environ['DJANGO_SETTINGS_MODULE'] = 'backend.settings'
        
        # Verify critical environment variables
        required_vars = ['DATABASE_URL']
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        
        if missing_vars:
            logger.warning(f"⚠️  Missing environment variables: {missing_vars}")
        else:
            logger.info("✅ All critical environment variables present")
        
        # Create necessary directories
        os.makedirs('staticfiles', exist_ok=True)
        os.makedirs('media', exist_ok=True)
        
        logger.info("📁 Created necessary directories")
        
        # Pre-download models if credentials available
        if all([os.getenv('AWS_ACCESS_KEY_ID'), os.getenv('AWS_SECRET_ACCESS_KEY')]):
            logger.info("🔑 AWS credentials found - models will be downloaded at runtime")
        else:
            logger.warning("⚠️  AWS credentials not found - using fallback models")
        
        logger.info("✅ Railway build optimization completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Build optimization failed: {str(e)}")
        return False

if __name__ == "__main__":
    success = optimize_build()
    sys.exit(0 if success else 1)