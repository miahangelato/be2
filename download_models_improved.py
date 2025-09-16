#!/usr/bin/env python
"""
Improved model download utility for Railway deployment
- Adds parallel downloads, retry logic, and caching
"""
import os
import sys
import time
import requests
import threading
from pathlib import Path
import hashlib
import json
import concurrent.futures

# Define all models that need to be downloaded
MODELS = [
    {
        "name": "Fingerprint CNN Model",
        "local_path": "core/improved_pattern_cnn_model.h5",
        "s3_url": "https://team3thesis.s3.us-east-1.amazonaws.com/models/backend/core%5Cimproved_pattern_cnn_model.h5",
        "expected_size_mb": 45,  # Approximate size in MB
    },
    {
        "name": "Blood Group Model",
        "local_path": "core/bloodgroup_model_20250823-140933.h5",
        "s3_url": "https://team3thesis.s3.us-east-1.amazonaws.com/models/backend/core/bloodgroup_model_20250823-140933.h5",
        "expected_size_mb": 15,  # Approximate size in MB
    },
    {
        "name": "Diabetes Risk Model",
        "local_path": "core/diabetes_risk_model.pkl",
        "s3_url": "https://team3thesis.s3.us-east-1.amazonaws.com/models/backend/core/diabetes_risk_model.pkl",
        "expected_size_mb": 2,  # Approximate size in MB
    },
    {
        "name": "Diabetes Model Columns",
        "local_path": "core/diabetes_risk_model_columns.pkl",
        "s3_url": "https://team3thesis.s3.us-east-1.amazonaws.com/models/backend/core/diabetes_risk_model_columns.pkl",
        "expected_size_mb": 1,  # Approximate size in MB
    }
]

# Create a manifest file path to track downloaded models
BASE_DIR = Path(__file__).resolve().parent
MANIFEST_PATH = BASE_DIR / '.model_manifest.json'

def get_file_hash(filepath):
    """Calculate SHA-256 hash of a file"""
    sha256_hash = hashlib.sha256()
    try:
        with open(filepath, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    except:
        return None

def load_manifest():
    """Load the model manifest file or create it if it doesn't exist"""
    if os.path.exists(MANIFEST_PATH):
        try:
            with open(MANIFEST_PATH, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_manifest(manifest):
    """Save the model manifest file"""
    with open(MANIFEST_PATH, 'w') as f:
        json.dump(manifest, f, indent=2)

def download_model(model_info, max_retries=3):
    """Download a model with retry logic and progress reporting"""
    model_path = BASE_DIR / model_info["local_path"]
    
    # Create directories if they don't exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Check if we already have the file
    if os.path.exists(model_path):
        file_size = os.path.getsize(model_path) / (1024 * 1024)
        print(f"✅ Model {model_info['name']} already exists ({file_size:.2f} MB)")
        
        # Update manifest
        manifest = load_manifest()
        manifest[model_info["local_path"]] = {
            "hash": get_file_hash(model_path),
            "last_downloaded": time.strftime("%Y-%m-%d %H:%M:%S"),
            "size_mb": file_size
        }
        save_manifest(manifest)
        
        return True
    
    print(f"⬇️ Downloading {model_info['name']} (~{model_info['expected_size_mb']} MB)")
    
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(model_info["s3_url"], stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(model_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):  # 1MB chunks
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        percent = (downloaded / total_size) * 100 if total_size > 0 else 0
                        # Print progress every 5MB
                        if downloaded % (5 * 1024 * 1024) == 0:
                            print(f"  {model_info['name']}: {percent:.1f}% ({downloaded/(1024*1024):.1f} MB)")
            
            # Verify download
            file_size = os.path.getsize(model_path) / (1024 * 1024)
            file_hash = get_file_hash(model_path)
            
            print(f"✅ {model_info['name']} downloaded successfully ({file_size:.2f} MB)")
            
            # Update manifest
            manifest = load_manifest()
            manifest[model_info["local_path"]] = {
                "hash": file_hash,
                "last_downloaded": time.strftime("%Y-%m-%d %H:%M:%S"),
                "size_mb": file_size
            }
            save_manifest(manifest)
            
            return True
        
        except Exception as e:
            if attempt < max_retries:
                print(f"⚠️ Attempt {attempt} failed for {model_info['name']}: {e}")
                print(f"   Retrying in 5 seconds...")
                time.sleep(5)
            else:
                print(f"❌ Failed to download {model_info['name']} after {max_retries} attempts: {e}")
                return False

def download_all_models():
    """Download all models in parallel"""
    print("=== Downloading ML models ===")
    
    # Create a thread pool to download models in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        # Submit download tasks
        futures = {executor.submit(download_model, model): model["name"] for model in MODELS}
        
        # Process results as they complete
        results = {}
        for future in concurrent.futures.as_completed(futures):
            model_name = futures[future]
            try:
                success = future.result()
                results[model_name] = success
            except Exception as e:
                results[model_name] = False
                print(f"❌ Error downloading {model_name}: {e}")
    
    # Print summary
    print("\n=== Download Summary ===")
    all_success = True
    for model_name, success in results.items():
        status = "✅ Success" if success else "❌ Failed"
        print(f"{model_name}: {status}")
        if not success:
            all_success = False
    
    return all_success

if __name__ == "__main__":
    print(f"Working directory: {os.getcwd()}")
    success = download_all_models()
    sys.exit(0 if success else 1)