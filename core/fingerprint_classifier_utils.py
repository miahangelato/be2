# ml/fingerprint_classifier.py
import os
from django.conf import settings
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

import requests
import tempfile

MODEL_PATH = os.path.join(settings.BASE_DIR, "core", "improved_pattern_cnn_model.h5")
MODEL_S3_URL = "https://team3thesis.s3.us-east-1.amazonaws.com/models/backend/core/improved_pattern_cnn_model.h5"
CLASS_NAMES = ["Arc", "Whorl", "Loop"]

def load_model_from_s3_url(url, cache_path=None):
    """
    Load Keras model directly from S3 URL using requests with local caching
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    tmp.write(chunk)
                    downloaded += len(chunk)
            
            tmp.flush()
            
            # Load the model from temporary file
            model = tf.keras.models.load_model(tmp.name)
            
            # Cache the model locally if cache_path is provided
            if cache_path:
                try:
                    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                    import shutil
                    shutil.copy2(tmp.name, cache_path)
                except Exception as e:
                    pass
            
            # Clean up temporary file
            os.unlink(tmp.name)
            
            return model
            
    except Exception as e:
        raise Exception(f"Failed to download and load model from S3: {e}")

def load_model_with_fallback(local_path, s3_url):
    """
    Load model with smart caching: local file > S3 download > cache locally
    """
    # First, try to load from local cache
    if os.path.exists(local_path):
        return tf.keras.models.load_model(local_path)
    
    # If no local cache, download from S3 and cache it
    try:
        return load_model_from_s3_url(s3_url, cache_path=local_path)
    except Exception as s3_error:
        error_msg = f"""
Failed to load model from S3:
- S3 URL: {s3_url} 
- Error: {s3_error}
- Local cache path: {local_path}

The model will be downloaded once and cached locally for future use.
        """.strip()
        raise Exception(error_msg)

# Global variable to store the loaded model
_model = None

def get_model():
    """Lazy load the model only when needed"""
    global _model
    if _model is None:
        _model = load_model_with_fallback(MODEL_PATH, MODEL_S3_URL)
    return _model

def classify_fingerprint_pattern(img_file):
    model = get_model()  # Lazy load the model
    img = image.load_img(img_file, color_mode="grayscale", target_size=(128, 128))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0) / 255.0
    preds = model.predict(x)
    return CLASS_NAMES[np.argmax(preds)]
