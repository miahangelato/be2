import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import logging
import requests
import tempfile

logger = logging.getLogger(__name__)

class BloodGroupClassifier:
    def __init__(self):
        self.model = None
        self.model_path = os.path.join(os.path.dirname(__file__), 'bloodgroup_model_20250823-140933.h5')
        self.s3_url = "https://team3thesis.s3.us-east-1.amazonaws.com/models/backend/core%5Cbloodgroup_model_20250823-140933.h5"
        self.load_model()
        
        # Blood group classes based on the Kaggle dataset
        self.blood_groups = ['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-']
    
    def load_model_from_s3_url(self, url, cache_path=None):
        """
        Load Keras model directly from S3 URL using requests with local caching
        """
        try:
            logger.info(f"Downloading blood group model from: {url}")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
                total_size = int(response.headers.get('content-length', 0))
                downloaded = 0
                
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        tmp.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0 and downloaded % (1024 * 1024) == 0:  # Log every MB
                            percent = (downloaded / total_size) * 100
                            logger.info(f"Download progress: {percent:.1f}%")
                
                tmp.flush()
                logger.info(f"Blood group model downloaded successfully ({downloaded} bytes)")
                
                # Load the model from temporary file
                model = load_model(tmp.name)
                
                # Cache the model locally if cache_path is provided
                if cache_path:
                    try:
                        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                        import shutil
                        shutil.copy2(tmp.name, cache_path)
                        logger.info(f"Blood group model cached to: {cache_path}")
                    except Exception as e:
                        logger.warning(f"Failed to cache blood group model: {e}")
                
                # Clean up temporary file
                os.unlink(tmp.name)
                
                return model
                
        except Exception as e:
            raise Exception(f"Failed to download and load blood group model from S3: {e}")
    
    def load_model(self):
        """Load the blood group classification model with smart caching"""
        try:
            # First, try to load from local cache
            if os.path.exists(self.model_path):
                logger.info(f"Loading cached blood group model from: {self.model_path}")
                self.model = load_model(self.model_path)
                return
            
            # If no local cache, download from S3 and cache it
            logger.info("No local cache found, downloading blood group model from S3...")
            self.model = self.load_model_from_s3_url(self.s3_url, cache_path=self.model_path)
            logger.info("Blood group model loaded and cached successfully")
                        
        except Exception as e:
            logger.error(f"Error loading blood group model: {e}")
            raise
    
    def preprocess_fingerprint(self, image_path):
        """
        Preprocess fingerprint image for blood group classification
        For model input shape (128, 128, 1) - grayscale
        """
        try:
            # Read the image as grayscale
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            print(f"[DEBUG] Reading image: {image_path}")
            print(f"[DEBUG] After cv2.imread: shape={None if img is None else img.shape}, dtype={None if img is None else img.dtype}")
            if img is None:
                raise ValueError(f"Could not read image from {image_path}")

            # Resize to (128, 128)
            img = cv2.resize(img, (128, 128))
            print(f"[DEBUG] After cv2.resize: shape={img.shape}, dtype={img.dtype}")

            # Normalize pixel values
            img = img.astype(np.float32) / 255.0
            print(f"[DEBUG] After normalization: shape={img.shape}, dtype={img.dtype}, min={img.min()}, max={img.max()}")

            # Ensure the image has 3 channels (convert grayscale to RGB)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            print(f"[DEBUG] After cv2.cvtColor: shape={img.shape}, dtype={img.dtype}")

            # Expand dims to (1, 128, 128, 3)
            img = np.expand_dims(img, axis=0)
            print(f"[DEBUG] After np.expand_dims: shape={img.shape}, dtype={img.dtype}")

            return img

        except Exception as e:
            logger.error(f"Error preprocessing image {image_path}: {e}")
            raise
    
    def predict_blood_group(self, fingerprint_image_path):
        """
        Predict blood group from fingerprint image
        
        Args:
            fingerprint_image_path (str): Path to the fingerprint image
            
        Returns:
            dict: {
                'predicted_blood_group': str,
                'confidence': float,
                'all_probabilities': dict
            }
        """
        try:
            if self.model is None:
                raise ValueError("Model not loaded")
            
            # Preprocess the image
            processed_image = self.preprocess_fingerprint(fingerprint_image_path)
            print(f"[DEBUG] Model input shape: {processed_image.shape}, dtype={processed_image.dtype}")
            
            # Make prediction
            predictions = self.model.predict(processed_image, verbose=0)
            print(f"[DEBUG] Model output shape: {predictions.shape}, dtype={predictions.dtype}")
            
            # Get the predicted class and confidence
            predicted_class_index = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class_index])
            predicted_blood_group = self.blood_groups[predicted_class_index]
            
            # Create probability dictionary for all classes
            all_probabilities = {}
            for i, blood_group in enumerate(self.blood_groups):
                all_probabilities[blood_group] = float(predictions[0][i])
            
            result = {
                'predicted_blood_group': predicted_blood_group,
                'confidence': confidence,
                'all_probabilities': all_probabilities
            }
            
            logger.info(f"Blood group prediction: {predicted_blood_group} (confidence: {confidence:.2f})")
            return result
            
        except Exception as e:
            logger.error(f"Error predicting blood group: {e}")
            raise
    
    def predict_from_multiple_fingerprints(self, fingerprint_paths):
        """
        Predict blood group from multiple fingerprint images and return the most confident prediction
        
        Args:
            fingerprint_paths (list): List of paths to fingerprint images
            
        Returns:
            dict: Best prediction result
        """
        try:
            if not fingerprint_paths:
                raise ValueError("No fingerprint paths provided")
            
            all_predictions = []
            
            for path in fingerprint_paths:
                if os.path.exists(path):
                    prediction = self.predict_blood_group(path)
                    all_predictions.append(prediction)
            
            if not all_predictions:
                raise ValueError("No valid fingerprint images found")
            
            # Find the prediction with highest confidence
            best_prediction = max(all_predictions, key=lambda x: x['confidence'])
            
            # Add ensemble information
            best_prediction['ensemble_size'] = len(all_predictions)
            best_prediction['all_predictions'] = all_predictions
            
            return best_prediction
            
        except Exception as e:
            logger.error(f"Error in ensemble prediction: {e}")
            raise
        
    def parse_and_find_highest_confidence(self, predictions):
        """
        Parse predictions from all 10 fingers and find the blood group with the highest confidence.

        Args:
            predictions (list): List of prediction dictionaries for each finger.

        Returns:
            str: Predicted blood group with the highest confidence.
        """
        if not predictions:
            raise ValueError("No predictions provided")

        # Find the prediction with the highest confidence
        highest_confidence_entry = max(predictions, key=lambda x: x['confidence'])
        return highest_confidence_entry['predicted_blood_group']

# Create a global instance
blood_group_classifier = BloodGroupClassifier()

def classify_blood_group(fingerprint_image_path):
    """
    Convenience function to classify blood group from a single fingerprint
    """
    return blood_group_classifier.predict_blood_group(fingerprint_image_path)

def classify_blood_group_from_multiple(fingerprint_paths):
    """
    Convenience function to classify blood group from multiple fingerprints
    """
    return blood_group_classifier.predict_from_multiple_fingerprints(fingerprint_paths)
