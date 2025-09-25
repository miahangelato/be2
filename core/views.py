from ninja import NinjaAPI, File, Query, Form, UploadedFile, Schema
from .models import Participant, Fingerprint, Result
from .ml_availability import ml_available, get_ml_status
import base64
import os
from typing import Dict, Any
import json as pyjson
import json
from django.http import JsonResponse
from django.conf import settings
from .encryption_utils import encryption_service
from .backend_decryption import backend_decryption

# Import ML modules conditionally
try:
    from .fingerprint_classifier_utils import classify_fingerprint_pattern
    ML_FINGERPRINT_AVAILABLE = True
except ImportError:
    ML_FINGERPRINT_AVAILABLE = False

try:
    from .bloodgroup_classifier import classify_blood_group_from_multiple
    ML_BLOODGROUP_AVAILABLE = True
except ImportError:
    ML_BLOODGROUP_AVAILABLE = False

try:
    from .diabetes_predictor import DiabetesPredictor
    ML_DIABETES_AVAILABLE = True
except ImportError:
    ML_DIABETES_AVAILABLE = False

api = NinjaAPI()

@api.get("/ping/")
def ping(request):
    """Simple ping endpoint for basic health check"""
    return {"status": "ok", "message": "Django app is running"}

@api.get("/test-ml/")
def test_ml_models(request):
    """Test endpoint to verify ML models are working on Railway"""
    try:
        results = {
            "ml_availability": {
                "fingerprint": ML_FINGERPRINT_AVAILABLE,
                "bloodgroup": ML_BLOODGROUP_AVAILABLE,
                "diabetes": ML_DIABETES_AVAILABLE,
            },
            "model_tests": {}
        }
        
        # Test blood group model loading
        if ML_BLOODGROUP_AVAILABLE:
            try:
                from .bloodgroup_classifier import BloodGroupClassifier
                classifier = BloodGroupClassifier()
                classifier.load_model()
                if classifier.model is not None:
                    results["model_tests"]["bloodgroup"] = {
                        "status": "loaded",
                        "input_shape": str(classifier.model.input_shape),
                        "output_shape": str(classifier.model.output_shape)
                    }
                else:
                    results["model_tests"]["bloodgroup"] = {"status": "failed_to_load"}
            except Exception as e:
                results["model_tests"]["bloodgroup"] = {"status": "error", "error": str(e)}
        
        # Test diabetes model
        if ML_DIABETES_AVAILABLE:
            try:
                from .diabetes_predictor import DiabetesPredictor
                predictor = DiabetesPredictor()
                results["model_tests"]["diabetes"] = {"status": "loaded"}
            except Exception as e:
                results["model_tests"]["diabetes"] = {"status": "error", "error": str(e)}
        
        return results
        
    except Exception as e:
        return {"error": str(e)}

@api.get("/health/")
def health_check(request):
    """
    Health check endpoint that checks system status
    Use this to check deployment status and ML availability
    """
    try:
        # Get ML package status
        ml_status = get_ml_status()
        
        # Check database connectivity
        from django.db import connection
        with connection.cursor() as cursor:
            cursor.execute("SELECT 1")
            db_status = "connected"
    except Exception as e:
        db_status = f"error: {str(e)}"
    
    # Check S3 connectivity
    try:
        import boto3
        from django.conf import settings
        s3_client = boto3.client('s3')
        s3_client.head_bucket(Bucket=settings.AWS_STORAGE_BUCKET_NAME)
        s3_status = "connected"
    except Exception as e:
        s3_status = f"error: {str(e)}"
    
    return {
        "status": "healthy" if ml_status['all_available'] else "partial",
        "database": db_status,
        "s3_storage": s3_status,
        "ml_packages": ml_status,
        "features": {
            "fingerprint_analysis": ML_FINGERPRINT_AVAILABLE,
            "bloodgroup_analysis": ML_BLOODGROUP_AVAILABLE,
            "diabetes_prediction": ML_DIABETES_AVAILABLE,
        },
        "message": "All systems operational" if ml_status['all_available'] else "ML features disabled - minimal deployment mode"
    }

@api.post("/identify-blood-group-from-participant/")
def identify_blood_group_from_participant(request, participant_id: int = Query(...)):
    """
    Identify blood group for each fingerprint image of a participant (by participant_id).
    Returns a list of predictions, one per fingerprint.
    """
    try:
        # Check if ML is available
        if not ML_BLOODGROUP_AVAILABLE:
            return {
                "success": False,
                "participant_id": participant_id,
                "predicted_blood_group": "ML_UNAVAILABLE",
                "confidence": 0.0,
                "results": [],
                "error": "Machine learning models are not available"
            }
        
        # Check if participant exists
        try:
            participant = Participant.objects.get(id=participant_id)
        except Participant.DoesNotExist:
            return {
                "success": False,
                "error": "Participant not found.", 
                "participant_id": participant_id,
                "predicted_blood_group": "ERROR",
                "confidence": 0.0,
                "results": []
            }

        # Fetch fingerprints
        fingerprints = participant.fingerprints.all()
        if not fingerprints:
            return {
                "success": False,
                "error": "No fingerprints found.", 
                "participant_id": participant_id,
                "predicted_blood_group": "NO_FINGERPRINTS",
                "confidence": 0.0,
                "results": []
            }

        results = []
        predicted_blood_group = None
        best_confidence = 0.0
        
        for fp in fingerprints:
            if fp.image:
                try:
                    # Handle both local files and S3 storage
                    try:
                        # Try to get local path first (for local development)
                        image_path = fp.image.path
                        if not os.path.exists(image_path):
                            raise Exception("Local file not found")
                        pred = classify_blood_group_from_multiple([image_path])
                    except (NotImplementedError, Exception):
                        # S3 storage or file not found locally - download image first
                        import tempfile
                        import requests
                        from django.core.files.storage import default_storage
                        
                        # Create temporary file
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_file:
                            # Read image from S3 storage
                            with default_storage.open(fp.image.name, 'rb') as image_file:
                                temp_file.write(image_file.read())
                            temp_file.flush()
                            
                            # Now classify using temporary file
                            pred = classify_blood_group_from_multiple([temp_file.name])
                            
                            # Clean up temporary file
                            os.unlink(temp_file.name)
                    
                    current_prediction = pred['predicted_blood_group']
                    current_confidence = pred['confidence']
                    
                    # Update best prediction if this one has higher confidence
                    if predicted_blood_group is None or current_confidence > best_confidence:
                        predicted_blood_group = current_prediction
                        best_confidence = current_confidence
                    
                    results.append({
                        "finger": fp.finger,
                        "filename": fp.image.name.split('/')[-1] if '/' in fp.image.name else fp.image.name,
                        "predicted_blood_group": current_prediction,
                        "confidence": current_confidence,
                        "all_probabilities": pred.get('all_probabilities'),
                    })
                except Exception as e:
                    import traceback
                    error_msg = f"Blood group prediction error for {fp.finger}: {str(e)}\n{traceback.format_exc()}"
                    print(error_msg)  # This will show in Railway logs
                    results.append({"finger": fp.finger, "error": str(e)})
            else:
                results.append({"finger": fp.finger, "error": "No image uploaded"})

        # If no successful predictions, set fallback
        if predicted_blood_group is None:
            predicted_blood_group = "UNKNOWN"
            best_confidence = 0.0

        return {
            "success": True,
            "participant_id": participant_id, 
            "results": results, 
            "predicted_blood_group": predicted_blood_group,
            "confidence": best_confidence
        }
        
    except Exception as e:
        import traceback
        error_msg = f"Critical error in blood group identification: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)  # This will show in Railway logs
        raise Exception(error_msg)
    
@api.post("/identify-blood-group-from-json/")
def identify_blood_group_from_json(request, json: str = Form(...), files: list[UploadedFile] = File(...)):
    """
    Identify blood group for each uploaded fingerprint image, using metadata from JSON (consent=false flow).
    Expects:
      - json: JSON string with 'fingerprints' (list of dicts with 'finger', 'image_name', ...)
      - files: uploaded fingerprint images (order or name must match JSON)
    Returns a list of predictions, one per image.
    """
    import tempfile, shutil
    results = []
    temp_paths = []
    predicted_blood_group = None  # Initialize outside the loop
    
    try:
        if not ML_BLOODGROUP_AVAILABLE:
            return {
                "success": False,
                "predicted_blood_group": "ML_UNAVAILABLE",
                "confidence": 0.0,
                "results": [],
                "error": "Machine learning models are not available"
            }
            
        data = pyjson.loads(json)
        fingerprints_meta = data.get('fingerprints', [])
        
        # Map image_name to file
        file_map = {f.name: f for f in files}
        
        for fp_meta in fingerprints_meta:
            img_name = fp_meta.get('image_name')
            f = file_map.get(img_name)
            if not f:
                results.append({"image_name": img_name, "error": "No file uploaded for this fingerprint"})
                continue
                
            temp_fd, temp_path = tempfile.mkstemp(suffix='.png')
            os.close(temp_fd)
            
            with open(temp_path, 'wb') as out:
                f.file.seek(0)
                shutil.copyfileobj(f.file, out)
            temp_paths.append(temp_path)
            
            try:
                pred = classify_blood_group_from_multiple([temp_path])
                current_prediction = pred['predicted_blood_group']
                
                # Update the best prediction if this one has higher confidence
                if predicted_blood_group is None or pred['confidence'] > results[-1]['confidence'] if results else 0:
                    predicted_blood_group = current_prediction
                
                results.append({
                    "finger": fp_meta.get('finger'),
                    "image_name": img_name,
                    "predicted_blood_group": current_prediction,
                    "confidence": pred['confidence'],
                    "all_probabilities": pred.get('all_probabilities'),
                })
            except Exception as e:
                import traceback
                error_msg = f"Error processing {img_name}: {str(e)}\n{traceback.format_exc()}"
                print(error_msg)  # This will show in Railway logs
                results.append({
                    "finger": fp_meta.get('finger'),
                    "image_name": img_name,
                    "error": str(e)
                })
        
        # If no successful predictions, set a fallback
        if predicted_blood_group is None:
            predicted_blood_group = "UNKNOWN"
            
        return {
            "success": True, 
            "results": results, 
            "predicted_blood_group": predicted_blood_group,
            "confidence": max([r.get('confidence', 0) for r in results if 'confidence' in r], default=0)
        }
        
    except Exception as e:
        import traceback
        error_msg = f"Blood group identification failed: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)  # This will show in Railway logs
        return {
            "success": False, 
            "error": error_msg,
            "predicted_blood_group": "ERROR",
            "confidence": 0.0,
            "results": []
        }
    finally:
        for p in temp_paths:
            try:
                if os.path.exists(p):
                    os.unlink(p)
            except Exception:
                pass

@api.post("/consent/")
def submit_consent(request, consent: bool = Form(...)):
    return {"consent": consent}

@api.post("/submit/")
def submit(
    request,
    consent: bool = Form(...),
    age: str = Form(...),  # Changed to string to handle encrypted data
    height: str = Form(...),  # Changed to string to handle encrypted data
    weight: str = Form(...),  # Changed to string to handle encrypted data
    gender: str = Form(...),
    blood_type: str = Form(None),
    willing_to_donate: bool = Form(...),
    sleep_hours: str = Form(None),  # Changed to string to handle encrypted data
    had_alcohol_last_24h: str = Form(None),  # Changed to string to handle encrypted data
    ate_before_donation: str = Form(None),  # Changed to string to handle encrypted data
    ate_fatty_food: str = Form(None),  # Changed to string to handle encrypted data
    recent_tattoo_or_piercing: str = Form(None),  # Changed to string to handle encrypted data
    has_chronic_condition: str = Form(None),  # Changed to string to handle encrypted data
    condition_controlled: str = Form(None),  # Changed to string to handle encrypted data
    last_donation_date: str = Form(None),
    left_thumb: UploadedFile = File(...),
    left_index: UploadedFile = File(...),
    left_middle: UploadedFile = File(...),
    left_ring: UploadedFile = File(...),
    left_pinky: UploadedFile = File(...),
    right_thumb: UploadedFile = File(...),
    right_index: UploadedFile = File(...),
    right_middle: UploadedFile = File(...),
    right_ring: UploadedFile = File(...),
    right_pinky: UploadedFile = File(...),
):
    received_data = {
        "consent": consent,
        "age": age,
        "height": height,
        "weight": weight,
        "gender": gender,
        "blood_type": blood_type,
        "willing_to_donate": willing_to_donate,
        "sleep_hours": sleep_hours,
        "had_alcohol_last_24h": had_alcohol_last_24h,
        "ate_before_donation": ate_before_donation,
        "ate_fatty_food": ate_fatty_food,
        "recent_tattoo_or_piercing": recent_tattoo_or_piercing,
        "has_chronic_condition": has_chronic_condition,
        "condition_controlled": condition_controlled,
        "last_donation_date": last_donation_date,
    }
    
    # Decrypt the received form data
    decrypted_data = backend_decryption.decrypt_form_data(received_data)
    
    # Helper function to safely convert values
    def safe_convert(value, target_type, fallback=None):
        """Safely convert a value to the target type, return fallback if conversion fails"""
        try:
            if value is None or value == "":
                return fallback
            if target_type == int:
                return int(float(str(value)))  # Handle "25.0" -> 25
            elif target_type == float:
                return float(str(value))
            elif target_type == str:
                return str(value)
            else:
                return value
        except (ValueError, TypeError) as e:
            return fallback
    
    # Use decrypted values for processing with safe conversion
    age = safe_convert(decrypted_data.get('age'), int, age)
    height = safe_convert(decrypted_data.get('height'), float, height)
    weight = safe_convert(decrypted_data.get('weight'), float, weight)
    gender = decrypted_data.get('gender', gender)
    blood_type = decrypted_data.get('blood_type', blood_type)
    
    # Convert string boolean values back to boolean
    def str_to_bool(value):
        if isinstance(value, str):
            return value.lower() in ('true', '1', 'yes')
        return value
    
    sleep_hours = safe_convert(decrypted_data.get('sleep_hours'), int, sleep_hours)
    had_alcohol_last_24h = str_to_bool(decrypted_data.get('had_alcohol_last_24h', had_alcohol_last_24h))
    ate_before_donation = str_to_bool(decrypted_data.get('ate_before_donation', ate_before_donation))
    ate_fatty_food = str_to_bool(decrypted_data.get('ate_fatty_food', ate_fatty_food))
    recent_tattoo_or_piercing = str_to_bool(decrypted_data.get('recent_tattoo_or_piercing', recent_tattoo_or_piercing))
    has_chronic_condition = str_to_bool(decrypted_data.get('has_chronic_condition', has_chronic_condition))
    condition_controlled = str_to_bool(decrypted_data.get('condition_controlled', condition_controlled))
    last_donation_date = decrypted_data.get('last_donation_date', last_donation_date)
    
    # Process fingerprints
    finger_files = {
        "left_thumb": left_thumb,
        "left_index": left_index,
        "left_middle": left_middle,
        "left_ring": left_ring,
        "left_pinky": left_pinky,
        "right_thumb": right_thumb,
        "right_index": right_index,
        "right_middle": right_middle,
        "right_ring": right_ring,
        "right_pinky": right_pinky,
    }

    fingerprints = []
    for finger_name, img_file in finger_files.items():
        if img_file and hasattr(img_file, 'file'):
            if not ML_FINGERPRINT_AVAILABLE:
                pattern = "ML_UNAVAILABLE"  # Placeholder when ML packages not installed
            else:
                pattern = classify_fingerprint_pattern(img_file.file)
            fingerprints.append({
                "finger": finger_name,
                "pattern": pattern,
                "image_name": img_file.name,
            })

    # Save or return data based on consent
    if consent:
        # Save participant and fingerprints to database
        participant = Participant.objects.create(
            age=age,
            height=height,
            weight=weight,
            gender=gender,
            blood_type=blood_type,
            willing_to_donate=willing_to_donate,
            sleep_hours=sleep_hours,
            had_alcohol_last_24h=had_alcohol_last_24h,
            ate_before_donation=ate_before_donation,
            ate_fatty_food=ate_fatty_food,
            recent_tattoo_or_piercing=recent_tattoo_or_piercing,
            has_chronic_condition=has_chronic_condition,
            condition_controlled=condition_controlled,
            last_donation_date=last_donation_date,
        )
        for fp in fingerprints:
            Fingerprint.objects.create(
                participant=participant,
                finger=fp["finger"],
                image=finger_files[fp["finger"]],
                pattern=fp["pattern"],
            )

        return {
            "saved": True,
            "participant_id": participant.id,
            "message": "Data saved successfully."
        }
    else:
        # Don't save, just return basic info
        return {
            "saved": False,
            "message": "Data not saved due to consent=false.",
            "participant_data": {
                "age": age,
                "height": height,
                "weight": weight,
                "gender": gender,
                "willing_to_donate": willing_to_donate,
                "blood_type": blood_type,
                "sleep_hours": sleep_hours,
                "had_alcohol_last_24h": had_alcohol_last_24h,
                "ate_before_donation": ate_before_donation,
                "ate_fatty_food": ate_fatty_food,
                "recent_tattoo_or_piercing": recent_tattoo_or_piercing,
                "has_chronic_condition": has_chronic_condition,
                "condition_controlled": condition_controlled,
                "last_donation_date": last_donation_date,
            },
            "fingerprints": fingerprints,
        }

# @api.post("/scan-finger/")
# def scan_finger(request, finger_name: str = Query(...)):
#     scanner = None
#     try:
#         status = dpfpdd.dpfpdd_init()
#         if status != DPFPDD_SUCCESS:
#             return {
#                 "success": False,
#                 "error": "Failed to initialize fingerprint scanner. Please try again.",
#                 "debug_info": f"Status = 0x{status:x}"
#             }
#         scanner = FingerprintScanner()
#         image_data = scanner.capture_fingerprint()
#         if not image_data:
#             return {
#                 "success": False,
#                 "error": "Failed to capture fingerprint. Please ensure your finger is properly placed on the scanner.",
#                 "debug_info": "No image data returned"
#             }
#         base64_image = base64.b64encode(image_data).decode('utf-8')
#         return {
#             "success": True,
#             "image": base64_image,
#             "finger": finger_name
#         }
#     except Exception as e:
#         return {
#             "success": False,
#             "error": "An error occurred while scanning. Please try again.",
#             "debug_info": str(e)
#         }
#     finally:
#         if scanner:
#             try:
#                 scanner.close()
#             except Exception:
#                 pass
#         try:
#             dpfpdd.dpfpdd_exit()
#         except Exception:
#             pass

def analyze_blood_donation_eligibility(participant_data):
    """
    Analyze blood donation eligibility based on Philippine Red Cross criteria
    
    Args:
        participant_data: Dictionary containing participant information
        
    Returns:
        Dictionary with eligibility status and detailed criteria results
    """
    
    eligibility_results = {
        "eligible": True,
        "reasons": [],
        "criteria": {}
    }
    
    try:
        # Age criteria: 16-65 years old
        age = int(participant_data.get('age', 0))
        if age < 16 or age > 65:
            eligibility_results["eligible"] = False
            eligibility_results["reasons"].append(f"Age ({age}) must be between 16-65 years")
            eligibility_results["criteria"]["age"] = {"status": "fail", "value": age, "requirement": "16-65 years"}
        else:
            eligibility_results["criteria"]["age"] = {"status": "pass", "value": age, "requirement": "16-65 years"}
        
        # Weight criteria: At least 50 kg
        weight = float(participant_data.get('weight', 0))
        if weight < 50:
            eligibility_results["eligible"] = False
            eligibility_results["reasons"].append(f"Weight ({weight}kg) must be at least 50kg")
            eligibility_results["criteria"]["weight"] = {"status": "fail", "value": f"{weight}kg", "requirement": "≥50kg"}
        else:
            eligibility_results["criteria"]["weight"] = {"status": "pass", "value": f"{weight}kg", "requirement": "≥50kg"}
        
        # Sleep criteria: At least 5 hours
        sleep_hours = participant_data.get('sleep_hours')
        if sleep_hours is not None:
            try:
                sleep_hours = float(sleep_hours)
                if sleep_hours < 5:
                    eligibility_results["eligible"] = False
                    eligibility_results["reasons"].append(f"Sleep ({sleep_hours}h) must be at least 5 hours")
                    eligibility_results["criteria"]["sleep"] = {"status": "fail", "value": f"{sleep_hours}h", "requirement": "≥5 hours"}
                else:
                    eligibility_results["criteria"]["sleep"] = {"status": "pass", "value": f"{sleep_hours}h", "requirement": "≥5 hours"}
            except (ValueError, TypeError):
                eligibility_results["criteria"]["sleep"] = {"status": "unknown", "value": "Not provided", "requirement": "≥5 hours"}
        else:
            eligibility_results["criteria"]["sleep"] = {"status": "unknown", "value": "Not provided", "requirement": "≥5 hours"}
        
        # Alcohol criteria: No alcohol in last 24h
        had_alcohol = participant_data.get('had_alcohol_last_24h')
        if had_alcohol is not None:
            try:
                had_alcohol = str(had_alcohol).lower() in ['true', '1', 'yes']
                if had_alcohol:
                    eligibility_results["eligible"] = False
                    eligibility_results["reasons"].append("Had alcohol within last 24 hours")
                    eligibility_results["criteria"]["alcohol"] = {"status": "fail", "value": "Yes", "requirement": "None in 24h"}
                else:
                    eligibility_results["criteria"]["alcohol"] = {"status": "pass", "value": "No", "requirement": "None in 24h"}
            except:
                eligibility_results["criteria"]["alcohol"] = {"status": "unknown", "value": "Not provided", "requirement": "None in 24h"}
        else:
            eligibility_results["criteria"]["alcohol"] = {"status": "unknown", "value": "Not provided", "requirement": "None in 24h"}
        
        # Food criteria: Must have eaten, avoid fatty food
        ate_before = participant_data.get('ate_before_donation')
        ate_fatty = participant_data.get('ate_fatty_food')
        
        if ate_before is not None:
            try:
                ate_before = str(ate_before).lower() in ['true', '1', 'yes']
                if not ate_before:
                    eligibility_results["eligible"] = False
                    eligibility_results["reasons"].append("Must eat before donation")
                    eligibility_results["criteria"]["food_eaten"] = {"status": "fail", "value": "No", "requirement": "Must eat before"}
                else:
                    eligibility_results["criteria"]["food_eaten"] = {"status": "pass", "value": "Yes", "requirement": "Must eat before"}
            except:
                eligibility_results["criteria"]["food_eaten"] = {"status": "unknown", "value": "Not provided", "requirement": "Must eat before"}
        else:
            eligibility_results["criteria"]["food_eaten"] = {"status": "unknown", "value": "Not provided", "requirement": "Must eat before"}
            
        if ate_fatty is not None:
            try:
                ate_fatty = str(ate_fatty).lower() in ['true', '1', 'yes']
                if ate_fatty:
                    eligibility_results["eligible"] = False
                    eligibility_results["reasons"].append("Ate fatty food before donation")
                    eligibility_results["criteria"]["fatty_food"] = {"status": "fail", "value": "Yes", "requirement": "Avoid fatty food"}
                else:
                    eligibility_results["criteria"]["fatty_food"] = {"status": "pass", "value": "No", "requirement": "Avoid fatty food"}
            except:
                eligibility_results["criteria"]["fatty_food"] = {"status": "unknown", "value": "Not provided", "requirement": "Avoid fatty food"}
        else:
            eligibility_results["criteria"]["fatty_food"] = {"status": "unknown", "value": "Not provided", "requirement": "Avoid fatty food"}
        
        # Tattoo/Piercing criteria: Recent ones may defer donation
        recent_tattoo = participant_data.get('recent_tattoo_or_piercing')
        if recent_tattoo is not None:
            try:
                recent_tattoo = str(recent_tattoo).lower() in ['true', '1', 'yes']
                if recent_tattoo:
                    eligibility_results["eligible"] = False
                    eligibility_results["reasons"].append("Recent tattoo/piercing (within 6-12 months)")
                    eligibility_results["criteria"]["tattoo_piercing"] = {"status": "fail", "value": "Yes", "requirement": "None recent (6-12 months)"}
                else:
                    eligibility_results["criteria"]["tattoo_piercing"] = {"status": "pass", "value": "No", "requirement": "None recent (6-12 months)"}
            except:
                eligibility_results["criteria"]["tattoo_piercing"] = {"status": "unknown", "value": "Not provided", "requirement": "None recent (6-12 months)"}
        else:
            eligibility_results["criteria"]["tattoo_piercing"] = {"status": "unknown", "value": "Not provided", "requirement": "None recent (6-12 months)"}
        
        # Chronic condition criteria: Must be controlled
        has_chronic = participant_data.get('has_chronic_condition')
        condition_controlled = participant_data.get('condition_controlled')
        
        if has_chronic is not None:
            try:
                has_chronic = str(has_chronic).lower() in ['true', '1', 'yes']
                if has_chronic:
                    if condition_controlled is not None:
                        try:
                            condition_controlled = str(condition_controlled).lower() in ['true', '1', 'yes']
                            if not condition_controlled:
                                eligibility_results["eligible"] = False
                                eligibility_results["reasons"].append("Chronic condition not controlled")
                                eligibility_results["criteria"]["chronic_condition"] = {"status": "fail", "value": "Not controlled", "requirement": "Must be controlled"}
                            else:
                                eligibility_results["criteria"]["chronic_condition"] = {"status": "pass", "value": "Controlled", "requirement": "Must be controlled"}
                        except:
                            eligibility_results["eligible"] = False
                            eligibility_results["reasons"].append("Chronic condition control status unknown")
                            eligibility_results["criteria"]["chronic_condition"] = {"status": "fail", "value": "Unknown control", "requirement": "Must be controlled"}
                    else:
                        eligibility_results["eligible"] = False
                        eligibility_results["reasons"].append("Chronic condition control status not provided")
                        eligibility_results["criteria"]["chronic_condition"] = {"status": "fail", "value": "Control unknown", "requirement": "Must be controlled"}
                else:
                    eligibility_results["criteria"]["chronic_condition"] = {"status": "pass", "value": "None", "requirement": "Must be controlled"}
            except:
                eligibility_results["criteria"]["chronic_condition"] = {"status": "unknown", "value": "Not provided", "requirement": "Must be controlled"}
        else:
            eligibility_results["criteria"]["chronic_condition"] = {"status": "unknown", "value": "Not provided", "requirement": "Must be controlled"}
        
        # Last donation date criteria (optional check)
        last_donation = participant_data.get('last_donation_date')
        if last_donation:
            # Could add logic here to check if enough time has passed since last donation
            # For now, just record the information
            eligibility_results["criteria"]["last_donation"] = {"status": "info", "value": last_donation, "requirement": "Check interval"}
        
        # Summary
        eligibility_results["summary"] = {
            "total_criteria": len([c for c in eligibility_results["criteria"].values() if c["status"] in ["pass", "fail"]]),
            "passed_criteria": len([c for c in eligibility_results["criteria"].values() if c["status"] == "pass"]),
            "failed_criteria": len([c for c in eligibility_results["criteria"].values() if c["status"] == "fail"]),
            "unknown_criteria": len([c for c in eligibility_results["criteria"].values() if c["status"] == "unknown"])
        }
        
    except Exception as e:
        eligibility_results = {
            "eligible": False,
            "error": f"Error analyzing eligibility: {str(e)}",
            "reasons": ["Unable to complete eligibility analysis"],
            "criteria": {}
        }
    
    return eligibility_results


@api.post("/predict-diabetes/")
def predict_diabetes(request, participant_id: int = Form(...), consent: bool = Form(True)):
    """Predict diabetes risk for a participant using their data and fingerprints. If consent is True, save result."""
    try:
        # Check if ML packages are available
        if not ML_DIABETES_AVAILABLE:
            return JsonResponse({
                "error": "Diabetes prediction not available - ML packages not installed",
                "message": "This feature is disabled in minimal deployment mode",
                "debug": {
                    "ml_diabetes_available": ML_DIABETES_AVAILABLE,
                    "help": "Check /api/core/health/ for detailed ML package status"
                }
            }, status=503)
            
        # Get participant
        participant = Participant.objects.get(id=participant_id)
        # Initialize predictor
        predictor = DiabetesPredictor()
        # Get prediction
        prediction_result = predictor.predict_diabetes_risk(participant)
        if prediction_result.get('error'):
            return {
                "success": False,
                "error": prediction_result['error']
            }
        if consent:
            # Save result to database
            result = Result.objects.create(
                participant=participant,
                diabetes_risk=prediction_result['risk'],
                confidence_score=prediction_result['confidence']
            )
            return {
                "success": True,
                "participant_id": participant_id,
                "diabetes_risk": prediction_result['risk'],
                "confidence": prediction_result['confidence'],
                "result_id": result.id,
                "features_used": prediction_result.get('features_used', []),
                "prediction_details": {
                    "age": participant.age,
                    "gender": participant.gender,
                    "height": participant.height,
                    "weight": participant.weight,
                    "blood_type": participant.blood_type,
                    "fingerprint_count": participant.fingerprints.count()
                },
                "saved": True
            }
        else:
            return {
                "success": True,
                "participant_id": participant_id,
                "diabetes_risk": prediction_result['risk'],
                "confidence": prediction_result['confidence'],
                "features_used": prediction_result.get('features_used', []),
                "prediction_details": {
                    "age": participant.age,
                    "gender": participant.gender,
                    "height": participant.height,
                    "weight": participant.weight,
                    "blood_type": participant.blood_type,
                    "fingerprint_count": participant.fingerprints.count()
                },
                "saved": False
            }
    except Participant.DoesNotExist:
        return {
            "success": False,
            "error": f"Participant with ID {participant_id} not found"
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Prediction failed: {str(e)}"
        }


@api.get("/participant/{participant_id}/data/")
def get_participant_data(request, participant_id: int):
    """Get participant data formatted like the dataset for diabetes prediction"""
    try:
        participant = Participant.objects.get(id=participant_id)
        predictor = DiabetesPredictor()
        
        # Get formatted data
        participant_data = predictor.prepare_participant_data(participant)
        
        return {
            "success": True,
            "participant_id": participant_id,
            "raw_data": participant_data,
            "dataset_format": {
                "ready_for_model": True,
                "missing_fingerprints": [k for k, v in participant_data.items() 
                                       if k.startswith(('left_', 'right_')) and v is None]
            }
        }
        
    except Participant.DoesNotExist:
        return {
            "success": False,
            "error": f"Participant with ID {participant_id} not found"
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Data extraction failed: {str(e)}"
        }


@api.post("/predict-diabetes-from-json/")
def predict_diabetes_from_json(request):
    """Predict diabetes risk using JSON data from submit response (for consent=false)"""
    try:
        
        # Parse JSON body
        body = json.loads(request.body)
        participant_data = body.get('participant_data', {})
        fingerprints = body.get('fingerprints', [])
        
        # Build fingerprint patterns dict
        fingerprint_patterns = {}
        for fp in fingerprints:
            finger_name = fp.get('finger')
            pattern = fp.get('pattern')
            if finger_name and pattern:
                fingerprint_patterns[finger_name] = pattern
        
        # Build data for prediction
        prediction_data = {
            "age": participant_data.get("age", 0),
            "weight": participant_data.get("weight", 0),
            "height": participant_data.get("height", 0),
            "blood_type": participant_data.get("blood_type", ""),
            "gender": participant_data.get("gender", ""),
            "left_thumb": fingerprint_patterns.get("left_thumb"),
            "left_index": fingerprint_patterns.get("left_index"),
            "left_middle": fingerprint_patterns.get("left_middle"),
            "left_ring": fingerprint_patterns.get("left_ring"),
            "left_pinky": fingerprint_patterns.get("left_pinky"),
            "right_thumb": fingerprint_patterns.get("right_thumb"),
            "right_index": fingerprint_patterns.get("right_index"),
            "right_middle": fingerprint_patterns.get("right_middle"),
            "right_ring": fingerprint_patterns.get("right_ring"),
            "right_pinky": fingerprint_patterns.get("right_pinky"),
        }
        
        # Make prediction
        predictor = DiabetesPredictor()
        df = predictor.prepare_input_df(prediction_data, model_key='A')
        model = predictor.models.get('A')
        
        if model is None:
            return {"success": False, "error": "Model not loaded"}
        
        pred = model.predict(df)[0]
        risk = 'At risk' if str(pred).lower() in ['diabetic', '1', 'at risk', 'risk', 'positive'] else 'Not at risk'
        
        return {
            "success": True,
            "diabetes_risk": risk,
            "confidence": 1.0,
            "model_used": "A",
            "prediction_details": {
                "age": prediction_data["age"],
                "gender": prediction_data["gender"],
                "height": prediction_data["height"],
                "weight": prediction_data["weight"],
                "blood_type": prediction_data["blood_type"],
                "fingerprint_count": len([p for p in fingerprint_patterns.values() if p])
            },
            "saved": False,
            "consent_given": False
        }
        
    except Exception as e:
        return {"success": False, "error": f"JSON prediction failed: {str(e)}"}

@api.post("/decrypt-data/")
def decrypt_data(request, encrypted_data: str):
    """
    Decrypt data sent from the frontend.
    """
    try:
        decrypted_data = encryption_service.decrypt_string(encrypted_data)
        return JsonResponse({"success": True, "decrypted_data": decrypted_data})
    except Exception as e:
        return JsonResponse({"success": False, "error": str(e)})

@api.post("/encrypt-data/")
def encrypt_data(request, data: str):
    """
    Encrypt data and return encrypted result.
    """
    try:
        encrypted_data = encryption_service.encrypt_string(data)
        return JsonResponse({"success": True, "encrypted_data": encrypted_data})
    except Exception as e:
        return JsonResponse({"success": False, "error": str(e)})

# QR Code PDF Download Implementation
import uuid
from django.core.cache import cache
from django.http import FileResponse, Http404
import tempfile
from reportlab.lib.pagesizes import letter, A4
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
import io
from datetime import datetime

@api.post("/generate-pdf-token-direct/")
def generate_pdf_token_direct(request):
    """Generate a temporary token for PDF download using provided data (no database storage required)"""
    try:
        # Parse JSON data from request body
        import json
        data = json.loads(request.body)
        
        participant_data = data.get('participant_data', {})
        diabetes_result = data.get('diabetes_result', {})
        blood_group_result = data.get('blood_group_result', {})
        
        # Validate required data
        if not participant_data:
            return {"success": False, "error": "Participant data is required"}
        
        # Generate unique token
        token = str(uuid.uuid4())
        
        # Prepare participant data for analysis
        participant_analysis_data = {
            'id': None,  # No database ID since not saved
            'age': participant_data.get('age'),
            'gender': participant_data.get('gender'),
            'height': participant_data.get('height'),
            'weight': participant_data.get('weight'),
            'blood_type': participant_data.get('blood_type'),
            'willing_to_donate': participant_data.get('willing_to_donate'),
            'sleep_hours': participant_data.get('sleep_hours'),
            'had_alcohol_last_24h': participant_data.get('had_alcohol_last_24h'),
            'ate_before_donation': participant_data.get('ate_before_donation'),
            'ate_fatty_food': participant_data.get('ate_fatty_food'),
            'recent_tattoo_or_piercing': participant_data.get('recent_tattoo_or_piercing'),
            'has_chronic_condition': participant_data.get('has_chronic_condition'),
            'condition_controlled': participant_data.get('condition_controlled'),
            'last_donation_date': participant_data.get('last_donation_date'),
            'created_at': datetime.now().isoformat()
        }
        
        # Analyze blood donation eligibility if willing to donate
        blood_donation_eligibility = None
        if participant_data.get('willing_to_donate'):
            print("🩸 Analyzing blood donation eligibility for non-consenting user...")
            blood_donation_eligibility = analyze_blood_donation_eligibility(participant_analysis_data)
            print(f"🩸 Eligibility result: {blood_donation_eligibility.get('eligible', False)}")
        
        # Store data temporarily (10 minutes)
        pdf_data = {
            'participant': participant_analysis_data,
            'diabetes_result': {
                'risk': diabetes_result.get('diabetes_risk') or diabetes_result.get('risk'),
                'confidence': diabetes_result.get('confidence', 0)
            },
            'blood_group_result': {
                'predicted_blood_group': blood_group_result.get('predicted_blood_group'),
                'confidence': blood_group_result.get('confidence', 0)
            },
            'blood_donation_eligibility': blood_donation_eligibility,
            'fingerprint_count': len(data.get('fingerprints', [])),
            'generated_at': datetime.now().isoformat()
        }
        
        cache.set(f"pdf_data_{token}", pdf_data, timeout=600)  # 10 minutes
        
        return {
            "success": True,
            "download_token": token,
            "expires_in": 600,
            "download_url": f"/api/core/download-pdf/{token}/"
        }
        
    except json.JSONDecodeError:
        return {"success": False, "error": "Invalid JSON data"}
    except Exception as e:
        return {"success": False, "error": f"Failed to generate PDF token: {str(e)}"}

@api.post("/generate-pdf-token/")
def generate_pdf_token(request, participant_id: int = Form(...)):
    """Generate a temporary token for PDF download without saving PDF to database"""
    try:
        # Get participant data
        participant = Participant.objects.get(id=participant_id)
        
        # Get results data
        diabetes_result = None
        blood_group_result = None
        
        try:
            # Try to get saved results first
            result = Result.objects.filter(participant=participant).first()
            if result:
                diabetes_result = {
                    'risk': result.diabetes_risk,
                    'confidence': result.confidence_score
                }
        except:
            pass
        
        # Generate blood group prediction if not available
        if not blood_group_result:
            try:
                bg_response = identify_blood_group_from_participant(request, participant_id)
                if bg_response.get('success'):
                    blood_group_result = {
                        'predicted_blood_group': bg_response.get('predicted_blood_group'),
                        'confidence': bg_response.get('confidence', 0)
                    }
            except:
                blood_group_result = {
                    'predicted_blood_group': 'Unknown',
                    'confidence': 0
                }
        
        # Generate diabetes prediction if not available
        if not diabetes_result:
            try:
                diabetes_response = predict_diabetes(request, participant_id, consent=False)
                if diabetes_response.get('success'):
                    diabetes_result = {
                        'risk': diabetes_response.get('diabetes_risk'),
                        'confidence': diabetes_response.get('confidence', 0)
                    }
            except:
                diabetes_result = {
                    'risk': 'Unknown',
                    'confidence': 0
                }
        
        # Generate unique token
        token = str(uuid.uuid4())
        print(f"🔑 Generated token: {token}")
        
        # Prepare participant data for analysis
        participant_data = {
            'id': participant.id,
            'age': participant.age,
            'gender': participant.gender,
            'height': participant.height,
            'weight': participant.weight,
            'blood_type': participant.blood_type,
            'willing_to_donate': participant.willing_to_donate,
            'sleep_hours': participant.sleep_hours,
            'had_alcohol_last_24h': participant.had_alcohol_last_24h,
            'ate_before_donation': participant.ate_before_donation,
            'ate_fatty_food': participant.ate_fatty_food,
            'recent_tattoo_or_piercing': participant.recent_tattoo_or_piercing,
            'has_chronic_condition': participant.has_chronic_condition,
            'condition_controlled': participant.condition_controlled,
            'last_donation_date': participant.last_donation_date.isoformat() if participant.last_donation_date else None,
            'created_at': participant.created_at.isoformat() if participant.created_at else None
        }
        
        # Analyze blood donation eligibility if willing to donate
        blood_donation_eligibility = None
        if participant.willing_to_donate:
            print("🩸 Analyzing blood donation eligibility...")
            blood_donation_eligibility = analyze_blood_donation_eligibility(participant_data)
            print(f"🩸 Eligibility result: {blood_donation_eligibility.get('eligible', False)}")
        
        # Store data temporarily (10 minutes)
        pdf_data = {
            'participant': participant_data,
            'diabetes_result': diabetes_result,
            'blood_group_result': blood_group_result,
            'blood_donation_eligibility': blood_donation_eligibility,
            'fingerprint_count': participant.fingerprints.count(),
            'generated_at': datetime.now().isoformat()
        }
        
        cache_key = f"pdf_data_{token}"
        print(f"💾 Storing data in cache with key: {cache_key}")
        print(f"💾 Data to store: {pdf_data}")
        
        cache.set(cache_key, pdf_data, timeout=600)  # 10 minutes
        
        # Verify it was stored
        stored_data = cache.get(cache_key)
        if stored_data:
            print(f"✅ Data successfully stored in cache")
        else:
            print(f"❌ Failed to store data in cache")
        
        download_url = f"/api/core/download-pdf/{token}/"
        print(f"🔗 Generated download URL: {download_url}")
        
        return {
            "success": True,
            "download_token": token,
            "expires_in": 600,
            "download_url": download_url
        }
        
    except Participant.DoesNotExist:
        return {"success": False, "error": "Participant not found"}
    except Exception as e:
        return {"success": False, "error": f"Failed to generate PDF token: {str(e)}"}

@api.get("/test-cache/")
def test_cache(request):
    """Test if cache is working"""
    try:
        test_key = "test_cache_key"
        test_value = {"test": "data", "timestamp": datetime.now().isoformat()}
        
        # Set cache
        cache.set(test_key, test_value, timeout=60)
        
        # Get cache
        retrieved_value = cache.get(test_key)
        
        return {
            "success": True,
            "cache_working": retrieved_value is not None,
            "stored_value": test_value,
            "retrieved_value": retrieved_value
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

@api.get("/test-pdf/")
def test_pdf_generation(request):
    """Test PDF generation with sample data"""
    try:
        # Sample data
        test_data = {
            'participant': {
                'age': 25,
                'gender': 'Male',
                'height': 175,
                'weight': 70,
                'blood_type': 'A+',
                'willing_to_donate': True
            },
            'diabetes_result': {
                'risk': 'Not at risk',
                'confidence': 0.95
            },
            'blood_group_result': {
                'predicted_blood_group': 'A+',
                'confidence': 0.92
            },
            'fingerprint_count': 10,
            'generated_at': datetime.now().isoformat()
        }
        
        # Try to generate PDF
        pdf_buffer, content_type, extension = generate_health_report_pdf(test_data)
        
        # Return the PDF directly
        from django.http import FileResponse
        response = FileResponse(
            pdf_buffer,
            as_attachment=True,
            filename=f"test_report{extension}",
            content_type=content_type
        )
        return response
        
    except Exception as e:
        import traceback
        return {
            "success": False, 
            "error": str(e),
            "traceback": traceback.format_exc()
        }

@api.get("/download-pdf/{token}/")
def download_pdf(request, token: str):
    """Download PDF using temporary token"""
    try:
        print(f"🔍 PDF download request for token: {token}")
        
        # Get data from cache
        cache_key = f"pdf_data_{token}"
        print(f"🔍 Looking for cache key: {cache_key}")
        
        pdf_data = cache.get(cache_key)
        if not pdf_data:
            print(f"❌ No data found in cache for token: {token}")
            # Let's check what keys exist in cache and test cache functionality
            try:
                from django.core.cache import cache as django_cache
                print(f"🔍 Cache backend: {type(django_cache)}")
                
                # Test if cache is working at all
                test_key = f"test_{token}"
                cache.set(test_key, {"test": "value"}, timeout=10)
                test_result = cache.get(test_key)
                print(f"🔍 Cache test result: {test_result}")
                
                # Try to list any pdf_data keys (if possible)
                import re
                if hasattr(django_cache, '_cache'):
                    print(f"🔍 Cache keys preview: {list(django_cache._cache.keys())[:10] if hasattr(django_cache._cache, 'keys') else 'Cannot list keys'}")
                    
            except Exception as cache_test_error:
                print(f"❌ Cache test error: {cache_test_error}")
                
            raise Http404("Download link has expired or is invalid")
        
        print(f"✅ Found PDF data in cache: {pdf_data}")
        
        # Generate PDF with fallback handling
        print("📄 Generating PDF...")
        pdf_buffer, content_type, extension = generate_health_report_pdf(pdf_data)
        print(f"✅ PDF generated successfully - Type: {content_type}, Extension: {extension}")
        
        # Don't delete immediately - allow multiple downloads within time limit
        # The cache will expire naturally after 10 minutes
        print(f"� PDF downloaded successfully, keeping cache entry until expiry")
        
        # Return appropriate response based on content type
        filename = f"printalyzer_health_report_{token[:8]}{extension}"
        
        if content_type == 'application/pdf':
            response = FileResponse(
                pdf_buffer,
                as_attachment=True,
                filename=filename,
                content_type=content_type
            )
        else:
            # Text fallback
            from django.http import HttpResponse
            content = pdf_buffer.read().decode('utf-8')
            response = HttpResponse(
                content,
                content_type=content_type,
                headers={
                    'Content-Disposition': f'attachment; filename="{filename}"'
                }
            )
        
        # Add CORS headers for mobile browsers
        response['Access-Control-Allow-Origin'] = '*'
        response['Access-Control-Allow-Methods'] = 'GET'
        response['Access-Control-Allow-Headers'] = 'Content-Type'
        
        print("✅ Returning PDF response")
        return response
        
    except Exception as e:
        print(f"❌ PDF download error: {str(e)}")
        print(f"❌ Exception type: {type(e)}")
        import traceback
        print(f"❌ Traceback: {traceback.format_exc()}")
        raise Http404(f"Download failed: {str(e)}")

def generate_health_report_pdf_simple(pdf_data):
    """Generate a basic PDF with proper PDF format"""
    import io
    
    # Extract values safely
    age = pdf_data['participant'].get('age', 'Unknown')
    gender = pdf_data['participant'].get('gender', 'Unknown')
    height = pdf_data['participant'].get('height', 'Unknown')
    weight = pdf_data['participant'].get('weight', 'Unknown')
    blood_type = pdf_data['participant'].get('blood_type', 'Not specified')
    willing_to_donate = 'Yes' if pdf_data['participant'].get('willing_to_donate') else 'No'
    
    predicted_bg = pdf_data.get('blood_group_result', {}).get('predicted_blood_group', 'Unknown')
    bg_confidence = pdf_data.get('blood_group_result', {}).get('confidence', 0) * 100
    
    diabetes_risk = pdf_data.get('diabetes_result', {}).get('risk', 'Unknown')
    diabetes_confidence = pdf_data.get('diabetes_result', {}).get('confidence', 0) * 100
    
    try:
        gen_date = datetime.fromisoformat(pdf_data['generated_at']).strftime('%B %d, %Y')
    except:
        gen_date = 'Unknown'
    
    # Create PDF content with actual values
    content = f"""%PDF-1.4
1 0 obj
<<
/Type /Catalog
/Pages 2 0 R
>>
endobj

2 0 obj
<<
/Type /Pages
/Kids [3 0 R]
/Count 1
>>
endobj

3 0 obj
<<
/Type /Page
/Parent 2 0 R
/MediaBox [0 0 612 792]
/Contents 4 0 R
/Resources <<
  /Font << /F1 5 0 R >>
>>
>>
endobj

4 0 obj
<<
/Length 1500
>>
stream
BT
/F1 16 Tf
50 750 Td
(PRINTALYZER HEALTH REPORT) Tj
0 -40 Td
/F1 12 Tf
(AI-Powered Health Analysis) Tj
0 -40 Td
(Patient Information:) Tj
0 -20 Td
(Age: {age} years) Tj
0 -20 Td
(Gender: {gender}) Tj
0 -20 Td
(Height: {height} cm) Tj
0 -20 Td
(Weight: {weight} kg) Tj
0 -20 Td
(Blood Type: {blood_type}) Tj
0 -20 Td
(Willing to Donate: {willing_to_donate}) Tj
0 -40 Td
(Blood Group Prediction:) Tj
0 -20 Td
(Predicted: {predicted_bg}) Tj
0 -20 Td
(Confidence: {bg_confidence:.1f}%) Tj
0 -40 Td
(Diabetes Risk Assessment:) Tj
0 -20 Td
(Risk Level: {diabetes_risk}) Tj
0 -20 Td
(Confidence: {diabetes_confidence:.1f}%) Tj
0 -40 Td
(IMPORTANT DISCLAIMER:) Tj
0 -20 Td
(This is a screening tool for educational purposes only.) Tj
0 -20 Td
(Consult healthcare professionals for medical advice.) Tj
0 -40 Td
(Generated: {gen_date}) Tj
ET
endstream
endobj

5 0 obj
<<
/Type /Font
/Subtype /Type1
/BaseFont /Helvetica
>>
endobj

xref
0 6
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
0000000254 00000 n 
0000001800 00000 n 
trailer
<<
/Size 6
/Root 1 0 R
>>
startxref
1878
%%EOF"""

    buffer = io.BytesIO()
    buffer.write(content.encode('utf-8'))
    buffer.seek(0)
    return buffer

def generate_simple_text_pdf(pdf_data):
    """Generate a simple text-based file as last fallback"""
    import io
    
    # Create a simple text content
    content = f"""
PRINTALYZER HEALTH REPORT
========================

Patient Information:
- Age: {pdf_data['participant']['age']} years
- Gender: {pdf_data['participant']['gender']}
- Height: {pdf_data['participant']['height']} cm
- Weight: {pdf_data['participant']['weight']} kg
- Blood Type: {pdf_data['participant'].get('blood_type', 'Not specified')}
- Willing to Donate: {'Yes' if pdf_data['participant']['willing_to_donate'] else 'No'}

Blood Group Prediction:
- Predicted: {pdf_data.get('blood_group_result', {}).get('predicted_blood_group', 'Unknown')}
- Confidence: {pdf_data.get('blood_group_result', {}).get('confidence', 0) * 100:.1f}%

Diabetes Risk Assessment:
- Risk Level: {pdf_data.get('diabetes_result', {}).get('risk', 'Unknown')}
- Confidence: {pdf_data.get('diabetes_result', {}).get('confidence', 0) * 100:.1f}%

IMPORTANT DISCLAIMER:
This is a screening tool for educational purposes only.
Consult healthcare professionals for medical advice.

Generated: {pdf_data.get('generated_at', 'Unknown')}
"""
    
    # Return as plain text in a buffer
    buffer = io.BytesIO()
    buffer.write(content.encode('utf-8'))
    buffer.seek(0)
    return buffer

def generate_health_report_pdf(pdf_data):
    """Generate a styled PDF health report with multiple fallback options"""
    try:
        # Try our simple PDF generator first (most reliable)
        print("🔄 Attempting simple PDF generation")
        pdf_buffer = generate_health_report_pdf_simple(pdf_data)
        print("✅ Simple PDF generation successful")
        return pdf_buffer, 'application/pdf', '.pdf'
    except Exception as e:
        print(f"⚠️ Simple PDF generation failed: {str(e)}")
        
        try:
            # Try using ReportLab
            from reportlab.pdfgen import canvas
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.colors import HexColor
            
            buffer = io.BytesIO()
            c = canvas.Canvas(buffer, pagesize=A4)
            width, height = A4
            
            # If we get here, ReportLab is available
            print("🔄 Attempting ReportLab PDF generation")
            pdf_buffer = generate_reportlab_pdf(pdf_data, c, width, height, buffer)
            print("✅ ReportLab PDF generation successful")
            return pdf_buffer, 'application/pdf', '.pdf'
            
        except ImportError:
            print("⚠️ ReportLab not available, trying manual PDF generation")
            try:
                pdf_buffer = generate_manual_pdf(pdf_data)
                print("✅ Manual PDF generation successful")
                return pdf_buffer, 'application/pdf', '.pdf'
            except Exception as e2:
                print(f"❌ Manual PDF generation failed: {str(e2)}")
                # Final fallback to text
                text_buffer = generate_simple_text_pdf(pdf_data)
                return text_buffer, 'text/plain', '.txt'
        except Exception as e:
            print(f"❌ ReportLab PDF generation failed: {str(e)}")
            # Final fallback to text
            text_buffer = generate_simple_text_pdf(pdf_data)
            return text_buffer, 'text/plain', '.txt'

def generate_manual_pdf(pdf_data):
    """Generate a basic PDF manually without ReportLab"""
    import io
    
    # Create a very basic PDF structure
    content = f"""%PDF-1.4
1 0 obj
<<
/Type /Catalog
/Pages 2 0 R
>>
endobj

2 0 obj
<<
/Type /Pages
/Kids [3 0 R]
/Count 1
>>
endobj

3 0 obj
<<
/Type /Page
/Parent 2 0 R
/MediaBox [0 0 612 792]
/Contents 4 0 R
/Resources <<
  /Font << /F1 5 0 R >>
>>
>>
endobj

4 0 obj
<<
/Length {len("BT /F1 12 Tf 50 750 Td")}
>>
stream
BT
/F1 12 Tf
50 750 Td
(Printalyzer Health Report) Tj
0 -30 Td
(Patient Information:) Tj
0 -20 Td
(Age: {pdf_data['participant']['age']} years) Tj
0 -20 Td
(Gender: {pdf_data['participant']['gender']}) Tj
0 -20 Td
(Height: {pdf_data['participant']['height']} cm) Tj
0 -20 Td
(Weight: {pdf_data['participant']['weight']} kg) Tj
0 -20 Td
(Blood Type: {pdf_data['participant'].get('blood_type', 'Not specified')}) Tj
0 -40 Td
(Blood Group Prediction:) Tj
0 -20 Td
(Predicted: {pdf_data.get('blood_group_result', {}).get('predicted_blood_group', 'Unknown')}) Tj
0 -20 Td
(Confidence: {pdf_data.get('blood_group_result', {}).get('confidence', 0) * 100:.1f}%) Tj
0 -40 Td
(Diabetes Risk Assessment:) Tj
0 -20 Td
(Risk Level: {pdf_data.get('diabetes_result', {}).get('risk', 'Unknown')}) Tj
0 -20 Td
(Confidence: {pdf_data.get('diabetes_result', {}).get('confidence', 0) * 100:.1f}%) Tj
0 -40 Td
(DISCLAIMER: This is for educational purposes only.) Tj
0 -20 Td
(Consult healthcare professionals for medical advice.) Tj
ET
endstream
endobj

5 0 obj
<<
/Type /Font
/Subtype /Type1
/BaseFont /Helvetica
>>
endobj

xref
0 6
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
0000000254 00000 n 
0000001000 00000 n 
trailer
<<
/Size 6
/Root 1 0 R
>>
startxref
1078
%%EOF"""

    buffer = io.BytesIO()
    buffer.write(content.encode('utf-8'))
    buffer.seek(0)
    return buffer

def generate_reportlab_pdf(pdf_data, c, width, height, buffer):
    """Generate PDF using ReportLab"""
    from reportlab.lib.colors import HexColor
    
    # Colors
    primary_color = HexColor('#00c2cb')
    text_color = HexColor('#1f2937')
    gray_color = HexColor('#6b7280')
    red_color = HexColor('#dc2626')
    green_color = HexColor('#059669')
    
    # Colors
    primary_color = HexColor('#00c2cb')
    text_color = HexColor('#1f2937')
    gray_color = HexColor('#6b7280')
    red_color = HexColor('#dc2626')
    green_color = HexColor('#059669')
    
    # Header
    c.setFont("Helvetica-Bold", 24)
    c.setFillColor(primary_color)
    c.drawCentredText(width/2, height-80, "🔬 Printalyzer Health Report")
    
    c.setFont("Helvetica", 14)
    c.setFillColor(text_color)
    c.drawCentredText(width/2, height-110, "AI-Powered Diabetes Risk & Blood Group Analysis")
    
    # Generation date
    gen_date = datetime.fromisoformat(pdf_data['generated_at']).strftime('%B %d, %Y at %I:%M %p')
    c.setFont("Helvetica", 10)
    c.setFillColor(gray_color)
    c.drawCentredText(width/2, height-130, f"Generated: {gen_date}")
    
    # Patient Information Section
    y = height - 180
    c.setFont("Helvetica-Bold", 16)
    c.setFillColor(text_color)
    c.drawString(50, y, "👤 Patient Information")
    
    y -= 30
    c.setFont("Helvetica", 11)
    patient_info = [
        f"Age: {pdf_data['participant']['age']} years",
        f"Gender: {pdf_data['participant']['gender']}",
        f"Height: {pdf_data['participant']['height']} cm",
        f"Weight: {pdf_data['participant']['weight']} kg",
        f"Actual Blood Type: {pdf_data['participant'].get('blood_type', 'Not specified')}",
        f"Willing to Donate: {'Yes' if pdf_data['participant']['willing_to_donate'] else 'No'}"
    ]
    
    for info in patient_info:
        c.drawString(70, y, info)
        y -= 18
    
    # Blood Group Results
    y -= 30
    c.setFont("Helvetica-Bold", 14)
    c.setFillColor(red_color)
    c.drawString(50, y, "🩸 Blood Group Prediction")
    
    y -= 25
    c.setFont("Helvetica", 11)
    c.setFillColor(text_color)
    bg_result = pdf_data.get('blood_group_result', {})
    bg_confidence = bg_result.get('confidence', 0) * 100
    
    bg_info = [
        f"Predicted Blood Group: {bg_result.get('predicted_blood_group', 'Unknown')}",
        f"Confidence Level: {bg_confidence:.1f}%",
        f"Analysis Method: Dermatoglyphic Pattern Recognition"
    ]
    
    for info in bg_info:
        c.drawString(70, y, info)
        y -= 18
    
    # Diabetes Risk Results
    y -= 30
    c.setFont("Helvetica-Bold", 14)
    c.setFillColor(green_color)
    c.drawString(50, y, "🏥 Diabetes Risk Assessment")
    
    y -= 25
    c.setFont("Helvetica", 11)
    c.setFillColor(text_color)
    diabetes_result = pdf_data.get('diabetes_result', {})
    diabetes_confidence = diabetes_result.get('confidence', 0) * 100
    risk_level = diabetes_result.get('risk', 'Unknown')
    
    diabetes_info = [
        f"Risk Assessment: {risk_level}",
        f"Confidence Level: {diabetes_confidence:.1f}%",
        f"Analysis Method: Machine Learning Algorithm",
        f"Fingerprints Analyzed: {pdf_data.get('fingerprint_count', 0)}"
    ]
    
    for info in diabetes_info:
        c.drawString(70, y, info)
        y -= 18
    
    # Medical Disclaimer
    y -= 40
    c.setFont("Helvetica-Bold", 12)
    c.setFillColor(HexColor('#d97706'))
    c.drawString(50, y, "⚠️ Important Medical Disclaimer")
    
    y -= 25
    c.setFont("Helvetica", 9)
    c.setFillColor(HexColor('#92400e'))
    
    disclaimer_lines = [
        "This is a risk prediction tool for educational and screening purposes only.",
        "",
        "• Results are generated using artificial intelligence and should not be considered as medical diagnosis",
        "• Always consult qualified healthcare professionals for medical advice and formal testing",
        "• This analysis is based on fingerprint patterns and statistical correlations",
        "• Individual health conditions may vary and require professional medical evaluation",
        "",
        "For medical concerns, please contact your healthcare provider immediately."
    ]
    
    for line in disclaimer_lines:
        if line:  # Skip empty lines for spacing
            c.drawString(70, y, line)
        y -= 12
    
    # Footer
    c.setFont("Helvetica", 8)
    c.setFillColor(gray_color)
    c.drawCentredText(width/2, 50, "Generated by Printalyzer AI Health Analysis System")
    
    c.save()
    buffer.seek(0)
    return buffer

