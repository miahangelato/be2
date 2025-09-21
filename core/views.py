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
        # Check if participant exists
        try:
            participant = Participant.objects.get(id=participant_id)
        except Participant.DoesNotExist:
            return {"error": "Participant not found.", "participant_id": participant_id}

        # Fetch fingerprints
        fingerprints = participant.fingerprints.all()
        if not fingerprints:
            return {"error": "No fingerprints found.", "participant_id": participant_id}

        results = []
        predicted_blood_group = None
        
        for fp in fingerprints:
            if fp.image:
                try:
                    if not ML_BLOODGROUP_AVAILABLE:
                        pred = {
                            'predicted_blood_group': 'ML_UNAVAILABLE',
                            'confidence': 0.0,
                            'all_probabilities': {}
                        }
                    else:
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
                    
                    predicted_blood_group = pred['predicted_blood_group']
                    results.append({
                        "finger": fp.finger,
                        "filename": fp.image.name.split('/')[-1] if '/' in fp.image.name else fp.image.name,
                        "predicted_blood_group": pred['predicted_blood_group'],
                        "confidence": pred['confidence'],
                        "all_probabilities": pred.get('all_probabilities'),
                    })
                except Exception as e:
                    import traceback
                    error_msg = f"Blood group prediction error: {str(e)}\n{traceback.format_exc()}"
                    print(error_msg)  # This will show in Railway logs
                    results.append({"finger": fp.finger, "error": str(e)})
            else:
                results.append({"finger": fp.finger, "error": "No image uploaded"})

        return {"participant_id": participant_id, "results": results, "predicted_blood_group": predicted_blood_group}
        
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
    try:
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
                predicted_blood_group = pred['predicted_blood_group']
                results.append({
                    "finger": fp_meta.get('finger'),
                    "image_name": img_name,
                    "predicted_blood_group": pred['predicted_blood_group'],
                    "confidence": pred['confidence'],
                    "all_probabilities": pred.get('all_probabilities'),
                })
            except Exception as e:
                results.append({
                    "finger": fp_meta.get('finger'),
                    "image_name": img_name,
                    "error": str(e)
                })
        return {"success": True, "results": results, "predicted_blood_group": predicted_blood_group}
    except Exception as e:
        return {"success": False, "error": f"Blood group identification failed: {e}"}
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
        risk = 'DIABETIC' if str(pred).lower() in ['diabetic', '1', 'at risk', 'risk', 'positive'] else 'HEALTHY'
        
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

