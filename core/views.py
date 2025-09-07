from ninja import NinjaAPI, Form, Schema, File, UploadedFile
from .models import Participant, Fingerprint, Result
from .diabetes_predictor import DiabetesPredictor
import json
from django.http import JsonResponse
from django.conf import settings
from .encryption_utils import encryption_service
from .backend_decryption import backend_decryption
import logging
import os
from typing import List

logger = logging.getLogger(__name__)

# Conditional import for blood group classification
try:
    from .bloodgroup_classifier import classify_blood_group_from_multiple
    BLOOD_GROUP_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Blood group classification not available: {e}")
    BLOOD_GROUP_AVAILABLE = False
    classify_blood_group_from_multiple = None

logger = logging.getLogger(__name__)

api = NinjaAPI()

@api.get("/health/")
def health_check(request):
    """
    Simple health check endpoint for Railway deployment
    """
    try:
        return {
            "status": "healthy",
            "message": "Application is running"
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JsonResponse({
            "status": "unhealthy",
            "error": str(e),
            "message": "Application not ready"
        }, status=503)

@api.get("/debug/")
def debug_info(request):
    """
    Debug endpoint to check environment and setup
    """
    import os
    return {
        "env_vars": {
            "DEBUG": os.getenv('DEBUG'),
            "RAILWAY_DEPLOYMENT": os.getenv('RAILWAY_DEPLOYMENT'),
            "DATABASE_URL_exists": bool(os.getenv('DATABASE_URL')),
            "PORT": os.getenv('PORT'),
        },
        "paths": {
            "BASE_DIR": str(settings.BASE_DIR),
            "working_dir": os.getcwd(),
        },
        "django": {
            "DEBUG": settings.DEBUG,
            "ALLOWED_HOSTS": settings.ALLOWED_HOSTS,
        }
    }

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
    # Fingerprint files (optional)
    left_thumb: UploadedFile = File(None),
    left_index: UploadedFile = File(None),
    left_middle: UploadedFile = File(None),
    left_ring: UploadedFile = File(None),
    left_pinky: UploadedFile = File(None),
    right_thumb: UploadedFile = File(None),
    right_index: UploadedFile = File(None),
    right_middle: UploadedFile = File(None),
    right_ring: UploadedFile = File(None),
    right_pinky: UploadedFile = File(None),
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

    # Save or return data based on consent
    if consent:
        try:
            # Test database connection first
            from django.db import connection
            with connection.cursor() as cursor:
                cursor.execute("SELECT 1")
            
            # Save participant to database
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
                consent=True  # Explicitly set consent
            )

            # Save fingerprint files if provided
            fingerprint_files = {
                'left_thumb': left_thumb,
                'left_index': left_index,
                'left_middle': left_middle,
                'left_ring': left_ring,
                'left_pinky': left_pinky,
                'right_thumb': right_thumb,
                'right_index': right_index,
                'right_middle': right_middle,
                'right_ring': right_ring,
                'right_pinky': right_pinky,
            }
            
            saved_fingerprints = []
            for finger_name, file in fingerprint_files.items():
                if file and file.name:  # Check if file is provided
                    try:
                        # Save fingerprint to database
                        fingerprint = Fingerprint.objects.create(
                            participant=participant,
                            finger=finger_name,  # Use 'finger' not 'finger_name'
                            pattern=''  # Leave pattern empty for now
                        )
                        
                        # Save the actual file (Django will handle the path via upload_path function)
                        fingerprint.image.save(file.name, file, save=True)
                        saved_fingerprints.append(finger_name)
                        
                    except Exception as fp_error:
                        logger.error(f"Failed to save fingerprint {finger_name}: {fp_error}")

            return {
                "saved": True,
                "participant_id": participant.id,
                "message": f"Data saved successfully. Participant: {participant.id}",
                "fingerprints_saved": saved_fingerprints,
                "fingerprints_count": len(saved_fingerprints)
            }
            
        except Exception as save_error:
            return {
                "saved": False,
                "error": str(save_error),
                "message": "Database save failed"
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
        }

@api.post("/predict-diabetes/")
def predict_diabetes(request, participant_id: int = Form(...), consent: bool = Form(True)):
    """Predict diabetes risk for a participant using their data. If consent is True, save result."""
    try:
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
        
        # Build data for prediction
        prediction_data = {
            "age": participant_data.get("age", 0),
            "weight": participant_data.get("weight", 0),
            "height": participant_data.get("height", 0),
            "blood_type": participant_data.get("blood_type", ""),
            "gender": participant_data.get("gender", ""),
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

@api.post("/identify-blood-group-from-participant/")
def identify_blood_group_from_participant(request, participant_id: int):
    """
    Identify blood group for each fingerprint image of a participant (by participant_id).
    Returns a list of predictions, one per fingerprint.
    """
    if not BLOOD_GROUP_AVAILABLE:
        return JsonResponse({
            "error": "Blood group classification service is not available in this deployment",
            "participant_id": participant_id
        }, status=503)
    
    print("[DEBUG] Incoming request data:")
    print(f"Participant ID: {participant_id}")
    print("[DEBUG] Request validation started")
    
    # Check if participant exists
    try:
        participant = Participant.objects.get(id=participant_id)
        print(f"[DEBUG] Found participant: {participant.id}")
    except Participant.DoesNotExist:
        print("[ERROR] Participant does not exist.")
        return JsonResponse({"error": "Participant not found.", "participant_id": participant_id}, status=404)

    # Fetch fingerprints
    fingerprints = participant.fingerprints.all()
    if not fingerprints:
        print("[ERROR] No fingerprints found for participant.")
        return JsonResponse({"error": "No fingerprints found.", "participant_id": participant_id}, status=404)

    print(f"[DEBUG] Found {len(fingerprints)} fingerprints for participant.")

    results = []
    predicted_blood_group = None
    
    for fp in fingerprints:
        print(f"[DEBUG] Processing fingerprint for finger: {fp.finger}")
        if fp.image:
            try:
                import tempfile
                import shutil
                
                print(f"[DEBUG] Processing fingerprint image: {fp.image.name}")
                
                # Create temporary file from stored image
                temp_fd, temp_path = tempfile.mkstemp(suffix='.png')
                os.close(temp_fd)
                
                try:
                    # Copy image content to temporary file
                    with open(temp_path, 'wb') as temp_file:
                        fp.image.seek(0)  # Reset file pointer
                        shutil.copyfileobj(fp.image, temp_file)
                    
                    print(f"[DEBUG] Classifying fingerprint from temp file: {temp_path}")
                    pred = classify_blood_group_from_multiple([temp_path])
                    predicted_blood_group = pred['predicted_blood_group']
                    print(f"[DEBUG] Classification result: {predicted_blood_group}")
                    
                    results.append({
                        "finger": fp.finger,
                        "filename": fp.image.name,
                        "predicted_blood_group": pred['predicted_blood_group'],
                        "confidence": pred['confidence'],
                        "all_probabilities": pred.get('all_probabilities'),
                    })
                    
                finally:
                    # Clean up temporary file
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)
                        
            except Exception as e:
                print(f"[ERROR] Failed to classify fingerprint: {e}")
                results.append({
                    "finger": fp.finger,
                    "filename": fp.image.name if fp.image else "unknown",
                    "error": str(e)
                })
        else:
            print(f"[WARNING] Fingerprint image not found or invalid for finger {fp.finger}.")
            results.append({"finger": fp.finger, "error": "Image not found"})

    print(f"[DEBUG] Final results: {results}")
    return JsonResponse({"participant_id": participant_id, "results": results, "predicted_blood_group": predicted_blood_group})

@api.post("/identify-blood-group-from-json/")
def identify_blood_group_from_json(request, json_data: str = Form(...), files: List[UploadedFile] = File(None)):
    """
    Identify blood group for each uploaded fingerprint image, using metadata from JSON (consent=false flow).
    """
    if not BLOOD_GROUP_AVAILABLE:
        return JsonResponse({
            "success": False,
            "error": "Blood group classification service is not available in this deployment"
        }, status=503)
    
    import tempfile
    import shutil
    import json as pyjson
    
    results = []
    temp_paths = []
    predicted_blood_group = None
    
    try:
        data = pyjson.loads(json_data)
        fingerprints_meta = data.get('fingerprints', [])
        
        if files:
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
                    
        return JsonResponse({"success": True, "results": results, "predicted_blood_group": predicted_blood_group})
        
    except Exception as e:
        return JsonResponse({"success": False, "error": f"Blood group identification failed: {e}"}, status=500)
    finally:
        for p in temp_paths:
            try:
                if os.path.exists(p):
                    os.unlink(p)
            except Exception:
                pass

@api.post("/process-fingerprint/")
def process_fingerprint(request):
    """
    Process fingerprint data and participant information from scanner API.
    This endpoint handles the complete data processing pipeline.
    """
    try:
        import json as pyjson
        import base64
        import tempfile
        import os
        
        logger.info("Processing fingerprint data from scanner")
        
        # Get the JSON data from request
        data = request.body
        if isinstance(data, bytes):
            data = data.decode('utf-8')
        
        parsed_data = pyjson.loads(data)
        
        # Extract participant data, fingerprint data, and frontend callback URL
        participant_data = parsed_data.get('participant_data', {})
        fingerprint_data = parsed_data.get('fingerprint_data', {})
        finger_name = fingerprint_data.get('finger_name', parsed_data.get('finger_name', 'unknown'))
        
        # Check for frontend callback URL in both header and body
        frontend_callback_url = (
            request.headers.get('X-Frontend-Callback-URL') or 
            parsed_data.get('frontend_callback_url')
        )
        
        logger.info(f"Processing data for finger: {finger_name}")
        logger.info(f"Participant ID: {participant_data.get('participant_id', 'N/A')}")
        logger.info(f"Fingerprint data keys: {list(fingerprint_data.keys())}")
        logger.info(f"Has fingerprint image: {'image' in fingerprint_data}")
        
        if frontend_callback_url:
            logger.info(f"Frontend callback URL provided: {frontend_callback_url}")
        else:
            logger.info("No frontend callback URL provided - using traditional response")
        
        # Initialize response
        response_data = {
            "success": True,
            "finger_name": finger_name,
            "participant_id": participant_data.get('participant_id'),
            "processing_results": {}
        }
        
        # Helper function to safely convert values (same as in submit endpoint)
        def safe_convert(value, target_type, fallback=None):
            try:
                if value is None or value == "":
                    return fallback
                if target_type == int:
                    return int(float(str(value)))
                elif target_type == float:
                    return float(str(value))
                elif target_type == str:
                    return str(value)
                else:
                    return value
            except (ValueError, TypeError):
                return fallback
        
        def str_to_bool(value):
            if isinstance(value, str):
                return value.lower() in ('true', '1', 'yes')
            return value
        
        # Process participant data if provided
        if participant_data:
            try:
                # Save participant to database if consent is given
                consent = participant_data.get('consent', False)
                if consent:
                    participant = Participant.objects.create(
                        age=safe_convert(participant_data.get('age'), int),
                        height=safe_convert(participant_data.get('height'), float),
                        weight=safe_convert(participant_data.get('weight'), float),
                        gender=participant_data.get('gender', ''),
                        blood_type=participant_data.get('blood_type', ''),
                        willing_to_donate=participant_data.get('willing_to_donate', False),
                        sleep_hours=safe_convert(participant_data.get('sleep_hours'), int),
                        had_alcohol_last_24h=str_to_bool(participant_data.get('had_alcohol_last_24h')),
                        ate_before_donation=str_to_bool(participant_data.get('ate_before_donation')),
                        ate_fatty_food=str_to_bool(participant_data.get('ate_fatty_food')),
                        recent_tattoo_or_piercing=str_to_bool(participant_data.get('recent_tattoo_or_piercing')),
                        has_chronic_condition=str_to_bool(participant_data.get('has_chronic_condition')),
                        condition_controlled=str_to_bool(participant_data.get('condition_controlled')),
                        last_donation_date=participant_data.get('last_donation_date'),
                        consent=True
                    )
                    
                    response_data["participant_saved"] = True
                    response_data["participant_id"] = participant.id
                    
                    # Run diabetes prediction
                    try:
                        predictor = DiabetesPredictor()
                        prediction_result = predictor.predict_diabetes_risk(participant)
                        
                        if not prediction_result.get('error'):
                            # Save diabetes result
                            result = Result.objects.create(
                                participant=participant,
                                diabetes_risk=prediction_result['risk'],
                                confidence_score=prediction_result['confidence']
                            )
                            
                            response_data["processing_results"]["diabetes_prediction"] = {
                                "risk": prediction_result['risk'],
                                "confidence": prediction_result['confidence'],
                                "result_id": result.id
                            }
                        else:
                            response_data["processing_results"]["diabetes_prediction"] = {
                                "error": prediction_result['error']
                            }
                    except Exception as e:
                        logger.error(f"Diabetes prediction failed: {e}")
                        response_data["processing_results"]["diabetes_prediction"] = {
                            "error": f"Diabetes prediction failed: {str(e)}"
                        }
                else:
                    response_data["participant_saved"] = False
                    response_data["message"] = "Participant data not saved due to missing consent"
                    
            except Exception as e:
                logger.error(f"Failed to save participant data: {e}")
                response_data["participant_saved"] = False
                response_data["participant_error"] = str(e)
        
        # Process fingerprint data if provided
        if fingerprint_data and BLOOD_GROUP_AVAILABLE:
            try:
                # Get base64 image data
                image_data = fingerprint_data.get('image')
                if image_data:
                    # Decode base64 image
                    image_bytes = base64.b64decode(image_data)
                    
                    # Create temporary file
                    temp_fd, temp_path = tempfile.mkstemp(suffix='.png')
                    os.close(temp_fd)
                    
                    try:
                        # Write image to temporary file
                        with open(temp_path, 'wb') as f:
                            f.write(image_bytes)
                        
                        # Run blood group classification
                        blood_group_result = classify_blood_group_from_multiple([temp_path])
                        
                        response_data["processing_results"]["blood_group_classification"] = {
                            "predicted_blood_group": blood_group_result['predicted_blood_group'],
                            "confidence": blood_group_result['confidence'],
                            "all_probabilities": blood_group_result.get('all_probabilities')
                        }
                        
                    finally:
                        # Clean up temporary file
                        if os.path.exists(temp_path):
                            os.unlink(temp_path)
                            
            except Exception as e:
                logger.error(f"Blood group classification failed: {e}")
                response_data["processing_results"]["blood_group_classification"] = {
                    "error": f"Blood group classification failed: {str(e)}"
                }
        elif fingerprint_data and not BLOOD_GROUP_AVAILABLE:
            response_data["processing_results"]["blood_group_classification"] = {
                "error": "Blood group classification service is not available in this deployment"
            }
        
        # Save fingerprint data if participant was saved and fingerprint exists
        if response_data.get("participant_saved") and fingerprint_data:
            try:
                participant = Participant.objects.get(id=response_data["participant_id"])
                fingerprint = Fingerprint.objects.create(
                    participant=participant,
                    finger_name=finger_name,
                    image_data=fingerprint_data.get('image', ''),
                    upload_path=f"fingerprint_images/participant_{participant.id}/"
                )
                response_data["fingerprint_saved"] = True
                response_data["fingerprint_id"] = fingerprint.id
                
            except Exception as e:
                logger.error(f"Failed to save fingerprint: {e}")
                response_data["fingerprint_saved"] = False
                response_data["fingerprint_error"] = str(e)
        
        logger.info("Fingerprint processing completed successfully")
        
        # Send response directly to frontend if callback URL is provided
        if frontend_callback_url:
            try:
                import requests
                logger.info(f"Sending response directly to frontend: {frontend_callback_url}")
                
                # Send the complete response to frontend
                frontend_response = requests.post(
                    frontend_callback_url,
                    json=response_data,
                    headers={'Content-Type': 'application/json'},
                    timeout=10
                )
                
                if frontend_response.status_code == 200:
                    logger.info("✅ Successfully sent response to frontend")
                    # Return simple acknowledgment to scanner
                    return JsonResponse({
                        "success": True,
                        "message": "Processing completed and sent to frontend",
                        "frontend_delivery": "success"
                    })
                else:
                    logger.warning(f"Failed to send to frontend: {frontend_response.status_code}")
                    # Return full response to scanner as fallback
                    return JsonResponse(response_data)
                    
            except Exception as e:
                logger.error(f"Failed to send response to frontend: {e}")
                # Return full response to scanner as fallback
                return JsonResponse(response_data)
        else:
            # No frontend callback, return response to scanner as before
            return JsonResponse(response_data)
        
    except Exception as e:
        logger.error(f"Process fingerprint endpoint failed: {e}")
        return JsonResponse({
            "success": False,
            "error": f"Processing failed: {str(e)}"
        }, status=500)