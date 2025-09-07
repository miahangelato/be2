# models.py
from django.db import models
import uuid

def fingerprint_upload_path(instance, filename):
    """
    Generate upload path: fingerprint_images/participant_ID/participant_ID_age_weight_height_bloodtype_gender_diabeticstatus_fingername.png
    """
    participant = instance.participant
    participant_id = participant.id
    finger_name = instance.finger
    
    # Extract file extension
    file_extension = filename.split('.')[-1] if '.' in filename else 'png'
    
    # Get participant data with safe values for filenames
    age = participant.age if participant.age else "Unknown"
    weight = f"{participant.weight}kg" if participant.weight else "Unknown"
    height = participant.height if participant.height else "Unknown"
    gender = participant.gender if participant.gender else "Unknown"
    
    # Get blood type - use result if available, otherwise participant's input, otherwise Unknown
    blood_type = "Unknown"
    if participant.blood_type and participant.blood_type != "unknown":
        blood_type = participant.blood_type
    else:
        # Try to get blood type from results
        try:
            latest_result = participant.results.latest('created_at')
            if latest_result.blood_group:
                blood_type = latest_result.blood_group
        except:
            blood_type = "Unknown"
    
    # Get diabetes status from results
    diabetes_status = "Unknown"
    try:
        latest_result = participant.results.latest('created_at')
        if latest_result.diabetes_risk:
            diabetes_status = latest_result.diabetes_risk
    except:
        diabetes_status = "Unknown"
    
    # Clean values to be filesystem-safe (remove special characters)
    def clean_filename_part(value):
        """Remove characters that could cause filesystem issues"""
        import re
        # Convert to string and replace problematic characters
        safe_value = str(value).replace('/', '_').replace('\\', '_').replace(':', '_')
        safe_value = re.sub(r'[<>:"/\\|?*]', '_', safe_value)
        return safe_value
    
    age = clean_filename_part(age)
    weight = clean_filename_part(weight)
    height = clean_filename_part(height)
    blood_type = clean_filename_part(blood_type)
    gender = clean_filename_part(gender)
    diabetes_status = clean_filename_part(diabetes_status)
    finger_name = clean_filename_part(finger_name)
    
    # Create filename: participant_ID_age_weight_height_bloodtype_gender_diabeticstatus_fingername.extension
    new_filename = f"participant_{participant_id}_{age}_{weight}_{height}_{blood_type}_{gender}_{diabetes_status}_{finger_name}.{file_extension}"
    
    return f"fingerprint_images/participant_{participant_id}/{new_filename}"

class Participant(models.Model):
    age = models.IntegerField()
    gender = models.CharField(
        max_length=10, 
        choices=[("male", "Male"), ("female", "Female")]
    )
    blood_type = models.CharField(
        max_length=None, 
        choices=[("A", "A"), ("B", "B"), ("AB", "AB"), ("O", "O"), ("unknown", "Unknown")],
        default="unknown"
    )
    weight = models.FloatField(null=True, blank=True)
    height = models.FloatField(null=True, blank=True)
    
    willing_to_donate = models.BooleanField(default=False)

    # Health & consent
    has_chronic_condition = models.BooleanField(null=True, default=False)
    condition_controlled = models.BooleanField(null=True, default=False)  # ✅ new
    consent = models.BooleanField(default=False)

    # Lifestyle / donation restrictions
    recent_tattoo_or_piercing = models.BooleanField(null=True, default=False)
    ate_fatty_food = models.BooleanField(null=True,default=False)
    ate_before_donation = models.BooleanField(null=True,default=True)
    had_alcohol_last_24h = models.BooleanField(null=True, default=False)
    sleep_hours = models.IntegerField(null=True, blank=True, default=None)
    last_donation_date = models.DateField(null=True, blank=True)

    def check_prc_eligibility(self):
        """Apply Philippine Red Cross blood donation criteria"""
        reasons = []
        eligible = True

        # Age
        if not (16 <= self.age <= 65):
            eligible = False
            reasons.append("Age not within 16-65 years")

        # Weight
        if self.weight and self.weight < 50:
            eligible = False
            reasons.append("Weight below 50kg")

        # Chronic condition
        if self.has_chronic_condition and not self.condition_controlled:
            eligible = False
            reasons.append("Uncontrolled chronic condition")

        # Lifestyle
        if self.recent_tattoo_or_piercing:
            eligible = False
            reasons.append("Recent tattoo/piercing")

        if self.ate_fatty_food:
            eligible = False
            reasons.append("Ate fatty food")

        if self.had_alcohol_last_24h:
            eligible = False
            reasons.append("Had alcohol in last 24 hours")

        if self.sleep_hours < 6:
            eligible = False
            reasons.append("Less than 6 hours sleep")

        # Last donation
        if self.last_donation_date:
            from datetime import timedelta, date
            if (date.today() - self.last_donation_date) < timedelta(days=90):
                eligible = False
                reasons.append("Last donation less than 3 months ago")

        # Consent
        if not self.consented:
            eligible = False
            reasons.append("No consent given")

        return eligible, reasons



class Fingerprint(models.Model):
    participant = models.ForeignKey(Participant, on_delete=models.CASCADE, related_name="fingerprints")
    finger = models.CharField(max_length=20)  # e.g., "left_thumb"
    image = models.ImageField(upload_to=fingerprint_upload_path)
    pattern = models.CharField(max_length=20, null=True, blank=True)  # Arc, Loop, Whorl

class Result(models.Model):
    participant = models.ForeignKey(Participant, on_delete=models.CASCADE, related_name="results")
    diabetes_risk = models.CharField(max_length=20)  # "low", "medium", "high"
    confidence_score = models.FloatField(null=True, blank=True)

    blood_group = models.CharField(max_length=3, choices=[("A+", "A+"), ("B+", "B+"), ("AB+", "AB+"), ("O+", "O+")])
    blood_group_confidence_score = models.FloatField(null=True, blank=True)
    
    created_at = models.DateTimeField(auto_now_add=True)


class Session(models.Model):
    session_id = models.UUIDField(default=uuid.uuid4, unique=True, editable=False)
    participant_data = models.JSONField()  # Store participant data
    diabetes_result = models.JSONField(null=True, blank=True)  # Store diabetes prediction result
    blood_group_result = models.JSONField(null=True, blank=True)  # Store blood group prediction result
    fingerprints = models.JSONField(null=True, blank=True)  # Store fingerprint data
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    expires_at = models.DateTimeField()  # Session expiration
    
    def __str__(self):
        return f"Session {self.session_id}"