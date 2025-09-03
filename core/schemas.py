from ninja import NinjaAPI, Schema, ModelSchema
from typing import List
from .models import Participant, Fingerprint
class ParticipantSchema(ModelSchema):
    class Config:
        model = Participant
        model_fields = [
            "id", "age", "weight", "gender", "blood_type", "consent", "willing_to_donate",
            "sleep_hours", "had_alcohol_last_24h", "ate_before_donation", "ate_fatty_food",
            "recent_tattoo_or_piercing", "has_chronic_condition", "condition_controlled",
            "last_donation_date"
        ]


class ParticipantCreateSchema(Schema):
    age: int
    weight: float
    gender: str
    blood_type: str = "unknown"
    consent: bool
    willing_to_donate: bool = False
    sleep_hours: int = None
    had_alcohol_last_24h: bool = False
    ate_before_donation: bool = False
    ate_fatty_food: bool = False
    recent_tattoo_or_piercing: bool = False
    has_chronic_condition: bool = False
    condition_controlled: bool = True
    last_donation_date: str = None


class FingerprintSchema(ModelSchema):
    class Config:
        model = Fingerprint
        model_fields = ["participant", "finger", "image", "pattern"]


