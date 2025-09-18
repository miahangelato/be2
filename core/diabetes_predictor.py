

import pandas as pd
import numpy as np
import pickle
import os
from django.conf import settings
from .models import Participant, Fingerprint

class DiabetesPredictor:
    def __init__(self):
        base = settings.BASE_DIR
        self.model_paths = {
            'A': os.path.join(base, "core", "diabetes_risk_model.pkl"),
            'B': os.path.join(base, "core", "diabetes_risk_model_B.pkl"),
        }
        self.cols_paths = {
            'A': os.path.join(base, "core", "diabetes_risk_model_columns.pkl"),
            'B': os.path.join(base, "core", "diabetes_risk_model_columns_B.pkl"),
        }
        self.models = {}
        self.model_columns = {}
        self.load_models()

    def load_models(self):
        for key in self.model_paths:
            # Load model
            model_path = self.model_paths[key]
            try:
                if os.path.exists(model_path):
                    with open(model_path, 'rb') as f:
                        self.models[key] = pickle.load(f)
                else:
                    self.models[key] = None
            except Exception as e:
                self.models[key] = None
            # Load columns
            cols_path = self.cols_paths[key]
            try:
                if os.path.exists(cols_path):
                    with open(cols_path, 'rb') as f:
                        self.model_columns[key] = pickle.load(f)
                else:
                    self.model_columns[key] = None
            except Exception as e:
                self.model_columns[key] = None

    def prepare_input_df(self, participant_data, model_key):
        feature_order = [
            'age', 'weight', 'height', 'blood_type', 'gender',
            'left_thumb', 'left_index', 'left_middle', 'left_ring', 'left_pinky',
            'right_thumb', 'right_index', 'right_middle', 'right_ring', 'right_pinky'
        ]
        row = [participant_data.get(f) for f in feature_order]
        df = pd.DataFrame([row], columns=feature_order)
        for col in ['blood_type','gender','left_thumb','left_index','left_middle','left_ring','left_pinky',
                    'right_thumb','right_index','right_middle','right_ring','right_pinky']:
            df[col] = df[col].astype(str).str.lower()
        model_cols = self.model_columns[model_key]
        df = pd.get_dummies(df, columns=['blood_type','gender','left_thumb','left_index','left_middle','left_ring','left_pinky',
                    'right_thumb','right_index','right_middle','right_ring','right_pinky'], drop_first=True)
        for col in model_cols:
            if col not in df.columns:
                df[col] = 0
        df = df[model_cols]
        return df
    
    def prepare_participant_data(self, participant):
        """Convert participant data to match your simple dataset format"""
        # Get all fingerprints for this participant
        fingerprints = Fingerprint.objects.filter(participant=participant)
        
        # Create dictionary with all finger positions
        finger_data = {
            'left_thumb': None,
            'left_index': None, 
            'left_middle': None,
            'left_ring': None,
            'left_pinky': None,
            'right_thumb': None,
            'right_index': None,
            'right_middle': None,
            'right_ring': None,
            'right_pinky': None
        }
        
        # Fill in the fingerprint patterns
        for fp in fingerprints:
            if fp.finger in finger_data:
                finger_data[fp.finger] = fp.pattern
        
        # Create participant row exactly matching your dataset format
        participant_data = {
            'age': participant.age,
            'weight': participant.weight,
            'height': participant.height,
            'blood_type': participant.blood_type if participant.blood_type != 'unknown' else 'UNKNOWN',
            'gender': participant.gender,
            'left_thumb': finger_data['left_thumb'],
            'left_index': finger_data['left_index'],
            'left_middle': finger_data['left_middle'],
            'left_ring': finger_data['left_ring'],
            'left_pinky': finger_data['left_pinky'],
            'right_thumb': finger_data['right_thumb'],
            'right_index': finger_data['right_index'],
            'right_middle': finger_data['right_middle'],
            'right_ring': finger_data['right_ring'],
            'right_pinky': finger_data['right_pinky']
        }
        
        return participant_data
    
    def predict_diabetes_risk(self, participant, model_key='A'):
        """Predict diabetes risk using the selected model (A or B)."""
        try:
            model = self.models.get(model_key)
            model_cols = self.model_columns.get(model_key)
            if model is None or model_cols is None:
                return {
                    'risk': 'unknown',
                    'confidence': 0.0,
                    'error': f'Model {model_key} not loaded',
                }
            participant_data = self.prepare_participant_data(participant)
            df = self.prepare_input_df(participant_data, model_key)
            pred = model.predict(df)[0]
            if str(pred).lower() in ['diabetic', '1', 'at risk', 'risk', 'positive']:
                risk = 'DIABETIC'
            else:
                risk = 'HEALTHY'
            return {
                'risk': risk,
                'confidence': 1.0,
                'model_used': model_key
            }
        except Exception as e:
            return {
                'risk': 'unknown',
                'confidence': 0.0,
                'error': f'Prediction failed: {str(e)}',
                'model_used': model_key
            }
    