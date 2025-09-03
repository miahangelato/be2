"""
Encryption utilities for securing sensitive data
"""
import os
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from django.conf import settings
import json

class DataEncryption:
    def __init__(self):
        self.key = self._get_or_create_key()
        self.cipher_suite = Fernet(self.key)
    
    def _get_or_create_key(self):
        """Get existing key or create a new one"""
        key_file = os.path.join(settings.BASE_DIR, 'encryption_key.key')
        
        if os.path.exists(key_file):
            with open(key_file, 'rb') as f:
                return f.read()
        else:
            # Generate a new key
            key = Fernet.generate_key()
            with open(key_file, 'wb') as f:
                f.write(key)
            return key
    
    def encrypt_string(self, text: str) -> str:
        """Encrypt a string and return base64 encoded result"""
        if not text:
            return text
        encrypted_data = self.cipher_suite.encrypt(text.encode())
        return base64.b64encode(encrypted_data).decode()
    
    def decrypt_string(self, encrypted_text: str) -> str:
        """Decrypt a base64 encoded encrypted string"""
        if not encrypted_text:
            return encrypted_text
        try:
            encrypted_data = base64.b64decode(encrypted_text.encode())
            decrypted_data = self.cipher_suite.decrypt(encrypted_data)
            return decrypted_data.decode()
        except Exception as e:
            raise ValueError(f"Decryption failed: {str(e)}")
    
    def encrypt_dict(self, data: dict) -> str:
        """Encrypt a dictionary by converting to JSON and encrypting"""
        json_string = json.dumps(data, default=str)
        return self.encrypt_string(json_string)
    
    def decrypt_dict(self, encrypted_json: str) -> dict:
        """Decrypt and parse JSON back to dictionary"""
        json_string = self.decrypt_string(encrypted_json)
        return json.loads(json_string)
    
    def encrypt_participant_data(self, participant_data: dict) -> dict:
        """Encrypt sensitive participant data fields"""
        sensitive_fields = [
            'age', 'height', 'weight', 'blood_type', 'gender',
            'sleep_hours', 'had_alcohol_last_24h', 'ate_before_donation',
            'ate_fatty_food', 'recent_tattoo_or_piercing', 'has_chronic_condition',
            'condition_controlled', 'last_donation_date'
        ]
        
        encrypted_data = {}
        for key, value in participant_data.items():
            if key in sensitive_fields and value is not None:
                encrypted_data[key] = self.encrypt_string(str(value))
            else:
                encrypted_data[key] = value
        
        return encrypted_data
    
    def decrypt_participant_data(self, encrypted_data: dict) -> dict:
        """Decrypt participant data fields"""
        sensitive_fields = [
            'age', 'height', 'weight', 'blood_type', 'gender',
            'sleep_hours', 'had_alcohol_last_24h', 'ate_before_donation',
            'ate_fatty_food', 'recent_tattoo_or_piercing', 'has_chronic_condition',
            'condition_controlled', 'last_donation_date'
        ]
        
        decrypted_data = {}
        for key, value in encrypted_data.items():
            if key in sensitive_fields and value is not None:
                try:
                    decrypted_value = self.decrypt_string(value)
                    # Convert back to appropriate type
                    if key in ['age', 'sleep_hours']:
                        decrypted_data[key] = int(decrypted_value) if decrypted_value != 'None' else None
                    elif key in ['height', 'weight']:
                        decrypted_data[key] = float(decrypted_value) if decrypted_value != 'None' else None
                    elif key in ['had_alcohol_last_24h', 'ate_before_donation', 'ate_fatty_food', 
                               'recent_tattoo_or_piercing', 'has_chronic_condition', 'condition_controlled']:
                        decrypted_data[key] = decrypted_value.lower() == 'true' if decrypted_value != 'None' else None
                    else:
                        decrypted_data[key] = decrypted_value if decrypted_value != 'None' else None
                except Exception as e:
                    print(f"Failed to decrypt field {key}: {e}")
                    decrypted_data[key] = value
            else:
                decrypted_data[key] = value
        
        return decrypted_data

# Global instance
encryption_service = DataEncryption()
