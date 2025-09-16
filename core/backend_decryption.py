"""
Backend encryption utilities that match frontend encryption
"""
import base64
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import hashlib

class BackendDecryption:
    def __init__(self):
        # Use the same key as frontend for compatibility
        self.secret_key = "my-secret-key-2024"
        print(f"[🔑 KEY LOADED] Using matching frontend key")
    
    def decrypt_crypto_js(self, encrypted_data):
        """
        Decrypt CryptoJS AES encrypted data
        CryptoJS format: "Salted__" + salt(8 bytes) + encrypted_data
        """
        try:
            print(f"[🔓 DECRYPT ATTEMPT] {encrypted_data[:30]}...")
            
            # Decode from base64
            encrypted_bytes = base64.b64decode(encrypted_data)
            print(f"[📝 DEBUG] Decoded length: {len(encrypted_bytes)} bytes")
            
            # Check if it starts with "Salted__"
            if len(encrypted_bytes) < 16:
                print(f"[❌ DECRYPT ERROR] Data too short: {len(encrypted_bytes)} bytes")
                return encrypted_data
                
            if not encrypted_bytes.startswith(b'Salted__'):
                print(f"[❌ DECRYPT ERROR] Not a valid CryptoJS format")
                return encrypted_data
            
            # Extract salt (bytes 8-16)
            salt = encrypted_bytes[8:16]
            ciphertext = encrypted_bytes[16:]
            print(f"[📝 DEBUG] Salt: {salt.hex()}, Ciphertext length: {len(ciphertext)}")
            
            # Derive key and IV using the same method as CryptoJS
            key_iv = self._derive_key_iv(self.secret_key, salt)
            key = key_iv[:32]  # 32 bytes for AES-256
            iv = key_iv[32:48]  # 16 bytes for IV
            
            # Decrypt
            cipher = Cipher(algorithms.AES(key), modes.CBC(iv))
            decryptor = cipher.decryptor()
            decrypted_padded = decryptor.update(ciphertext) + decryptor.finalize()
            
            # Remove PKCS7 padding
            if len(decrypted_padded) == 0:
                print(f"[❌ DECRYPT ERROR] Decrypted data is empty")
                return encrypted_data
                
            padding_length = decrypted_padded[-1]
            if padding_length > len(decrypted_padded) or padding_length == 0:
                print(f"[❌ DECRYPT ERROR] Invalid padding: {padding_length}")
                return encrypted_data
                
            decrypted = decrypted_padded[:-padding_length]
            
            result = decrypted.decode('utf-8')
            print(f"[✅ DECRYPT SUCCESS] {result}")
            return result
            
        except Exception as e:
            print(f"[❌ DECRYPT ERROR] {str(e)}")
            return encrypted_data
    
    def _derive_key_iv(self, password, salt):
        """
        Derive key and IV the same way CryptoJS does
        """
        password_bytes = password.encode('utf-8')
        derived = b''
        
        # CryptoJS uses MD5 for key derivation
        while len(derived) < 48:  # 32 bytes key + 16 bytes IV
            hash_input = derived[-16:] + password_bytes + salt if derived else password_bytes + salt
            hash_obj = hashlib.md5(hash_input)
            derived += hash_obj.digest()
            
        return derived
    
    def decrypt_form_data(self, form_data_dict):
        """
        Process form data and decrypt encrypted fields
        """
        print("[📨 BACKEND RECEIVED] Processing form data...")
        decrypted_data = {}
        
        # List of fields that might be encrypted
        encrypted_fields = ['age', 'height', 'weight', 'gender', 'blood_type', 'sleep_hours']
        
        for key, value in form_data_dict.items():
            if key in encrypted_fields and self._looks_encrypted(str(value)):
                print(f"[🔍 DETECTING] {key}: {str(value)[:30]}... (appears encrypted)")
                decrypted_value = self.decrypt_crypto_js(str(value))
                decrypted_data[key] = decrypted_value
                print(f"[🔓 DECRYPTED] {key}: {decrypted_value}")
            else:
                decrypted_data[key] = value
                print(f"[📝 PLAIN TEXT] {key}: {value}")
        
        return decrypted_data
    
    def _looks_encrypted(self, data):
        """
        Check if data looks like CryptoJS encrypted data
        """
        if not isinstance(data, str) or len(data) < 20:
            return False
        
        try:
            # Try to decode as base64
            decoded = base64.b64decode(data)
            # Check if it starts with "Salted__"
            return decoded.startswith(b'Salted__')
        except:
            return False

backend_decryption = BackendDecryption()
