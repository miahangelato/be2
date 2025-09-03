"""
Secure S3 Storage Backend with encryption and access control
Application-level security that doesn't require bucket permissions
"""
from django.conf import settings
from storages.backends.s3boto3 import S3Boto3Storage
import boto3
from botocore.config import Config
import logging
import hashlib
import time

logger = logging.getLogger(__name__)

class SecureS3Storage(S3Boto3Storage):
    """
    Secure S3 storage backend with enhanced security features
    Works without bucket-level permissions by implementing app-level security
    """
    
    def __init__(self, **settings_override):
        super().__init__(**settings_override)
        
        # Configure boto3 with security settings
        self.config = Config(
            use_ssl=True,
            signature_version='s3v4',
            retries={'max_attempts': 3, 'mode': 'adaptive'},
            max_pool_connections=50
        )
    
    def _save(self, name, content):
        """
        Override save to add application-level security
        """
        # Force server-side encryption at object level
        extra_args = getattr(self, 'object_parameters', {}).copy()
        
        # Add security headers and encryption
        extra_args.update({
            'ServerSideEncryption': 'AES256',  # Force encryption
            'ACL': 'private',  # Force private access
            'ContentDisposition': 'attachment',  # Force download, prevent inline viewing
            'Metadata': {
                'Content-Security-Policy': "default-src 'none'",
                'X-Content-Type-Options': 'nosniff',
                'X-Frame-Options': 'DENY',
                'upload-timestamp': str(int(time.time())),
                'app-source': 'fingerprint-analysis',
                'security-level': 'high'
            }
        })
        
        # Create a secure filename hash for audit trail
        file_hash = hashlib.sha256(f"{name}{time.time()}".encode()).hexdigest()[:8]
        logger.info(f"Uploading secure file: {name} (hash: {file_hash})")
        
        # Set the extra args for this specific upload
        original_object_parameters = getattr(self, 'object_parameters', {})
        self.object_parameters = extra_args
        
        try:
            result = super()._save(name, content)
            logger.info(f"Successfully uploaded secure file: {name}")
            return result
        except Exception as e:
            logger.error(f"Failed to upload secure file {name}: {str(e)}")
            raise
        finally:
            # Restore original object parameters
            self.object_parameters = original_object_parameters
    
    def url(self, name, parameters=None, expire=None, http_method=None):
        """
        Generate secure presigned URLs with short expiration
        """
        if expire is None:
            expire = getattr(settings, 'AWS_QUERYSTRING_EXPIRE', 1800)  # 30 minutes default
        
        # Always use HTTPS for URLs
        try:
            url = super().url(name, parameters, expire, http_method)
            if url and url.startswith('http://'):
                url = url.replace('http://', 'https://', 1)
            
            # Log URL generation for audit
            logger.info(f"Generated secure URL for: {name} (expires in {expire}s)")
            return url
        except Exception as e:
            logger.error(f"Failed to generate URL for {name}: {str(e)}")
            raise
    
    def exists(self, name):
        """
        Check if file exists with proper error handling
        """
        try:
            return super().exists(name)
        except Exception as e:
            logger.warning(f"Error checking file existence for {name}: {str(e)}")
            return False
    
    def delete(self, name):
        """
        Secure file deletion with logging
        """
        try:
            logger.info(f"Attempting to delete file: {name}")
            result = super().delete(name)
            logger.info(f"Successfully deleted file: {name}")
            return result
        except Exception as e:
            logger.error(f"Failed to delete file {name}: {str(e)}")
            raise

class SecureMediaStorage(SecureS3Storage):
    """
    Secure storage for media files (fingerprints, etc.)
    """
    location = 'media'
    default_acl = 'private'
    file_overwrite = False
    custom_domain = False  # Don't use CDN for sensitive files
    querystring_auth = True  # Always use signed URLs
    
    def __init__(self, **settings_override):
        # Override any settings to ensure security
        security_overrides = {
            'default_acl': 'private',
            'querystring_auth': True,
            'querystring_expire': 1800,  # 30 minutes
            'object_parameters': {
                'ServerSideEncryption': 'AES256',
                'ACL': 'private'
            }
        }
        security_overrides.update(settings_override)
        super().__init__(**security_overrides)

class SecureStaticStorage(SecureS3Storage):
    """
    Storage for static files (CSS, JS, etc.)
    """
    location = 'static'
    default_acl = 'public-read'
    querystring_auth = False  # Static files don't need signed URLs
    
    def _save(self, name, content):
        """
        Static files can have different security settings
        """
        # Less restrictive for static files but still secure
        extra_args = getattr(self, 'object_parameters', {}).copy()
        extra_args.update({
            'CacheControl': 'max-age=86400',  # 24 hour cache
            'ACL': 'public-read',
            'Metadata': {
                'file-type': 'static',
                'cache-policy': 'public'
            }
        })
        
        original_object_parameters = getattr(self, 'object_parameters', {})
        self.object_parameters = extra_args
        
        try:
            return super(SecureS3Storage, self)._save(name, content)  # Skip encryption for static files
        finally:
            self.object_parameters = original_object_parameters
