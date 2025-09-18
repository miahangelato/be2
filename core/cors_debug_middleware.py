"""
Custom CORS Debug Middleware
Logs all request/response headers to help debug CORS issues
"""
import logging

logger = logging.getLogger(__name__)

class CORSDebugMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        # Log incoming request details
        origin = request.META.get('HTTP_ORIGIN', 'No Origin')
        method = request.method
        path = request.path
        
        print(f"🔍 CORS DEBUG - Incoming Request:")
        print(f"   Method: {method}")
        print(f"   Path: {path}")
        print(f"   Origin: {origin}")
        print(f"   Headers: {dict(request.META)}")
        
        # Process the request
        response = self.get_response(request)
        
        # Log outgoing response headers
        print(f"🔍 CORS DEBUG - Outgoing Response:")
        print(f"   Status: {response.status_code}")
        response_headers = dict(response.items()) if hasattr(response, 'items') else {}
        print(f"   Headers: {response_headers}")
        
        # Check if CORS headers are present
        cors_headers = {
            'Access-Control-Allow-Origin': response.get('Access-Control-Allow-Origin'),
            'Access-Control-Allow-Methods': response.get('Access-Control-Allow-Methods'),
            'Access-Control-Allow-Headers': response.get('Access-Control-Allow-Headers'),
            'Access-Control-Allow-Credentials': response.get('Access-Control-Allow-Credentials'),
        }
        
        print(f"🔍 CORS Headers in Response: {cors_headers}")
        
        # Check if this is a preflight request
        if method == 'OPTIONS':
            print(f"🔍 PREFLIGHT REQUEST detected for origin: {origin}")
            
        return response