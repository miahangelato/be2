#!/usr/bin/env python
"""
Test S3 access for model files
This script helps verify S3 permissions are correctly configured
"""

import requests
import sys

# Test URLs for your models
MODEL_URLS = [
    "https://team3thesis.s3.us-east-1.amazonaws.com/models/backend/core%5Cimproved_pattern_cnn_model.h5",
    "https://team3thesis.s3.us-east-1.amazonaws.com/models/backend/core%5Cbloodgroup_model_20250823-140933.h5",
    "https://team3thesis.s3.us-east-1.amazonaws.com/models/backend/core%5Cdiabetes_risk_model.pkl",
    "https://team3thesis.s3.us-east-1.amazonaws.com/models/backend/core%5Cdiabetes_risk_model_columns.pkl"
]

def test_s3_access():
    """Test if we can access S3 model files"""
    print("=== Testing S3 Model Access ===\n")
    
    all_accessible = True
    
    for i, url in enumerate(MODEL_URLS, 1):
        model_name = url.split('/')[-1]
        print(f"{i}. Testing: {model_name}")
        print(f"   URL: {url}")
        
        try:
            # Make a HEAD request to check if file exists and is accessible
            response = requests.head(url, timeout=10)
            
            if response.status_code == 200:
                content_length = response.headers.get('Content-Length', 'Unknown')
                content_type = response.headers.get('Content-Type', 'Unknown')
                
                print(f"   ✅ SUCCESS - Status: {response.status_code}")
                print(f"   📁 Size: {content_length} bytes")
                print(f"   📄 Type: {content_type}")
                
            elif response.status_code == 403:
                print(f"   ❌ FORBIDDEN - Status: {response.status_code}")
                print(f"   🔒 Access denied - Check bucket policy allows s3:GetObject")
                all_accessible = False
                
            elif response.status_code == 404:
                print(f"   ❌ NOT FOUND - Status: {response.status_code}")
                print(f"   📂 File doesn't exist at this location")
                all_accessible = False
                
            else:
                print(f"   ⚠️  UNEXPECTED - Status: {response.status_code}")
                print(f"   📝 Response: {response.reason}")
                all_accessible = False
                
        except requests.exceptions.RequestException as e:
            print(f"   ❌ ERROR - Network issue: {e}")
            all_accessible = False
            
        print()  # Empty line for readability
    
    print("=== Summary ===")
    if all_accessible:
        print("✅ ALL MODELS ARE ACCESSIBLE!")
        print("Your S3 bucket policy is correctly configured.")
    else:
        print("❌ SOME MODELS ARE NOT ACCESSIBLE!")
        print("\nTo fix this, update your S3 bucket policy to:")
        print("1. Go to AWS S3 Console → team3thesis bucket → Permissions")
        print("2. Edit Bucket Policy and replace with:")
        print('''
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": "*",
            "Action": [
                "s3:GetObject",
                "s3:PutObject"
            ],
            "Resource": "arn:aws:s3:::team3thesis/*"
        }
    ]
}
        ''')
        print("3. Save the policy")
        print("4. Wait a few minutes for changes to propagate")
    
    return all_accessible

def test_single_download():
    """Test downloading a small portion of a file"""
    print("\n=== Testing Partial Download ===")
    
    test_url = MODEL_URLS[0]  # Test with fingerprint model
    model_name = test_url.split('/')[-1]
    
    try:
        print(f"Testing partial download of: {model_name}")
        
        # Download only first 1KB to test
        headers = {'Range': 'bytes=0-1023'}
        response = requests.get(test_url, headers=headers, timeout=10)
        
        if response.status_code in [200, 206]:  # 206 = Partial Content
            print(f"✅ Partial download successful!")
            print(f"   Downloaded: {len(response.content)} bytes")
            print(f"   Content starts with: {response.content[:50]}...")
            return True
        else:
            print(f"❌ Partial download failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Download test failed: {e}")
        return False

if __name__ == "__main__":
    print("S3 Model Access Test Utility\n")
    
    # Test basic access
    access_ok = test_s3_access()
    
    # If basic access works, test downloading
    if access_ok:
        download_ok = test_single_download()
        if download_ok:
            print("\n🎉 All tests passed! Your models should download successfully.")
        else:
            print("\n⚠️ Access works but download failed. Check network connectivity.")
    else:
        print("\n🔧 Fix the bucket policy first, then run this test again.")
    
    sys.exit(0 if access_ok else 1)