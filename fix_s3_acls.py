#!/usr/bin/env python
"""
Fix S3 Model File ACLs
This script updates the ACL on your model files to make them publicly readable
"""

import boto3
import os
import sys
from botocore.exceptions import ClientError, NoCredentialsError

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# S3 configuration
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
AWS_STORAGE_BUCKET_NAME = os.getenv('AWS_STORAGE_BUCKET_NAME', 'team3thesis')
AWS_S3_REGION_NAME = os.getenv('AWS_S3_REGION_NAME', 'us-east-1')

# Model file paths in S3
MODEL_FILES = [
    "models/backend/core\\improved_pattern_cnn_model.h5",
    "models/backend/core\\bloodgroup_model_20250823-140933.h5",
    "models/backend/core\\diabetes_risk_model.pkl",
    "models/backend/core\\diabetes_risk_model_columns.pkl"
]

def check_credentials():
    """Check if AWS credentials are available"""
    if not AWS_ACCESS_KEY_ID or not AWS_SECRET_ACCESS_KEY:
        print("❌ AWS credentials not found!")
        print("Make sure these environment variables are set:")
        print("  - AWS_ACCESS_KEY_ID")
        print("  - AWS_SECRET_ACCESS_KEY")
        print("  - AWS_STORAGE_BUCKET_NAME")
        return False
    return True

def fix_model_acls():
    """Update ACL on model files to make them publicly readable"""
    
    if not check_credentials():
        return False
    
    try:
        # Create S3 client
        s3_client = boto3.client(
            's3',
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            region_name=AWS_S3_REGION_NAME
        )
        
        print(f"🔧 Fixing ACLs for model files in bucket: {AWS_STORAGE_BUCKET_NAME}")
        print("=" * 60)
        
        success_count = 0
        total_count = len(MODEL_FILES)
        
        for i, file_key in enumerate(MODEL_FILES, 1):
            file_name = file_key.split('/')[-1]
            print(f"\n{i}. Processing: {file_name}")
            print(f"   S3 Key: {file_key}")
            
            try:
                # First, check if the file exists
                try:
                    s3_client.head_object(Bucket=AWS_STORAGE_BUCKET_NAME, Key=file_key)
                    print("   ✅ File exists in S3")
                except ClientError as e:
                    if e.response['Error']['Code'] == '404':
                        print("   ❌ File not found in S3")
                        continue
                    else:
                        print(f"   ❌ Error checking file: {e}")
                        continue
                
                # Update the ACL to public-read
                s3_client.put_object_acl(
                    Bucket=AWS_STORAGE_BUCKET_NAME,
                    Key=file_key,
                    ACL='public-read'
                )
                
                print("   ✅ ACL updated to public-read")
                
                # Verify the change by checking if we can access it publicly
                public_url = f"https://{AWS_STORAGE_BUCKET_NAME}.s3.{AWS_S3_REGION_NAME}.amazonaws.com/{file_key}"
                print(f"   🔗 Public URL: {public_url}")
                
                success_count += 1
                
            except ClientError as e:
                error_code = e.response['Error']['Code']
                error_message = e.response['Error']['Message']
                print(f"   ❌ Failed to update ACL: {error_code} - {error_message}")
                
                if error_code == 'AccessDenied':
                    print("   💡 Your AWS credentials may not have s3:PutObjectAcl permission")
                elif error_code == 'NoSuchKey':
                    print("   💡 File doesn't exist at this location")
                
            except Exception as e:
                print(f"   ❌ Unexpected error: {e}")
        
        print("\n" + "=" * 60)
        print(f"📊 Summary: {success_count}/{total_count} files updated successfully")
        
        if success_count == total_count:
            print("🎉 All model files are now publicly accessible!")
            print("\nNext steps:")
            print("1. Run: python test_s3_access.py")
            print("2. If all tests pass, redeploy on Railway")
        elif success_count > 0:
            print("⚠️ Some files were updated, but others failed")
            print("Check the errors above and fix any permission issues")
        else:
            print("❌ No files were updated successfully")
            print("Check your AWS credentials and permissions")
        
        return success_count == total_count
        
    except NoCredentialsError:
        print("❌ AWS credentials not found or invalid!")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def list_bucket_contents():
    """List contents of the models directory to verify file locations"""
    
    if not check_credentials():
        return
    
    try:
        s3_client = boto3.client(
            's3',
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            region_name=AWS_S3_REGION_NAME
        )
        
        print(f"\n📁 Listing contents of bucket: {AWS_STORAGE_BUCKET_NAME}")
        print("Looking for files in 'models/' directory...")
        
        response = s3_client.list_objects_v2(
            Bucket=AWS_STORAGE_BUCKET_NAME,
            Prefix='models/',
            MaxKeys=100
        )
        
        if 'Contents' in response:
            print(f"\nFound {len(response['Contents'])} files:")
            for obj in response['Contents']:
                size_mb = obj['Size'] / (1024 * 1024)
                print(f"  📄 {obj['Key']} ({size_mb:.2f} MB)")
        else:
            print("   No files found in 'models/' directory")
            
            # Check if files might be in a different location
            print("\n🔍 Checking entire bucket for model files...")
            response = s3_client.list_objects_v2(
                Bucket=AWS_STORAGE_BUCKET_NAME,
                MaxKeys=1000
            )
            
            if 'Contents' in response:
                model_files = [obj for obj in response['Contents'] if any(ext in obj['Key'] for ext in ['.h5', '.pkl'])]
                if model_files:
                    print(f"Found {len(model_files)} model-like files:")
                    for obj in model_files:
                        size_mb = obj['Size'] / (1024 * 1024)
                        print(f"  📄 {obj['Key']} ({size_mb:.2f} MB)")
                else:
                    print("   No model files found in the entire bucket")
            
    except Exception as e:
        print(f"❌ Error listing bucket contents: {e}")

if __name__ == "__main__":
    print("S3 Model File ACL Fixer\n")
    
    # First, list bucket contents to see what's there
    list_bucket_contents()
    
    # Ask user if they want to proceed
    print("\n" + "=" * 60)
    response = input("Do you want to update ACLs on the model files? (y/n): ").lower().strip()
    
    if response == 'y':
        success = fix_model_acls()
        sys.exit(0 if success else 1)
    else:
        print("Operation cancelled.")
        sys.exit(0)