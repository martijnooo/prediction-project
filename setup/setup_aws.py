import boto3
import argparse
import sys
from botocore.exceptions import ClientError

def create_s3_bucket(bucket_name, region=None):
    """Create an S3 bucket in a specified region"""
    try:
        if region is None:
            s3_client = boto3.client('s3')
            s3_client.create_bucket(Bucket=bucket_name)
        else:
            s3_client = boto3.client('s3', region_name=region)
            location = {'LocationConstraint': region}
            s3_client.create_bucket(Bucket=bucket_name, CreateBucketConfiguration=location)
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == 'BucketAlreadyOwnedByYou':
            print(f"Bucket {bucket_name} already exists and is owned by you.")
            return True
        print(f"Error creating bucket {bucket_name}: {e}")
        return False
    return True

def create_folders(bucket_name):
    """Create "folders" in S3 by creating 0-byte objects with trailing slash"""
    s3_client = boto3.client('s3')
    folders = ['raw/', 'processed/', 'models/', 'predictions/', 'monitoring/']
    
    print(f"Creating folder structure in {bucket_name}...")
    for folder in folders:
        try:
            s3_client.put_object(Bucket=bucket_name, Key=folder)
            print(f" - Created {folder}")
        except ClientError as e:
            print(f"Error creating folder {folder}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Setup AWS Resources")
    parser.add_argument('--bucket', type=str, required=True, help='Name of the S3 bucket to create')
    parser.add_argument('--region', type=str, default='eu-west-1', help='AWS Region')
    
    args = parser.parse_args()
    
    print(f"Setting up AWS resources for project...")
    print(f"Target Bucket: {args.bucket}")
    print(f"Region: {args.region}")
    
    if create_s3_bucket(args.bucket, args.region):
        print(f"Successfully created bucket: {args.bucket}")
        create_folders(args.bucket)
        print("Setup complete.")
    else:
        print("Setup failed.")
        sys.exit(1)
