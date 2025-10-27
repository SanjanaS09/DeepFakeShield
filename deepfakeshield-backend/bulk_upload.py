import os
import boto3
from botocore.exceptions import NoCredentialsError
from dotenv import load_dotenv

# This line loads the environment variables from the .env file
load_dotenv() 

# --- PLEASE CONFIGURE THESE VALUES ---
# The local directory containing your dataset files
LOCAL_DATASET_PATH = "C:\Users\Sanjana\DeepFakeShield\deepfakeshield-backend\dataset" 
# Your S3 bucket name
BUCKET_NAME = "sanjana-deepfake-dataset-unique"
# --- END OF CONFIGURATION ---

# Initialize the S3 client
# It will automatically use the credentials from your environment variables
s3_client = boto3.client('s3')

def upload_directory(path, bucketname):
    print(f"Starting upload of directory '{path}' to bucket '{bucketname}'...")
    try:
        for root, dirs, files in os.walk(path):
            for file in files:
                local_path = os.path.join(root, file)
                
                # Create a relative path to preserve the folder structure in S3
                relative_path = os.path.relpath(local_path, path)
                # On Windows, os.path.relpath can use backslashes; S3 needs forward slashes
                s3_key = relative_path.replace("\\", "/") 
                
                print(f"Uploading {local_path} to s3://{bucketname}/{s3_key}")
                s3_client.upload_file(local_path, bucketname, s3_key)

        print("\nUpload completed successfully!")
        return True

    except FileNotFoundError:
        print(f"Error: The directory '{path}' was not found.")
        return False
    except NoCredentialsError:
        print("Error: AWS credentials not found. Please configure your environment variables.")
        return False
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return False

# Start the upload process
upload_directory(LOCAL_DATASET_PATH, BUCKET_NAME)

