# src/upload_data.py
import os
from google.cloud import storage
import glob
from tqdm import tqdm
from google.api_core import exceptions

def upload_to_gcp():
    """Upload chunks and embeddings to GCP bucket"""
    # Get environment variables
    project_id = os.getenv('GCP_PROJECT')
    bucket_name = os.getenv('GCS_BUCKET_NAME')
    
    print(f"Starting upload to GCP Project: {project_id}, Bucket: {bucket_name}")
    
    # Initialize GCP client
    storage_client = storage.Client(project=project_id)
    
    # Get or create bucket
    try:
        bucket = storage_client.get_bucket(bucket_name)
        print(f"Using existing bucket: {bucket_name}")
    except exceptions.NotFound:
        try:
            bucket = storage_client.create_bucket(bucket_name, location="us-central1")
            print(f"Created new bucket: {bucket_name}")
        except exceptions.Forbidden as e:
            print(f"Error: Insufficient permissions to create bucket. Please create the bucket '{bucket_name}' manually in the GCP Console.")
            print(f"Detailed error: {str(e)}")
            return
    except Exception as e:
        print(f"Error accessing bucket: {str(e)}")
        return
    
    # Upload embeddings
    embedding_files = glob.glob("/app/data/embeddings-recursive-split-*.jsonl")
    print(f"Found {len(embedding_files)} embedding files to upload")
    
    for file_path in tqdm(embedding_files, desc="Uploading embeddings"):
        try:
            blob_name = f"embeddings/{os.path.basename(file_path)}"
            blob = bucket.blob(blob_name)
            
            # Check if file already exists
            if blob.exists():
                print(f"⚠️  File {blob_name} already exists, skipping...")
                continue
                
            blob.upload_from_filename(file_path)
            print(f"✓ Uploaded {os.path.basename(file_path)}")
        except Exception as e:
            print(f"❌ Error uploading {file_path}: {str(e)}")
    
    # Upload chunks
    chunk_files = glob.glob("/app/data/chunks-recursive-split-*.jsonl")
    print(f"Found {len(chunk_files)} chunk files to upload")
    
    for file_path in tqdm(chunk_files, desc="Uploading chunks"):
        try:
            blob_name = f"chunks/{os.path.basename(file_path)}"
            blob = bucket.blob(blob_name)
            
            # Check if file already exists
            if blob.exists():
                print(f"⚠️  File {blob_name} already exists, skipping...")
                continue
                
            blob.upload_from_filename(file_path)
            print(f"✓ Uploaded {os.path.basename(file_path)}")
        except Exception as e:
            print(f"❌ Error uploading {file_path}: {str(e)}")
    
    print("\nUpload completed successfully!")

if __name__ == "__main__":
    upload_to_gcp()