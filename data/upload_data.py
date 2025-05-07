import os
import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import HfApi, upload_folder

def parse_args():
    parser = argparse.ArgumentParser(description="Upload dataset to Hugging Face")
    parser.add_argument('--env', type=str, help='Path to .env file')
    return parser.parse_args()

# Load environment variables
args = parse_args()
if args.env and os.path.exists(args.env):
    load_dotenv(args.env)
    print(f"Loaded environment from {args.env}")
else:
    load_dotenv()

# Get HF token from environment variables
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    print("Error: HF_TOKEN not found in .env file")
    sys.exit(1)

# User info
USERNAME = "jd0g"
REPO_NAME = "Mistral-NLI-Thoughts"
REPO_ID = f"{USERNAME}/{REPO_NAME}"
LOCAL_DATA_DIR = Path("data")

# Files to exclude from upload
EXCLUDED_FILES = [
    "upload_data.py",
    "download_data.py",
    "README.md",
    "__pycache__"
]

def upload_dataset_to_hf():
    """Upload all data to the HuggingFace dataset repository, preserving structure."""
    api = HfApi(token=HF_TOKEN)
    
    try:
        # Check if repository exists or create it
        try:
            api.repo_info(repo_id=REPO_ID, repo_type="dataset")
            print(f"Repository {REPO_ID} exists")
        except Exception:
            print(f"Creating new repository {REPO_ID}...")
            api.create_repo(
                repo_id=REPO_ID,
                repo_type="dataset",
                private=True,
                exist_ok=True
            )
        
        # Upload the data directory
        print(f"Uploading files from {LOCAL_DATA_DIR}...")
        
        # Get directories to upload
        data_dirs = [d for d in os.listdir(LOCAL_DATA_DIR) if os.path.isdir(os.path.join(LOCAL_DATA_DIR, d))]
        data_dirs = [d for d in data_dirs if d not in EXCLUDED_FILES and not d.startswith('.')]
        
        # Upload each directory separately
        for data_dir in data_dirs:
            dir_path = os.path.join(LOCAL_DATA_DIR, data_dir)
            
            print(f"Uploading directory: {data_dir}")
            upload_folder(
                folder_path=dir_path,
                repo_id=REPO_ID,
                repo_type="dataset",
                path_in_repo=data_dir,
                token=HF_TOKEN
            )
            print(f"Successfully uploaded {data_dir}")
        
        print("Upload complete!")
    except Exception as e:
        print(f"Error accessing or uploading to repository: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Upload all files to HuggingFace
    upload_dataset_to_hf()
    
    print(f"Dataset uploaded to: https://huggingface.co/datasets/{REPO_ID}") 