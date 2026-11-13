import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download, HfApi, list_repo_files

# Load environment variables from .env file
load_dotenv()

# HF token is optional: the dataset is public. It is read from .env if present.
HF_TOKEN = os.getenv("HF_TOKEN") or None

# User info
USERNAME = "jd0g"
REPO_NAME = "Mistral-NLI-Thoughts"
REPO_ID = f"{USERNAME}/{REPO_NAME}"
# The complete copy of the data lives under the "data/" prefix in the repository,
# mirroring this project's layout, so files are downloaded into the project root.
REMOTE_PREFIX = "data/"
LOCAL_DATA_DIR = Path(".")

# Files to exclude when downloading
EXCLUDED_FILES = [
    "data/download_dataset.py",
    "data/upload_dataset.py",
    "data/download_data.py",
    "data/README.md"
]

def download_dataset_from_hf():
    """Download all files from the HuggingFace dataset repository, preserving structure."""
    api = HfApi(token=HF_TOKEN)
    
    try:
        # List all files in the repository
        all_files = list_repo_files(
            repo_id=REPO_ID,
            repo_type="dataset",
            token=HF_TOKEN
        )
        
        # Keep only the "data/"-prefixed copy and drop excluded files
        files_to_download = [f for f in all_files if f.startswith(REMOTE_PREFIX) and f not in EXCLUDED_FILES]
        
        if not files_to_download:
            print("No files found to download.")
            return
        
        print(f"Found {len(files_to_download)} files to download.")
        
        # Download each file
        for file_path in files_to_download:
            try:
                # Create the directory structure if it doesn't exist
                local_path = LOCAL_DATA_DIR / file_path
                local_dir = local_path.parent
                os.makedirs(local_dir, exist_ok=True)
                
                print(f"Downloading: {file_path}")
                hf_hub_download(
                    repo_id=REPO_ID,
                    filename=file_path,
                    repo_type="dataset",
                    token=HF_TOKEN,
                    local_dir=LOCAL_DATA_DIR,
                    local_dir_use_symlinks=False
                )
            except Exception as e:
                print(f"Error downloading {file_path}: {e}")
        
        print("Download complete!")
    except Exception as e:
        print(f"Error accessing repository: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Download all files from HuggingFace
    download_dataset_from_hf()
    
    print(f"Dataset downloaded from: https://huggingface.co/datasets/{REPO_ID}") 