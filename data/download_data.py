import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download, HfApi, list_repo_files

# Load environment variables from .env file
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

# Files to exclude when downloading
EXCLUDED_FILES = [
    "data/download_dataset.py",
    "data/upload_dataset.py",
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
        
        # Filter out excluded files
        files_to_download = [f for f in all_files if f not in EXCLUDED_FILES]
        
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