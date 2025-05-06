"""
Uploads models of specified path in models/ directory to HuggingFace repository.
"""

import os
import sys
import glob
import time
import argparse
from pathlib import Path
from huggingface_hub import HfApi, upload_folder
from dotenv import load_dotenv

# Parse command line arguments
parser = argparse.ArgumentParser(description='Upload models to HuggingFace Hub')
parser.add_argument('--model', type=str, help='Specific model to upload')
args = parser.parse_args()

# Automatically load environment variables from .env file
if os.path.exists('.env'):
    load_dotenv()
    print("✓ Loaded environment variables from .env file")
else:
    print("⚠ No .env file found in project root")

# Get HF token from environment variables
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    print("""
ERROR: HF_TOKEN not found in environment variables.
Please ensure:
1. You have created a token at https://huggingface.co/settings/tokens
2. Added it to your .env file with format: HF_TOKEN=your_token_here

Example usage:
  docker run --rm -v $(pwd):/app -w /app mistral-nli-ft python3 models/upload_model.py
""")
    sys.exit(1)

# User info (default values)
USERNAME = os.getenv("HF_USERNAME", "jd0g")
REPO_NAME = os.getenv("HF_REPO_NAME", "nlistral-7b-qlora")
REPO_ID = f"{USERNAME}/{REPO_NAME}"
LOCAL_MODEL_DIR = Path("models")

# Updated list of model paths with new naming convention
NEW_MODEL_PATHS = [
    "nlistral-7b-qlora-ablation0-best",     # Optimized setting for ablation study 0
    "nlistral-7b-qlora-ablation1-best",     # Best ablatio
    "nlistral-7b-qlora-ablation2-best",     # Optimized setting for ablation study 2
]

# Determine which models to upload
if args.model:
    # Upload a specific model
    MODEL_PATHS = [args.model]
else:
    # Upload all models
    MODEL_PATHS = NEW_MODEL_PATHS

print(f"Preparing to upload models to {REPO_ID}...")

try:
    # Verify HF token is valid
    try:
        api = HfApi(token=HF_TOKEN)
        user_info = api.whoami()
        print(f"✓ HF Token is valid (logged in as: {user_info['name']})")
    except Exception as e:
        print(f"""
ERROR: Invalid or expired Hugging Face token
{str(e)}

Please ensure your token has appropriate permissions and hasn't expired.
Generate a new token at https://huggingface.co/settings/tokens
""")
        sys.exit(1)
    
    # Check repository exists or create it
    try:
        repo_info = api.repo_info(repo_id=REPO_ID, repo_type="model")
        print(f"✓ Repository {REPO_ID} exists")
    except Exception:
        print(f"Repository {REPO_ID} not found. Creating new repository...")
        api.create_repo(
            repo_id=REPO_ID,  # Use full repo_id instead of just REPO_NAME
            repo_type="model",
            private=True,
            exist_ok=True
        )
        print(f"✓ Created repository {REPO_ID}")
        # Wait a moment for the repository to be fully created
        print("Waiting for repository to initialize...")
        time.sleep(5)
    
    # Upload each model path
    for model_path in MODEL_PATHS:
        full_path = LOCAL_MODEL_DIR / model_path
        
        if not os.path.exists(full_path):
            print(f"⚠ Model path not found: {full_path} (skipping)")
            continue
            
        print(f"\nUploading model: {model_path}")
        
        # Check for required files
        required_files = ["adapter_config.json", "adapter_model.safetensors", 
                         "special_tokens_map.json", "tokenizer_config.json", "tokenizer.json"]
        
        missing_files = []
        for req_file in required_files:
            if not os.path.exists(full_path / req_file):
                missing_files.append(req_file)
        
        if missing_files:
            print(f"⚠ Warning: Missing required files in {model_path}:")
            for missing in missing_files:
                print(f"  - {missing}")
            choice = input("Continue uploading this model anyway? (y/n): ")
            if choice.lower() != "y":
                print(f"Skipping upload of {model_path}")
                continue
        
        # Get list of files in the model directory
        files_to_upload = list(glob.glob(str(full_path / "*")))
        print(f"Found {len(files_to_upload)} files to upload")
        
        # Upload to HF Hub
        try:
            api.upload_folder(
                folder_path=str(full_path),
                repo_id=REPO_ID,
                repo_type="model",
                path_in_repo=model_path
            )
            print(f"✓ Successfully uploaded {model_path}")
        except Exception as e:
            print(f"⚠ Error uploading {model_path}: {e}")
    
    print("\nUpload complete!")
except Exception as e:
    print(f"""
ERROR: Failed to upload to repository
{str(e)}

This could be due to:
1. Repository permissions (you need write access)
2. Network connectivity issues
3. Invalid repository ID
""")
    sys.exit(1)

print(f"""
🎉 Success! Model files uploaded to: {REPO_ID}

View your models at: https://huggingface.co/{REPO_ID}

To customize the repository:
1. Edit the model card at https://huggingface.co/{REPO_ID}
2. Add tags and metadata through the web interface
""") 