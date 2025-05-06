import os
import sys
import time
import argparse
from pathlib import Path
from huggingface_hub import hf_hub_download, HfApi, list_repo_files
from dotenv import load_dotenv

# Parse command line arguments
parser = argparse.ArgumentParser(description='Download models from HuggingFace Hub')
parser.add_argument('--model', type=str, help='Specific model to download')
parser.add_argument('--best', action='store_true', help='Download best model')
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
  docker run --rm -v $(pwd):/app -w /app mistral-nli-ft python3 models/download_model.py
""")
    sys.exit(1)

# User info
USERNAME = "jd0g"
REPO_NAME = "nlistral-7b-qlora"
REPO_ID = f"{USERNAME}/{REPO_NAME}"
LOCAL_MODEL_DIR = Path("models")

# Remote HuggingFace repo paths for the models
HF_MODEL_PATHS = [
    "nlistral-7b-qlora-ablation0-best",     # Optimized setting for ablation study 0
    "nlistral-7b-qlora-ablation1-best",     # Best ablation
    "nlistral-7b-qlora-ablation2-best",     # Optimized setting for ablation study 2
]

# Map of HF repo paths to local directory names
MODEL_PATH_MAPPING = {
    "nlistral-7b-qlora-ablation0-best": "nlistral-ablation0",
    "nlistral-7b-qlora-ablation1-best": "nlistral-ablation1",
    "nlistral-7b-qlora-ablation2-best": "nlistral-ablation2",
}

if args.best:
    HF_MODEL_PATHS = ["nlistral-7b-qlora-ablation1-best"]

# If specific model requested, filter the list
if args.model:
    if args.model in HF_MODEL_PATHS:
        HF_MODEL_PATHS = [args.model]
    elif args.model in MODEL_PATH_MAPPING.values():
        # Find the corresponding HF path for the requested local directory name
        hf_path = next((hf for hf, local in MODEL_PATH_MAPPING.items() if local == args.model), None)
        if hf_path:
            HF_MODEL_PATHS = [hf_path]
        else:
            print(f"⚠ Specified model '{args.model}' not found.")
            sys.exit(1)
    else:
        print(f"⚠ Specified model '{args.model}' not found.")
        sys.exit(1)

# Create the model directory if it doesn't exist
os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)

print(f"Downloading model files from {REPO_ID}...")
print(f"Files will be saved to: {LOCAL_MODEL_DIR}")

try:
    # Verify HF token is valid
    try:
        api = HfApi(token=HF_TOKEN)
        # Try a simple operation to test token validity
        api.whoami()
        print("✓ HF Token is valid")
    except Exception as e:
        print(f"""
ERROR: Invalid or expired Hugging Face token
{str(e)}

Please ensure your token has appropriate permissions and hasn't expired.
Generate a new token at https://huggingface.co/settings/tokens
""")
        sys.exit(1)
        
    # Download required files for each model path
    for hf_model_path in HF_MODEL_PATHS:
        # Get the local directory name
        local_model_dir = MODEL_PATH_MAPPING.get(hf_model_path, hf_model_path)
        
        print(f"\nDownloading files for {hf_model_path} -> {local_model_dir}...")
        
        # Define required files for this model path
        required_files = [
            f"{hf_model_path}/adapter_config.json",
            f"{hf_model_path}/adapter_model.safetensors",
            f"{hf_model_path}/special_tokens_map.json",
            f"{hf_model_path}/tokenizer.model",
            f"{hf_model_path}/tokenizer.json",
            f"{hf_model_path}/tokenizer_config.json"
        ]
        
        # Create model path directory using the LOCAL name
        model_dir = LOCAL_MODEL_DIR / local_model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        # Download each required file and move to the correct local directory
        successful_downloads = 0
        for file_path in required_files:
            try:
                print(f"Downloading: {file_path}")
                try:
                    # Download to a temporary location first
                    temp_file = hf_hub_download(
                        repo_id=REPO_ID,
                        filename=file_path,
                        token=HF_TOKEN,
                        local_dir=LOCAL_MODEL_DIR,
                        local_dir_use_symlinks=False
                    )
                    
                    # Get just the filename without path
                    file_name = os.path.basename(file_path)
                    
                    # Destination path with the local model directory
                    dest_path = model_dir / file_name
                    
                    # If the file was downloaded to a different path than our desired local path
                    if os.path.dirname(temp_file) != str(model_dir):
                        # Copy to the correct location and remove the original if needed
                        import shutil
                        if os.path.exists(dest_path):
                            os.remove(dest_path)
                        shutil.copy2(temp_file, dest_path)
                        
                        # Remove the original if it's in a different directory
                        if os.path.dirname(temp_file) != str(model_dir):
                            # Make sure we don't delete the file we just copied
                            if os.path.exists(temp_file) and os.path.exists(dest_path):
                                os.remove(temp_file)
                                
                                # Remove empty directories
                                temp_dir = os.path.dirname(temp_file)
                                while temp_dir and temp_dir != str(LOCAL_MODEL_DIR):
                                    if not os.listdir(temp_dir):
                                        os.rmdir(temp_dir)
                                        temp_dir = os.path.dirname(temp_dir)
                                    else:
                                        break
                    
                    print(f"✓ Downloaded {file_path} to {dest_path}")
                    successful_downloads += 1
                except Exception as e:
                    if "404" in str(e):
                        print(f"⚠ File not found: {file_path} (skipping)")
                    else:
                        print(f"⚠ Error downloading {file_path}: {e}")
            except Exception as e:
                print(f"⚠ Error processing {file_path}: {e}")
        
        # Verify if we've downloaded enough files
        if successful_downloads < 2:
            print(f"""
WARNING: Few files were downloaded successfully for {hf_model_path}.
This may indicate an issue with repository access or structure.
""")
        else:
            print(f"✓ Successfully downloaded files for {hf_model_path} to {local_model_dir}")
    
    print("\nDownload complete!")
except Exception as e:
    print(f"""
ERROR: Failed to access repository
{str(e)}

This could be due to:
1. Repository permissions (private repo)
2. Network connectivity issues
3. Invalid repository ID
""")
    sys.exit(1)

print(f"""
🎉 Success! Model files downloaded to: {LOCAL_MODEL_DIR}

To run inference:
./run_inference.sh --model models/nlistral-ablation0 --data data/sample/demo.csv
""") 