import os
import sys
import time
from pathlib import Path
from huggingface_hub import hf_hub_download, HfApi, list_repo_files
from dotenv import load_dotenv

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
REPO_NAME = "Mistral-Thinking-NLI"
REPO_ID = f"{USERNAME}/{REPO_NAME}"
LOCAL_MODEL_DIR = Path("models")

# List of checkpoint model paths to download
MODEL_PATHS = [
    # "mistral-thinking-abl0",
    "mistral-thinking-abl0-ext", # Best model
    "mistral-thinking-abl0-merged",
    # "mistral-thinking-abl0_dist",
    # "mistral-thinking-abl1",
    # "mistral-thinking-abl2",
    # "mistral-thinking-abl3",
    # "mistral-7b-nli-cot"
]

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
    for MODEL_PATH in MODEL_PATHS:
        print(f"\nDownloading files for {MODEL_PATH}...")
        
        # Define required files for this model path
        required_files = [
            f"{MODEL_PATH}/adapter_config.json",
            f"{MODEL_PATH}/adapter_model.safetensors",
            f"{MODEL_PATH}/special_tokens_map.json",
            f"{MODEL_PATH}/tokenizer.model",
            f"{MODEL_PATH}/tokenizer.json",
            f"{MODEL_PATH}/tokenizer_config.json"
        ]
        
        # Create model path directory
        model_dir = LOCAL_MODEL_DIR / MODEL_PATH
        os.makedirs(model_dir, exist_ok=True)
        
        # Download each required file
        successful_downloads = 0
        for file_path in required_files:
            try:
                print(f"Downloading: {file_path}")
                try:
                    hf_hub_download(
                        repo_id=REPO_ID,
                        filename=file_path,
                        token=HF_TOKEN,
                        local_dir=LOCAL_MODEL_DIR,
                        local_dir_use_symlinks=False
                    )
                    print(f"✓ Downloaded {file_path}")
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
WARNING: Few files were downloaded successfully for {MODEL_PATH}.
This may indicate an issue with repository access or structure.
""")
        else:
            print(f"✓ Successfully downloaded files for {MODEL_PATH}")
    
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
./run_inference.sh --model models/Mistral_Thinking_Abl2/checkpoint-2000 --data data/sample/demo.csv
""") 