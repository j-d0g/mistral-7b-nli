import os
import sys
import glob
from pathlib import Path
from huggingface_hub import HfApi, upload_folder
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
  docker run --rm -v $(pwd):/app -w /app mistral-nli-ft python3 results/upload_results.py
""")
    sys.exit(1)

# User info (default values for results dataset)
USERNAME = os.getenv("HF_USERNAME", "jd0g")
REPO_NAME = os.getenv("HF_RESULTS_REPO_NAME", "Mistral-NLI-Results") # Default repo for results
REPO_ID = f"{USERNAME}/{REPO_NAME}"
LOCAL_RESULTS_DIR = Path("results")

if not os.path.exists(LOCAL_RESULTS_DIR) or not os.listdir(LOCAL_RESULTS_DIR):
    print(f"⚠ Results directory '{LOCAL_RESULTS_DIR}' is empty or does not exist. Nothing to upload.")
    sys.exit(0)

print(f"Preparing to upload results to dataset {REPO_ID}...")

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
"""")
        sys.exit(1)

    # Check repository exists or create it as a dataset
    try:
        repo_info = api.repo_info(repo_id=REPO_ID, repo_type="dataset")
        print(f"✓ Dataset repository {REPO_ID} exists")
    except Exception:
        print(f"Dataset repository {REPO_ID} not found. Creating new dataset repository...")
        api.create_repo(
            repo_id=REPO_ID,
            repo_type="dataset",
            private=True, # Results likely contain experiment details, keep private by default
            exist_ok=True
        )
        print(f"✓ Created dataset repository {REPO_ID}")
        # Wait a moment for the repository to be fully created
        import time
        print("Waiting for repository to initialize...")
        time.sleep(5)

    # Upload the entire results folder
    print(f"\nUploading contents of {LOCAL_RESULTS_DIR}...")

    try:
        api.upload_folder(
            folder_path=str(LOCAL_RESULTS_DIR),
            repo_id=REPO_ID,
            repo_type="dataset",
            path_in_repo="." # Upload to the root of the dataset repo
        )
        print(f"✓ Successfully uploaded results to {REPO_ID}")
    except Exception as e:
        print(f"⚠ Error uploading results: {e}")
        sys.exit(1)

    print("\nUpload complete!")
except Exception as e:
    print(f"""
ERROR: Failed to upload to repository
{str(e)}

This could be due to:
1. Repository permissions (you need write access)
2. Network connectivity issues
3. Invalid repository ID
"""")
    sys.exit(1)

print(f"""
🎉 Success! Results uploaded to dataset: {REPO_ID}

View your dataset at: https://huggingface.co/datasets/{REPO_ID}
""") 