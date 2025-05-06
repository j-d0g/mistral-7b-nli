import os
import sys
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download
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
  docker run --rm -v $(pwd):/app -w /app mistral-nli-ft python3 results/download_results.py
""")
    sys.exit(1)

# User info (default values for results dataset)
USERNAME = os.getenv("HF_USERNAME", "jd0g")
REPO_NAME = os.getenv("HF_RESULTS_REPO_NAME", "nlistral-7b-results") # Default repo for results
REPO_ID = f"{USERNAME}/{REPO_NAME}"
LOCAL_RESULTS_DIR = Path("results")

print(f"Preparing to download results from dataset {REPO_ID}...")

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
    
    # Check if dataset repository exists
    try:
        api.repo_info(repo_id=REPO_ID, repo_type="dataset")
        print(f"✓ Dataset repository {REPO_ID} found")
    except Exception:
        print(f"""
ERROR: Dataset repository {REPO_ID} not found.
Cannot download results. Please check the username and repository name.
""")
        sys.exit(1)

    # Create local results directory if it doesn't exist
    os.makedirs(LOCAL_RESULTS_DIR, exist_ok=True)
    print(f"Local results will be saved to: {LOCAL_RESULTS_DIR}")
    
    # Download the entire dataset repository snapshot
    print(f"\nDownloading results from {REPO_ID}...")
    try:
        snapshot_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            token=HF_TOKEN,
            local_dir=str(LOCAL_RESULTS_DIR),
            local_dir_use_symlinks=False, # Avoid symlinks for portability
            resume_download=True # Resume interrupted downloads
        )
        print(f"✓ Successfully downloaded results to {LOCAL_RESULTS_DIR}")
    except Exception as e:
        print(f"⚠ Error downloading results: {e}")
        sys.exit(1)

    print("\nDownload complete!")
except Exception as e:
    print(f"""
ERROR: Failed to download from repository
{str(e)}

This could be due to:
1. Repository permissions (you need read access)
2. Network connectivity issues
3. Invalid repository ID
""")
    sys.exit(1)

print(f"""
🎉 Success! Results downloaded to: {LOCAL_RESULTS_DIR}
""") 