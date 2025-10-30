import os
from kaggle.api.kaggle_api_extended import KaggleApi

# Initialize and authenticate Kaggle API
api = KaggleApi()
api.authenticate()

# Dataset name (small, image-based)
dataset = "sohamparolia/celeb-df-v2-images"

# Directory to save dataset
save_path = os.path.join(os.getcwd(), "data")
os.makedirs(save_path, exist_ok=True)

print(f"📂 Downloading dataset to: {save_path}")
print(f"🌐 Dataset: https://www.kaggle.com/datasets/{dataset}")

# ✅ Download and unzip dataset
try:
    print("⬇️  Starting download... (this may take a few minutes)")
    api.dataset_download_files(dataset, path=save_path, unzip=True)
    print("✅ Dataset downloaded and extracted successfully!")
except Exception as e:
    print("❌ Error downloading dataset:", e)
