import os, shutil, random

# Paths
data_dir = "data"
real_dir = os.path.join(data_dir, "real")
fake_dir = os.path.join(data_dir, "fake")

os.makedirs(real_dir, exist_ok=True)
os.makedirs(fake_dir, exist_ok=True)

# Get all image files
all_images = [f for f in os.listdir(data_dir) if f.endswith(".jpg")]
random.shuffle(all_images)

# Split half as real, half as fake (for prototype)
half = len(all_images) // 2
real_images = all_images[:half]
fake_images = all_images[half:]

for img in real_images:
    shutil.move(os.path.join(data_dir, img), os.path.join(real_dir, img))

for img in fake_images:
    shutil.move(os.path.join(data_dir, img), os.path.join(fake_dir, img))

print(f"✅ Moved {len(real_images)} images to 'real/' and {len(fake_images)} images to 'fake/'.")
