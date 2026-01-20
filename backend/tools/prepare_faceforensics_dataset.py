import os
import shutil
import random
from pathlib import Path

random.seed(42)

# =========================
# SOURCE PATHS (YOUR DATA)
# =========================
FAKE_SOURCES = [
    r"D:\faceforensics_data\manipulated_sequences\DeepFakeDetection\c23\videos",
    r"D:\faceforensics_data\manipulated_sequences\Deepfakes\c23\videos",
]

REAL_SOURCES = [
    r"D:\faceforensics_data\original_sequences\actors\c23\videos",
    r"D:\faceforensics_data\original_sequences\youtube\c23\videos",
]

# =========================
# TARGET BASE PATH
# =========================
TARGET_BASE = Path(
    r"C:\Users\Sanjana\DeepFakeShield\backend\dataset\video"
)

SPLITS = {
    "train": 0.70,
    "validation": 0.15,
    "test": 0.15,
}

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}

# =========================
def collect_videos(folders):
    videos = []
    for folder in folders:
        for file in os.listdir(folder):
            if Path(file).suffix.lower() in VIDEO_EXTS:
                videos.append(Path(folder) / file)
    return videos


def split_data(files):
    random.shuffle(files)
    n = len(files)

    train_end = int(n * SPLITS["train"])
    val_end = train_end + int(n * SPLITS["validation"])

    return {
        "train": files[:train_end],
        "validation": files[train_end:val_end],
        "test": files[val_end:],
    }


def copy_files(files, label):
    for split, split_files in files.items():
        target_dir = TARGET_BASE / split / label
        target_dir.mkdir(parents=True, exist_ok=True)

        for src in split_files:
            dst = target_dir / src.name
            if not dst.exists():
                shutil.copy2(src, dst)


def main():
    print("📥 Collecting FAKE videos...")
    fake_videos = collect_videos(FAKE_SOURCES)
    print(f"   Found {len(fake_videos)} FAKE videos")

    print("📥 Collecting REAL videos...")
    real_videos = collect_videos(REAL_SOURCES)
    print(f"   Found {len(real_videos)} REAL videos")

    print("🔀 Splitting FAKE videos...")
    fake_split = split_data(fake_videos)

    print("🔀 Splitting REAL videos...")
    real_split = split_data(real_videos)

    print("📂 Copying FAKE videos...")
    copy_files(fake_split, "FAKE")

    print("📂 Copying REAL videos...")
    copy_files(real_split, "REAL")

    print("\n✅ DATASET PREPARATION COMPLETE")
    for split in SPLITS:
        f = len(fake_split[split])
        r = len(real_split[split])
        print(f"{split.upper():10s} | FAKE: {f:4d} | REAL: {r:4d}")


if __name__ == "__main__":
    main()
