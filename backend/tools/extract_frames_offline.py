# tools/extract_frames_temp.py
import cv2
import numpy as np
from pathlib import Path

VIDEO_ROOT = Path("/content/drive/MyDrive/DeepFakeShield/backend/dataset/video")
FRAME_ROOT = Path("/content/drive/MyDrive/DeepFakeShield/backend/dataset/video_frames")  # 🔥 TEMP LOCATION

NUM_FRAMES = 8
IMAGE_SIZE = (224, 224)
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}

def extract(video_path: Path, out_dir: Path):
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return

    idxs = np.linspace(0, total - 1, NUM_FRAMES, dtype=int)
    for i, idx in enumerate(idxs):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.resize(frame, IMAGE_SIZE)
        cv2.imwrite(str(out_dir / f"{video_path.stem}_{i}.jpg"), frame)
    cap.release()

for split in ["train", "validation", "test"]:
    for label in ["REAL", "FAKE"]:
        src = VIDEO_ROOT / split / label
        dst = FRAME_ROOT / split / label
        dst.mkdir(parents=True, exist_ok=True)

        videos = [v for v in src.iterdir() if v.suffix.lower() in VIDEO_EXTS]
        print(f"{split}/{label}: {len(videos)} videos")

        for video in videos:
            extract(video, dst)

print("✅ Frames extracted to TEMP storage")
