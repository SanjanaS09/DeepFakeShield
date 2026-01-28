import os
import sys
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Allow imports from backend/
BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))

from models.video_detector_frame_based import FrameBasedVideoDetector

# ===============================
# CONFIG
# ===============================
DATASET_ROOT = Path("dataset/video")
MODEL_PATH = "checkpoints/video/best_model.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SPLITS = ["train", "validation", "test"]
LABEL_MAP = {"REAL": 0, "FAKE": 1}

# ===============================
# EVALUATION FUNCTION
# ===============================
def evaluate_split(split_name: str, model: FrameBasedVideoDetector):
    print(f"\n📊 Evaluating {split_name.upper()} split")

    y_true = []
    y_pred = []

    split_path = DATASET_ROOT / split_name

    for label_name, label_id in LABEL_MAP.items():
        video_dir = split_path / label_name

        if not video_dir.exists():
            print(f"⚠️ Skipping missing folder: {video_dir}")
            continue

        videos = list(video_dir.glob("*.mp4"))
        print(f"  ▶ {label_name}: {len(videos)} videos")

        for video_path in tqdm(videos, desc=f"{label_name}"):
            result = model.predict(str(video_path))

            if result.get("status") != "success":
                continue

            pred_label = 1 if result["prediction"] == "FAKE" else 0

            y_true.append(label_id)
            y_pred.append(pred_label)

    # ===============================
    # METRICS
    # ===============================
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    print("\n✅ RESULTS")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")

    print("\nConfusion Matrix [REAL, FAKE]")
    print(cm)

    print("\nClassification Report")
    print(classification_report(
        y_true,
        y_pred,
        target_names=["REAL", "FAKE"],
        zero_division=0
    ))

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "confusion_matrix": cm
    }

# ===============================
# MAIN
# ===============================
if __name__ == "__main__":
    print("🚀 Loading video model...")
    model = FrameBasedVideoDetector(
        model_path=MODEL_PATH,
        device=DEVICE,
        num_frames=8,
        frame_size=(224, 224)
    )

    results = {}

    for split in SPLITS:
        results[split] = evaluate_split(split, model)

    print("\n" + "=" * 60)
    print("📌 FINAL SUMMARY")
    for split, metrics in results.items():
        print(f"\n{split.upper()}")
        for k, v in metrics.items():
            if k != "confusion_matrix":
                print(f"  {k}: {v:.4f}")
