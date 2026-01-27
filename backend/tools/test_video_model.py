import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))

from models.video_detector_frame_based import FrameBasedVideoDetector

detector = FrameBasedVideoDetector(
    model_path="checkpoints/video/best_model.pth",
    device="cpu"
)

result = detector.predict("dataset/video/test/REAL/008.mp4")
print(result)
