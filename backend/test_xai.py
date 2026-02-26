import cv2
from pathlib import Path
from xai.xai_engine import XAIEngine

# ---------------------------
# Setup Downloads Folder
# ---------------------------
downloads_path = Path.home() / "Downloads"

# Initialize engine
engine = XAIEngine()

# ---------------------------
# TEST IMAGE
# ---------------------------
image_path = "dataset/image/train/FAKE/fake_0.jpg"
image = cv2.imread(image_path)

heatmap = engine.explain_image(image)

image_output_path = downloads_path / "output_image_heatmap.jpg"
cv2.imwrite(str(image_output_path), heatmap)

print(f"✅ Image explanation saved at {image_output_path}")

# ---------------------------
# TEST VIDEO
# ---------------------------
video_path = "dataset/video/train/FAKE/000_003.mp4"
video_results = engine.explain_video(video_path)

for i, frame in enumerate(video_results):
    video_output_path = downloads_path / f"video_heatmap_{i}.jpg"
    cv2.imwrite(str(video_output_path), frame)

print("✅ Video explanation saved in Downloads")

# ---------------------------
# TEST AUDIO
# ---------------------------
# audio_path = "test.wav"

# spec_img, saliency = engine.explain_audio(audio_path)

# audio_spec_path = downloads_path / "audio_spectrogram.jpg"
# saliency_path = downloads_path / "audio_saliency.jpg"

# cv2.imwrite(str(audio_spec_path), spec_img)

# # Convert saliency to image format
# saliency_img = (saliency * 255).astype("uint8")
# cv2.imwrite(str(saliency_path), saliency_img)

# print("✅ Audio explanation saved in Downloads")