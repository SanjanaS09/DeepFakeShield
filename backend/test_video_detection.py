"""Test script for video deepfake detection"""

import sys
from pathlib import Path
import logging
import torch

BACKEND_ROOT = Path(__file__).parent
sys.path.insert(0, str(BACKEND_ROOT))

from preprocessing.video_preprocessor import VideoPreprocessor
from models.video_detector import VideoDetector
from training.train_video_model import FastFeatureExtractor

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

def test_video_detection(video_path):
    """Test video detection with trained model"""
    
    logger.info("="*60)
    logger.info("VIDEO DETECTION TEST")
    logger.info("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}\n")
    
    # 1. Initialize preprocessor
    logger.info("1. Initializing preprocessor...")
    video_preprocessor = VideoPreprocessor(
        target_size=(224, 224),
        frames_per_clip=8,
        device=str(device)
    )
    
    # 2. Preprocess video
    logger.info("\n2. Preprocessing video...")
    preprocess_result = video_preprocessor.preprocess_video(
        video_path,
        detect_faces=True
    )
    
    video_tensor = preprocess_result['video_tensor']
    logger.info(f"   Frames extracted: {preprocess_result['frames_extracted']}")
    logger.info(f"   Frames sampled: {preprocess_result['frames_sampled']}")
    logger.info(f"   Tensor shape: {video_tensor.shape}")
    
    # 3. Initialize detector with TRAINED MODEL
    logger.info("\n3. Initializing detector...")
    detector = VideoDetector(
        backbone='lstm',
        num_classes=2,
        device=str(device),
        model_path='checkpoints/video/best_model.pth'  # ✅ Load trained model
    )
    
    # 4. Extract features
    logger.info("\n4. Extracting frame features...")
    feature_extractor = FastFeatureExtractor(device=str(device))
    
    # ✅ FIX: Add batch dimension if needed
    if video_tensor.dim() == 4:  # [T, C, H, W]
        video_tensor = video_tensor.unsqueeze(0)  # [1, T, C, H, W]
    
    with torch.no_grad():
        batch_size, num_frames, c, h, w = video_tensor.shape
        video_flat = video_tensor.view(batch_size * num_frames, c, h, w)
        frame_features = feature_extractor(video_flat.to(device))  # [B*T, 512]
        frame_features = frame_features.view(batch_size, num_frames, -1)  # [B, T, 512]
    
    logger.info(f"   Frame features shape: {frame_features.shape}")
    
    # 5. Run inference
    logger.info("\n5. Running inference...")
    result = detector.predict(frame_features)
    
    # 6. Display results
    logger.info("\n6. RESULTS:")
    logger.info("="*60)
    logger.info(f"Prediction: {result['prediction']}")
    logger.info(f"Confidence: {result['confidence']:.2%}")
    logger.info(f"Real probability: {result['probabilities']['REAL']:.4f}")
    logger.info(f"Fake probability: {result['probabilities']['FAKE']:.4f}")
    logger.info("="*60)


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python test_video_detection.py <video_path>")
        print("Example: python test_video_detection.py dataset/video/test/FAKE/vs21.mp4")
        sys.exit(1)
    
    video_path = sys.argv[1]
    
    if not Path(video_path).exists():
        print(f"❌ Video not found: {video_path}")
        sys.exit(1)
    
    test_video_detection(video_path)
