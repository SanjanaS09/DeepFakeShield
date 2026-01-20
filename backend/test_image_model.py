"""Test image model with checkpoint"""
import sys
from pathlib import Path
import argparse
import logging
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

BACKEND_ROOT = Path(__file__).parent
sys.path.insert(0, str(BACKEND_ROOT))

from models.image_detector import ImageDeepfakeDetector
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_model(checkpoint_path, dataset_dir):
    """Test image model on dataset"""
    logger.info("="*70)
    logger.info("TESTING IMAGE MODEL")
    logger.info("="*70)
    
    # Check checkpoint
    checkpoint = Path(checkpoint_path)
    if checkpoint.exists():
        size_mb = checkpoint.stat().st_size / (1024 * 1024)
        logger.info(f"✅ Model file found: {checkpoint_path}")
        logger.info(f"   Size: {size_mb:.2f} MB")
    else:
        logger.error(f"❌ Model file not found: {checkpoint_path}")
        return
    
    logger.info("\n" + "="*70)
    logger.info("Loading model...")
    
    # Load model (✅ NO num_classes parameter)
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {device}")
        
        detector = ImageDeepfakeDetector(
            model_path=checkpoint_path,
            device=device,
            pretrained=False  # Don't load pretrained, use checkpoint weights
        )
        logger.info("✅ Model loaded successfully!")
        
    except Exception as e:
        logger.error(f"❌ ERROR loading model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test on dataset
    logger.info("\n" + "="*70)
    logger.info("Testing on dataset...")
    
    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        logger.error(f"Dataset not found: {dataset_dir}")
        return
    
    predictions = []
    labels = []
    file_count = 0
    
    # Load REAL images
    real_dir = dataset_path / 'REAL'
    if real_dir.exists():
        for img_path in sorted(real_dir.glob('*')):
            if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                try:
                    result = detector.detect(str(img_path))
                    pred = 0 if result['prediction'] == 'REAL' else 1
                    predictions.append(pred)
                    labels.append(0)
                    file_count += 1
                except Exception as e:
                    logger.warning(f"Failed to process {img_path}: {e}")
    
    # Load FAKE images
    fake_dir = dataset_path / 'FAKE'
    if fake_dir.exists():
        for img_path in sorted(fake_dir.glob('*')):
            if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                try:
                    result = detector.detect(str(img_path))
                    pred = 0 if result['prediction'] == 'REAL' else 1
                    predictions.append(pred)
                    labels.append(1)
                    file_count += 1
                except Exception as e:
                    logger.warning(f"Failed to process {img_path}: {e}")
    
    if not predictions:
        logger.error("No images processed!")
        return
    
    # Calculate metrics
    logger.info(f"\nProcessed {file_count} images")
    logger.info("\n" + "="*70)
    logger.info("RESULTS")
    logger.info("="*70)
    
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, zero_division=0)
    recall = recall_score(labels, predictions, zero_division=0)
    f1 = f1_score(labels, predictions, zero_division=0)
    
    cm = confusion_matrix(labels, predictions)
    tn, fp, fn, tp = cm.ravel() if len(cm.shape) > 1 else [cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]]
    
    logger.info(f"\n📊 OVERALL METRICS:")
    logger.info(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    logger.info(f"  Precision: {precision:.4f}")
    logger.info(f"  Recall:    {recall:.4f}")
    logger.info(f"  F1-Score:  {f1:.4f}")
    
    logger.info(f"\n📈 CONFUSION MATRIX:")
    logger.info(f"                 Predicted")
    logger.info(f"                REAL    FAKE")
    logger.info(f"  Actual REAL    {tn:4d}    {fp:4d}")
    logger.info(f"  Actual FAKE    {fn:4d}    {tp:4d}")
    
    logger.info("\n" + "="*70)

def main():
    parser = argparse.ArgumentParser(description='Test image deepfake model')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--dataset', type=str, required=True, help='Path to test dataset')
    
    args = parser.parse_args()
    
    test_model(args.checkpoint, args.dataset)

if __name__ == '__main__':
    main()
