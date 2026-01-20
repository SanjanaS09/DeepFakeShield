"""
✅ FIXED VIDEO MODEL TESTER
Proper evaluation with ROC-AUC and detailed analysis
"""
import os
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).parent
sys.path.insert(0, str(BACKEND_ROOT))

import argparse
import logging
import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================
# ✅ LOAD MODEL PROPERLY
# ============================================
def load_model(checkpoint_path, device='cpu'):
    """Load model with proper error checking"""
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Verify checkpoint integrity
        logger.info("\n" + "="*70)
        logger.info("📊 CHECKPOINT INTEGRITY CHECK")
        logger.info("="*70)
        
        if 'best_acc' in checkpoint:
            best_acc = checkpoint['best_acc']
            # ✅ FIX: Check if accuracy is valid (0-100%)
            if best_acc > 1.0:
                logger.error(f"❌ CORRUPTED: best_acc = {best_acc}% (should be < 100%)")
                best_acc = best_acc / 100  # Try to fix
            logger.info(f"✓ Best accuracy from checkpoint: {best_acc:.2%}")
        
        if 'train_acc' in checkpoint:
            logger.info(f"✓ Final train accuracy: {checkpoint['train_acc']:.2%}")
        
        if 'val_acc' in checkpoint:
            logger.info(f"✓ Final val accuracy: {checkpoint['val_acc']:.2%}")
        
        logger.info("="*70 + "\n")
        
        # Load model
        from training.retrain_video_model_FIXED import ImprovedVideoLSTM
        
        model = ImprovedVideoLSTM(input_dim=1, hidden_dim=128, num_layers=2)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        
        logger.info("✅ Model loaded successfully")
        return model
    
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        raise


# ============================================
# ✅ OPTICAL FLOW EXTRACTION
# ============================================
def extract_optical_flow(video_path, num_frames=8):
    """Extract optical flow from video"""
    try:
        cap = cv2.VideoCapture(str(video_path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total < 2:
            return None
        
        frame_indices = np.linspace(0, total-1, num_frames, dtype=int)
        prev_gray = None
        flow_features = []
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            
            if not ret:
                continue
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            if prev_gray is not None:
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, gray, None, 
                    0.5, 3, 15, 3, 5, 1.2, 0
                )
                mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                flow_features.append(np.mean(mag) / 50.0)
            else:
                flow_features.append(0.0)
            
            prev_gray = gray
        
        cap.release()
        
        while len(flow_features) < num_frames:
            flow_features.append(0.0)
        
        return np.array(flow_features[:num_frames], dtype=np.float32)
    
    except Exception as e:
        logger.warning(f"Flow extraction failed: {e}")
        return None


# ============================================
# ✅ EVALUATE MODEL
# ============================================
def evaluate(model, test_dir, device='cpu'):
    """Evaluate on test set"""
    logger.info(f"Evaluating on: {test_dir}\n")
    
    predictions = []
    confidences = []
    true_labels = []
    video_names = []
    
    # Collect test videos
    test_videos = []
    
    real_dir = Path(test_dir) / 'REAL'
    if real_dir.exists():
        for vid in sorted(real_dir.glob('*')):
            if vid.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
                test_videos.append((vid, 0))
    
    fake_dir = Path(test_dir) / 'FAKE'
    if fake_dir.exists():
        for vid in sorted(fake_dir.glob('*')):
            if vid.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
                test_videos.append((vid, 1))
    
    logger.info(f"Testing on {len(test_videos)} videos\n")
    
    for video_path, true_label in tqdm(test_videos, desc="Evaluating"):
        try:
            flow = extract_optical_flow(video_path)
            
            if flow is None:
                continue
            
            flow_tensor = torch.from_numpy(flow).float().unsqueeze(1).unsqueeze(0).to(device)
            
            with torch.no_grad():
                logits = model(flow_tensor)
                probs = torch.softmax(logits, dim=1)
                confidence, pred_class = torch.max(probs, dim=1)
            
            predictions.append(pred_class.item())
            confidences.append(confidence.item())
            true_labels.append(true_label)
            video_names.append(video_path.name)
        
        except Exception as e:
            logger.warning(f"Error processing {video_path}: {e}")
    
    predictions = np.array(predictions)
    confidences = np.array(confidences)
    true_labels = np.array(true_labels)
    
    return predictions, confidences, true_labels, video_names


# ============================================
# ✅ PRINT DETAILED RESULTS
# ============================================
def print_results(predictions, confidences, true_labels, video_names):
    """Print comprehensive results"""
    
    metrics = {
        'accuracy': accuracy_score(true_labels, predictions),
        'precision': precision_score(true_labels, predictions, zero_division=0),
        'recall': recall_score(true_labels, predictions, zero_division=0),
        'f1': f1_score(true_labels, predictions, zero_division=0),
        'roc_auc': roc_auc_score(true_labels, confidences) if len(np.unique(true_labels)) > 1 else 0.0,
    }
    
    cm = confusion_matrix(true_labels, predictions)
    tn, fp, fn, tp = cm.ravel()
    
    logger.info("\n" + "="*70)
    logger.info("✅ VIDEO MODEL EVALUATION RESULTS")
    logger.info("="*70)
    
    logger.info(f"\n📊 OVERALL METRICS:")
    logger.info(f"  Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    logger.info(f"  Precision: {metrics['precision']:.4f}")
    logger.info(f"  Recall:    {metrics['recall']:.4f}")
    logger.info(f"  F1-Score:  {metrics['f1']:.4f}")
    logger.info(f"  ROC-AUC:   {metrics['roc_auc']:.4f} ⭐ (should be > 0.85)")
    
    logger.info(f"\n📈 CONFUSION MATRIX:")
    logger.info(f"  TP (Correct FAKE):  {tp}")
    logger.info(f"  TN (Correct REAL):  {tn}")
    logger.info(f"  FP (False FAKE):    {fp}")
    logger.info(f"  FN (False REAL):    {fn}")
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    logger.info(f"\n🎯 PER-CLASS PERFORMANCE:")
    logger.info(f"  Sensitivity (Recall): {sensitivity:.2%} ⭐ (should be > 85%)")
    logger.info(f"  Specificity:          {specificity:.2%} ⭐ (should be > 85%)")
    
    # ✅ FIX: Show problem cases
    logger.info(f"\n❌ MISCLASSIFICATIONS:")
    logger.info(f"  False Positives (Real→Fake): {fp}")
    logger.info(f"  False Negatives (Fake→Real): {fn} ⭐ (PROBLEM!)")
    
    # Identify specific problem videos
    if fn > 0:
        logger.info(f"\n  FAKE videos wrongly detected as REAL:")
        for i, (pred, true_label, video_name) in enumerate(zip(predictions, true_labels, video_names)):
            if true_label == 1 and pred == 0:  # FAKE but predicted REAL
                conf = confidences[i]
                logger.info(f"    • {video_name} (conf: {conf:.2%})")
    
    logger.info("="*70 + "\n")
    
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='checkpoints/video/best_model.pth')
    parser.add_argument('--test-dir', default='dataset/video/test')
    parser.add_argument('--device', default='cpu')
    
    args = parser.parse_args()
    
    # Load model
    model = load_model(args.model, args.device)
    
    # Evaluate
    predictions, confidences, true_labels, video_names = evaluate(
        model, args.test_dir, args.device
    )
    
    # Results
    metrics = print_results(predictions, confidences, true_labels, video_names)


if __name__ == '__main__':
    main()
