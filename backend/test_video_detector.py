"""
✅ TEST VIDEO DETECTOR + XAI VALIDATION
Tests trained model and verifies XAI explanations are working
"""
import os
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).parent
sys.path.insert(0, str(BACKEND_ROOT))

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import logging
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import transforms
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================
# ✅ LOAD MODEL
# ============================================
def load_model(checkpoint_path, device='cpu'):
    """Load trained EfficientNet video model"""
    try:
        from training.train_video_efficientnet import EfficientNetVideoModel
        
        model = EfficientNetVideoModel(num_classes=2, pretrained=False)
        model = model.to(device)
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        model.eval()
        
        logger.info(f"✅ Loaded model from {checkpoint_path}")
        return model
    
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        return None


# ============================================
# ✅ EXTRACT FRAMES & PREDICT
# ============================================
def extract_frames(video_path, num_frames=8, frame_size=(224, 224)):
    """Extract frames from video"""
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        logger.warning(f"No frames in {video_path}")
        return None
    
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = []
    
    transform = transforms.Compose([
        transforms.Resize(frame_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            frame_tensor = transform(frame_pil)
            frames.append(frame_tensor)
    
    cap.release()
    
    if len(frames) == num_frames:
        return torch.stack(frames)
    
    return None


def predict_video(model, video_path, device='cpu'):
    """Predict if video is deepfake"""
    frames = extract_frames(video_path)
    
    if frames is None:
        return {'error': 'Failed to extract frames'}
    
    frames = frames.unsqueeze(0).to(device)  # [1, T, C, H, W]
    
    with torch.no_grad():
        logits = model(frames)
        probs = F.softmax(logits, dim=1)
        confidence, pred_class = torch.max(probs, dim=1)
    
    return {
        'prediction': 'FAKE' if pred_class.item() == 1 else 'REAL',
        'confidence': float(confidence.item()),
        'probabilities': {
            'REAL': float(probs[0, 0].item()),
            'FAKE': float(probs[0, 1].item())
        }
    }


# ============================================
# ✅ XAI: ATTENTION VISUALIZATION
# ============================================
def visualize_attention(model, video_path, device='cpu', output_path='attention_map.png'):
    """Generate attention heatmap from model"""
    frames = extract_frames(video_path)
    
    if frames is None:
        logger.warning("Failed to extract frames")
        return None
    
    frames = frames.unsqueeze(0).to(device)  # [1, T, C, H, W]
    
    # Forward pass with gradient tracking
    frames.requires_grad_(True)
    logits = model(frames)
    
    # Get prediction
    pred_class = torch.argmax(logits, dim=1).item()
    
    # Backward to get gradients
    loss = logits[0, pred_class]
    loss.backward()
    
    # Get gradient magnitude
    grads = frames.grad.abs().mean(dim=[2, 3, 4])  # [1, T]
    attn_weights = F.softmax(grads, dim=1).detach().cpu().numpy()[0]  # [T]
    
    logger.info(f"Attention weights per frame: {attn_weights}")
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Frame importance
    ax1.bar(range(len(attn_weights)), attn_weights, color='steelblue')
    ax1.set_xlabel('Frame Index')
    ax1.set_ylabel('Attention Weight')
    ax1.set_title('Model Attention Across Frames')
    ax1.set_ylim([0, max(attn_weights) * 1.2])
    
    # Cumulative importance
    cum_weights = np.cumsum(attn_weights)
    ax2.plot(range(len(cum_weights)), cum_weights, marker='o', color='darkgreen', linewidth=2)
    ax2.set_xlabel('Frame Index')
    ax2.set_ylabel('Cumulative Attention')
    ax2.set_title('Cumulative Attention Distribution')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"✅ Saved attention map to {output_path}")
    plt.close()
    
    return attn_weights


# ============================================
# ✅ XAI: FEATURE IMPORTANCE
# ============================================
def analyze_frame_importance(model, video_path, device='cpu'):
    """Analyze which frames are most important for prediction"""
    frames = extract_frames(video_path)
    
    if frames is None:
        return None
    
    frames = frames.unsqueeze(0).to(device)  # [1, T, C, H, W]
    
    # Get baseline prediction
    with torch.no_grad():
        baseline_logits = model(frames)
        baseline_pred = torch.argmax(baseline_logits, dim=1).item()
    
    # Test importance by removing each frame
    importance_scores = []
    
    for t in range(frames.shape[1]):
        # Zero out frame
        masked_frames = frames.clone()
        masked_frames[0, t] = 0
        
        with torch.no_grad():
            masked_logits = model(masked_frames)
            masked_pred = torch.argmax(masked_logits, dim=1).item()
        
        # Importance = change in prediction
        importance = 1.0 if masked_pred != baseline_pred else 0.5
        importance_scores.append(importance)
    
    return np.array(importance_scores)


# ============================================
# ✅ EVALUATION ON TEST SET
# ============================================
def evaluate_on_testset(model, test_dir, device='cpu'):
    """Evaluate model on test dataset"""
    logger.info(f"Evaluating on test set: {test_dir}")
    
    predictions = []
    confidences = []
    true_labels = []
    video_paths = []
    
    test_dir = Path(test_dir)
    
    # Collect test videos
    videos_to_test = []
    
    real_dir = test_dir / 'REAL'
    if real_dir.exists():
        for vid in sorted(real_dir.glob('*')):
            if vid.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
                videos_to_test.append((vid, 0))
    
    fake_dir = test_dir / 'FAKE'
    if fake_dir.exists():
        for vid in sorted(fake_dir.glob('*')):
            if vid.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
                videos_to_test.append((vid, 1))
    
    logger.info(f"Testing on {len(videos_to_test)} videos")
    
    # Test each video
    for video_path, true_label in videos_to_test:
        try:
            result = predict_video(model, video_path, device)
            
            if 'error' not in result:
                pred_class = 1 if result['prediction'] == 'FAKE' else 0
                predictions.append(pred_class)
                confidences.append(result['confidence'])
                true_labels.append(true_label)
                video_paths.append(str(video_path.name))
        
        except Exception as e:
            logger.warning(f"Error predicting {video_path}: {e}")
    
    if len(predictions) == 0:
        logger.error("No successful predictions")
        return None
    
    # Calculate metrics
    predictions = np.array(predictions)
    true_labels = np.array(true_labels)
    confidences = np.array(confidences)
    
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    
    metrics = {
        'accuracy': accuracy_score(true_labels, predictions),
        'precision': precision_score(true_labels, predictions, zero_division=0),
        'recall': recall_score(true_labels, predictions, zero_division=0),
        'f1': f1_score(true_labels, predictions, zero_division=0),
        'roc_auc': roc_auc_score(true_labels, confidences) if len(np.unique(true_labels)) > 1 else 0.0,
        'confusion_matrix': confusion_matrix(true_labels, predictions),
        'n_tested': len(predictions)
    }
    
    return metrics, predictions, true_labels, confidences, video_paths


# ============================================
# ✅ GENERATE REPORT
# ============================================
def generate_test_report(model, test_dir, device='cpu', output_dir='test_results'):
    """Generate comprehensive test report"""
    
    Path(output_dir).mkdir(exist_ok=True)
    
    logger.info("\n" + "="*70)
    logger.info("🧪 VIDEO DEEPFAKE DETECTOR - TEST REPORT")
    logger.info("="*70)
    
    # Run evaluation
    result = evaluate_on_testset(model, test_dir, device)
    
    if result is None:
        logger.error("Evaluation failed")
        return
    
    metrics, predictions, true_labels, confidences, video_paths = result
    
    # Print metrics
    logger.info("\n📊 OVERALL METRICS:")
    logger.info(f"  Accuracy:  {metrics['accuracy']:.2%}")
    logger.info(f"  Precision: {metrics['precision']:.2%}")
    logger.info(f"  Recall:    {metrics['recall']:.2%}")
    logger.info(f"  F1-Score:  {metrics['f1']:.2%}")
    logger.info(f"  ROC-AUC:   {metrics['roc_auc']:.2%}")
    logger.info(f"  Videos:    {metrics['n_tested']}")
    
    # Confusion matrix
    tn, fp, fn, tp = metrics['confusion_matrix'].ravel()
    logger.info(f"\n📈 CONFUSION MATRIX:")
    logger.info(f"  TP (Correct FAKE):  {tp}")
    logger.info(f"  TN (Correct REAL):  {tn}")
    logger.info(f"  FP (False FAKE):    {fp}")
    logger.info(f"  FN (False REAL):    {fn}")
    
    # Specificity & Sensitivity
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    logger.info(f"\n🎯 PERFORMANCE ANALYSIS:")
    logger.info(f"  Sensitivity (Recall): {sensitivity:.2%}")
    logger.info(f"  Specificity:          {specificity:.2%}")
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Confusion matrix heatmap
    sns.heatmap(
        metrics['confusion_matrix'],
        annot=True,
        fmt='d',
        cmap='Blues',
        ax=axes[0, 0],
        xticklabels=['REAL', 'FAKE'],
        yticklabels=['REAL', 'FAKE']
    )
    axes[0, 0].set_title('Confusion Matrix')
    axes[0, 0].set_ylabel('True Label')
    axes[0, 0].set_xlabel('Predicted Label')
    
    # Confidence distribution
    axes[0, 1].hist(confidences[true_labels == 0], bins=20, label='REAL', alpha=0.7, color='green')
    axes[0, 1].hist(confidences[true_labels == 1], bins=20, label='FAKE', alpha=0.7, color='red')
    axes[0, 1].set_xlabel('Confidence')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Confidence Distribution')
    axes[0, 1].legend()
    
    # ROC curve
    fpr, tpr, _ = roc_curve(true_labels, confidences)
    axes[1, 0].plot(fpr, tpr, linewidth=2, label=f'ROC (AUC={metrics["roc_auc"]:.2%})')
    axes[1, 0].plot([0, 1], [0, 1], 'k--', linewidth=1)
    axes[1, 0].set_xlabel('False Positive Rate')
    axes[1, 0].set_ylabel('True Positive Rate')
    axes[1, 0].set_title('ROC Curve')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Metrics bar chart
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC']
    metric_values = [
        metrics['accuracy'],
        metrics['precision'],
        metrics['recall'],
        metrics['f1'],
        metrics['roc_auc']
    ]
    axes[1, 1].bar(metric_names, metric_values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_title('Performance Metrics')
    axes[1, 1].set_ylim([0, 1.1])
    axes[1, 1].axhline(y=0.9, color='red', linestyle='--', label='Target (90%)')
    for i, v in enumerate(metric_values):
        axes[1, 1].text(i, v + 0.02, f'{v:.2%}', ha='center', fontweight='bold')
    axes[1, 1].legend()
    
    plt.tight_layout()
    report_path = Path(output_dir) / 'test_report.png'
    plt.savefig(report_path, dpi=150, bbox_inches='tight')
    logger.info(f"✅ Saved test report to {report_path}")
    plt.close()
    
    # Summary
    logger.info("\n" + "="*70)
    if metrics['accuracy'] >= 0.90:
        logger.info("✅ MODEL PERFORMANCE: EXCELLENT (>90%)")
    elif metrics['accuracy'] >= 0.80:
        logger.info("✅ MODEL PERFORMANCE: GOOD (80-90%)")
    elif metrics['accuracy'] >= 0.70:
        logger.info("⚠️  MODEL PERFORMANCE: FAIR (70-80%)")
    else:
        logger.info("❌ MODEL PERFORMANCE: POOR (<70%)")
    logger.info("="*70 + "\n")
    
    return metrics


# ============================================
# ✅ MAIN TESTING SCRIPT
# ============================================
def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='checkpoints/video/best_model.pth')
    parser.add_argument('--test-dir', default='dataset/video/test')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--xai', action='store_true', help='Generate XAI visualizations')
    parser.add_argument('--sample-video', type=str, help='Path to single video to test with XAI')
    
    args = parser.parse_args()
    
    # Load model
    model = load_model(args.model, args.device)
    if model is None:
        logger.error("Failed to load model")
        return
    
    # Generate report
    metrics = generate_test_report(model, args.test_dir, args.device)
    
    # XAI analysis
    if args.xai and args.sample_video:
        logger.info("\n📊 GENERATING XAI VISUALIZATIONS...")
        
        # Attention map
        logger.info("\n1️⃣  Generating attention map...")
        visualize_attention(model, args.sample_video, args.device, 'attention_map.png')
        
        # Frame importance
        logger.info("\n2️⃣  Analyzing frame importance...")
        importance = analyze_frame_importance(model, args.sample_video, args.device)
        if importance is not None:
            logger.info(f"Frame importance scores: {importance}")
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(range(len(importance)), importance, color='coral')
            ax.set_xlabel('Frame Index')
            ax.set_ylabel('Importance Score')
            ax.set_title('Frame Importance Analysis')
            plt.savefig('frame_importance.png', dpi=150, bbox_inches='tight')
            logger.info("✅ Saved frame importance to frame_importance.png")
            plt.close()
        
        # Single video prediction
        logger.info("\n3️⃣  Predicting single video...")
        result = predict_video(model, args.sample_video, args.device)
        logger.info(f"Prediction: {result['prediction']} (Confidence: {result['confidence']:.2%})")
        logger.info(f"Probabilities: REAL={result['probabilities']['REAL']:.2%}, FAKE={result['probabilities']['FAKE']:.2%}")


if __name__ == '__main__':
    main()
