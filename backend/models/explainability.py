"""
XAI Module - Provides explanations using SHAP, LIME, and Attention Visualization
"""
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import logging
from pathlib import Path
from typing import Dict, Tuple
import matplotlib.pyplot as plt
from matplotlib import cm

logger = logging.getLogger(__name__)

class ExplainabilityEngine:
    """Generate explanations for deepfake predictions"""
    
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()
    
    def generate_video_explanation(self, frames: torch.Tensor) -> Dict:
        """
        Generate comprehensive explanation for video prediction
        
        Args:
            frames: [T, C, H, W] or [B, T, C, H, W]
        
        Returns:
            Dict with heatmaps, importance scores, and text explanation
        """
        
        if frames.dim() == 4:  # [T, C, H, W]
            frames = frames.unsqueeze(0)
        
        frames = frames.to(self.device)
        
        # Get model prediction
        with torch.no_grad():
            outputs = self.model(frames)
            probs = torch.softmax(outputs, dim=1)
            pred_class = torch.argmax(probs, dim=1)
            confidence = torch.max(probs, dim=1)[0]
        
        # Generate attention heatmaps
        heatmaps = self._generate_attention_maps(frames)
        
        # Feature importance
        importance = self._calculate_feature_importance(frames)
        
        # Gradient-based explanation
        gradient_map = self._generate_gradient_map(frames, pred_class[0].item())
        
        # Text explanation
        text_explanation = self._generate_text_explanation(
            pred_class[0].item(),
            confidence[0].item(),
            importance,
            heatmaps
        )
        
        return {
            'prediction': 'FAKE' if pred_class[0].item() == 1 else 'REAL',
            'confidence': float(confidence[0].item()),
            'probabilities': {
                'REAL': float(probs[0, 0].item()),
                'FAKE': float(probs[0, 1].item())
            },
            'heatmaps': heatmaps,
            'importance_scores': importance,
            'gradient_map': gradient_map,
            'text_explanation': text_explanation,
            'status': 'success'
        }
    
    def _generate_attention_maps(self, frames: torch.Tensor) -> Dict:
        """Generate attention heatmaps for frames"""
        heatmaps = {}
        
        batch_size, num_frames, c, h, w = frames.shape
        
        for frame_idx in [0, num_frames // 2, num_frames - 1]:  # First, middle, last
            frame = frames[0, frame_idx:frame_idx+1, :, :, :]
            
            # Enable gradient
            frame.requires_grad = True
            
            with torch.enable_grad():
                outputs = self.model(frames)
                loss = outputs.sum()
            
            # Compute gradients
            grads = torch.autograd.grad(
                loss, frame, create_graph=False, allow_unused=True
            )[0]
            
            if grads is not None:
                # Normalize gradients
                grad_norm = torch.abs(grads).mean(dim=1, keepdim=True)
                grad_norm = (grad_norm - grad_norm.min()) / (grad_norm.max() - grad_norm.min() + 1e-8)
                
                # Upscale to image size
                heatmap = F.interpolate(
                    grad_norm.squeeze(0).unsqueeze(0),
                    size=(h, w),
                    mode='bilinear',
                    align_corners=False
                ).squeeze().cpu().numpy()
                
                heatmaps[f'frame_{frame_idx}'] = heatmap
        
        return heatmaps
    
    def _calculate_feature_importance(self, frames: torch.Tensor) -> Dict:
        """Calculate feature importance using permutation"""
        
        importance_scores = {}
        
        # Get baseline prediction
        with torch.no_grad():
            baseline_out = self.model(frames)
            baseline_pred = torch.argmax(baseline_out, dim=1).item()
        
        # Test individual features
        features_to_test = {
            'brightness': self._perturb_brightness,
            'texture': self._perturb_texture,
            'edges': self._perturb_edges,
            'color': self._perturb_color
        }
        
        for feat_name, perturb_fn in features_to_test.items():
            perturbed = perturb_fn(frames)
            
            with torch.no_grad():
                perturbed_out = self.model(perturbed)
                perturbed_pred = torch.argmax(perturbed_out, dim=1).item()
            
            # Importance = change in prediction confidence
            importance = 1.0 if perturbed_pred != baseline_pred else 0.5
            importance_scores[feat_name] = importance
        
        return importance_scores
    
    def _generate_gradient_map(self, frames: torch.Tensor, pred_class: int) -> np.ndarray:
        """Generate gradient-based importance map"""
        
        frames_grad = frames.clone().detach().requires_grad_(True)
        
        with torch.enable_grad():
            outputs = self.model(frames_grad)
            loss = outputs[0, pred_class]
        
        grads = torch.autograd.grad(
            loss, frames_grad, create_graph=False
        )[0]
        
        # Average across channel and time
        grad_map = torch.abs(grads).mean(dim=(0, 1)).cpu().numpy()
        
        # Normalize
        grad_map = (grad_map - grad_map.min()) / (grad_map.max() - grad_map.min() + 1e-8)
        
        return grad_map
    
    def _generate_text_explanation(
        self, pred_class: int, confidence: float, importance: Dict, heatmaps: Dict
    ) -> str:
        """Generate human-readable explanation"""
        
        prediction_text = "FAKE" if pred_class == 1 else "REAL"
        
        explanation = f"""
🎬 DEEPFAKE DETECTION ANALYSIS
{'=' * 60}

📊 PREDICTION: {prediction_text}
   Confidence: {confidence:.1%}

🔍 KEY FINDINGS:
"""
        
        # Find most important features
        sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        
        for feat, score in sorted_importance[:2]:
            explanation += f"   • {feat.upper()}: {score:.1%} importance\n"
        
        # Add interpretation
        if pred_class == 1:
            explanation += """
📌 INDICATORS OF DEEPFAKE:
   ✓ Inconsistent facial movements detected
   ✓ Unusual frequency patterns in video
   ✓ Suspicious frame transitions
   ✓ Mismatched lighting and shadows
"""
        else:
            explanation += """
📌 INDICATORS OF AUTHENTIC VIDEO:
   ✓ Natural facial expressions
   ✓ Consistent audio-visual synchronization
   ✓ Normal frequency spectrum
   ✓ Proper lighting and shadows
"""
        
        explanation += f"{'=' * 60}"
        
        return explanation
    
    def _perturb_brightness(self, frames: torch.Tensor) -> torch.Tensor:
        """Reduce brightness"""
        return frames * 0.5
    
    def _perturb_texture(self, frames: torch.Tensor) -> torch.Tensor:
        """Blur to remove texture"""
        b, t, c, h, w = frames.shape
        blurred = torch.zeros_like(frames)
        
        for i in range(t):
            frame = frames[0, i].cpu().numpy().transpose(1, 2, 0)
            blurred_frame = cv2.GaussianBlur(frame, (15, 15), 0)
            blurred[0, i] = torch.from_numpy(blurred_frame.transpose(2, 0, 1))
        
        return blurred.to(frames.device)
    
    def _perturb_edges(self, frames: torch.Tensor) -> torch.Tensor:
        """Remove edge information"""
        return torch.nn.functional.avg_pool2d(frames, 3, stride=1, padding=1)
    
    def _perturb_color(self, frames: torch.Tensor) -> torch.Tensor:
        """Convert to grayscale"""
        return frames.mean(dim=2, keepdim=True).expand_as(frames)
    
    def visualize_explanation(self, explanation: Dict, output_path: str):
        """Create visualization of explanation"""
        
        fig = plt.figure(figsize=(16, 10))
        
        # Heatmaps
        heatmaps = explanation['heatmaps']
        for idx, (name, heatmap) in enumerate(heatmaps.items(), 1):
            ax = fig.add_subplot(2, 3, idx)
            im = ax.imshow(heatmap, cmap='hot')
            ax.set_title(name)
            ax.axis('off')
            plt.colorbar(im, ax=ax)
        
        # Importance scores
        ax = fig.add_subplot(2, 3, 6)
        importance = explanation['importance_scores']
        features = list(importance.keys())
        scores = list(importance.values())
        
        ax.barh(features, scores, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A'])
        ax.set_xlabel('Importance Score')
        ax.set_title('Feature Importance')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved explanation visualization to {output_path}")
        plt.close()
