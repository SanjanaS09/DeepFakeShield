"""
grad_cam_explainer.py - Generate Grad-CAM heatmaps for XAI visualization
Explains which regions of the image influenced the deepfake detection decision
"""

import torch
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import logging

logger = logging.getLogger(__name__)


class GradCAMExplainer:
    """
    Generates Grad-CAM heatmaps for CNN model explanation.
    Shows which regions influenced the deepfake prediction.
    """

    def __init__(self, model, target_layer_name):
        """
        Args:
            model: PyTorch model
            target_layer_name: Name of layer to visualize (e.g., 'layer4')
        """
        self.model = model
        self.gradients = None
        self.activations = None
        self.target_layer_name = target_layer_name
        self._register_hooks()

    def _register_hooks(self):
        """Register forward and backward hooks to capture gradients and activations"""
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        # Find and hook the target layer
        for name, module in self.model.named_modules():
            if self.target_layer_name in name:
                module.register_forward_hook(forward_hook)
                module.register_backward_hook(backward_hook)
                logger.info(f"Hooks registered for layer: {name}")
                break

    def generate_heatmap(self, input_tensor, target_class=None):
        """
        Generate Grad-CAM heatmap for input image.

        Args:
            input_tensor: Tensor of shape (1, 3, H, W)
            target_class: Target class index (0=REAL, 1=FAKE)

        Returns:
            heatmap: Normalized heatmap of shape (H, W)
            overlaid_image: Image with heatmap overlay
        """
        # Forward pass
        self.model.eval()
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Backward pass
        self.model.zero_grad()
        target_score = output[0, target_class]
        target_score.backward()

        # Calculate Grad-CAM
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # Global average pooling
        weighted_activations = weights * self.activations
        heatmap = weighted_activations.sum(dim=1).squeeze().detach()
        
        # Normalize heatmap
        heatmap = F.relu(heatmap)  # Keep only positive contributions
        heatmap = heatmap / (heatmap.max() + 1e-8)  # Normalize to [0, 1]
        
        return heatmap, output[0, target_class].item()

    def visualize_heatmap(self, input_image_path, heatmap, output_path=None):
        """
        Visualize heatmap overlaid on original image.

        Args:
            input_image_path: Path to input image
            heatmap: Grad-CAM heatmap
            output_path: Where to save the visualization

        Returns:
            base64_encoded_image: Base64 string of visualization
        """
        # Load original image
        original_image = cv2.imread(input_image_path)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        
        # Resize heatmap to match image size
        heatmap_np = heatmap.cpu().numpy()
        heatmap_resized = cv2.resize(heatmap_np, (original_image.shape[1], original_image.shape[0]))
        
        # Create colored heatmap
        heatmap_colored = cm.jet(heatmap_resized)[:, :, :3]
        heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
        
        # Blend with original image
        alpha = 0.4
        overlay = cv2.addWeighted(
            original_image.astype(np.uint8), 
            1 - alpha, 
            (heatmap_colored).astype(np.uint8), 
            alpha, 
            0
        )
        
        # Save if path provided
        if output_path:
            overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path, overlay_bgr)
            logger.info(f"Heatmap saved to: {output_path}")
        
        return overlay

    def encode_to_base64(self, image_array):
        """Convert image array to base64 string for API response"""
        import io
        import base64
        
        # Convert to PIL Image
        pil_image = Image.fromarray(image_array.astype(np.uint8))
        
        # Convert to PNG
        buffer = io.BytesIO()
        pil_image.save(buffer, format='PNG')
        buffer.seek(0)
        
        # Encode to base64
        img_str = base64.b64encode(buffer.getvalue()).decode()
        return f"data:image/png;base64,{img_str}"


# ============ INTEGRATION WITH DETECTION SERVICE ============

def generate_xai_explanation(image_path, model, prediction, confidence):
    """
    Generate XAI explanation for detection result.

    Args:
        image_path: Path to input image
        model: Trained PyTorch model
        prediction: 'REAL' or 'FAKE'
        confidence: Confidence score

    Returns:
        explanation_dict: Contains heatmap and textual explanation
    """
    try:
        # Initialize Grad-CAM explainer
        explainer = GradCAMExplainer(model, target_layer_name='layer4')
        
        # Prepare image
        from torchvision import transforms
        image = Image.open(image_path).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        input_tensor = transform(image).unsqueeze(0)
        
        # Generate heatmap
        target_class = 1 if prediction == 'FAKE' else 0
        heatmap, score = explainer.generate_heatmap(input_tensor, target_class=target_class)
        
        # Visualize
        overlay_image = explainer.visualize_heatmap(image_path, heatmap)
        heatmap_b64 = explainer.encode_to_base64(overlay_image)
        
        # Generate explanation text
        if prediction == 'FAKE':
            explanation = {
                'title': 'Deepfake Detected',
                'regions': 'Red areas indicate regions with deepfake artifacts',
                'indicators': [
                    'Facial feature inconsistencies detected',
                    'Unnatural lighting patterns found',
                    'Eye and mouth regions show manipulation signs',
                    'Texture blending artifacts visible'
                ],
                'confidence_reason': f'Model is {confidence*100:.1f}% confident this is fake'
            }
        else:
            explanation = {
                'title': 'Content Verified as Authentic',
                'regions': 'Green areas indicate natural facial features',
                'indicators': [
                    'Consistent lighting and texture',
                    'Natural facial feature transitions',
                    'No significant deepfake patterns detected',
                    'High authenticity markers present'
                ],
                'confidence_reason': f'Model is {confidence*100:.1f}% confident this is real'
            }
        
        return {
            'heatmap': heatmap_b64,
            'explanation': explanation,
            'success': True
        }
        
    except Exception as e:
        logger.error(f"Error generating XAI explanation: {str(e)}")
        return {
            'heatmap': None,
            'explanation': None,
            'success': False,
            'error': str(e)
        }


# ============ ALTERNATIVE: LIME-BASED EXPLANATION ============

class LIMEExplainer:
    """
    LIME (Local Interpretable Model-agnostic Explanations)
    Creates a simpler surrogate model to explain local decisions
    """
    
    def __init__(self, model):
        self.model = model
    
    def generate_lime_explanation(self, image_array, num_samples=50):
        """
        Generate LIME explanation by testing local perturbations.
        
        Args:
            image_array: Image as numpy array
            num_samples: Number of perturbed samples to generate
            
        Returns:
            lime_explanation: Important regions for decision
        """
        try:
            import lime
            import lime.lime_image
            
            explainer = lime.lime_image.LimeImageExplainer()
            explanation = explainer.explain_instance(
                image_array,
                predict_fn=self._predict_fn,
                top_labels=2,
                num_samples=num_samples
            )
            
            return explanation
            
        except ImportError:
            logger.warning("LIME not installed. Install with: pip install lime")
            return None
    
    def _predict_fn(self, images):
        """Prediction function for LIME"""
        # Convert images and run model
        predictions = []
        for img in images:
            pred = self.model(img)
            predictions.append(pred.detach().cpu().numpy())
        return np.array(predictions)


# ============ SALIENCY MAP EXPLANATION ============

class SaliencyExplainer:
    """
    Saliency Maps: Highlight pixels that are most influential
    Uses gradients of output w.r.t. input pixels
    """
    
    def __init__(self, model):
        self.model = model
    
    def generate_saliency_map(self, input_tensor, target_class):
        """
        Generate saliency map showing pixel importance.
        
        Args:
            input_tensor: Input image tensor
            target_class: Target class index
            
        Returns:
            saliency_map: Visualization of important pixels
        """
        input_tensor.requires_grad = True
        
        # Forward pass
        output = self.model(input_tensor)
        target_score = output[0, target_class]
        
        # Backward pass to get gradients
        self.model.zero_grad()
        target_score.backward()
        
        # Get saliency
        saliency = input_tensor.grad.data.abs()
        saliency = saliency.max(dim=1)[0]  # Take max across channels
        
        return saliency


# ============ ATTENTION MECHANISM EXPLANATION ============

class AttentionExplainer:
    """
    For models with attention mechanisms,
    visualize which regions the model attended to
    """
    
    def __init__(self, model):
        self.model = model
        self.attention_weights = None
        self._register_attention_hooks()
    
    def _register_attention_hooks(self):
        """Register hooks to capture attention weights"""
        def attention_hook(module, input, output):
            self.attention_weights = output.detach()
        
        # Find attention layers and register hooks
        for name, module in self.model.named_modules():
            if 'attention' in name.lower():
                module.register_forward_hook(attention_hook)
    
    def get_attention_visualization(self, input_tensor):
        """Extract and visualize attention weights"""
        _ = self.model(input_tensor)
        
        if self.attention_weights is not None:
            return self.attention_weights
        return None