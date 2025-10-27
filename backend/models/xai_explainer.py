"""
Explainable AI (XAI) Module for Deepfake Detection
Provides interpretability through Grad-CAM, attention visualization, and feature importance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Dict, List, Tuple, Union, Optional, Any
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.cm as cm

logger = logging.getLogger(__name__)

class XAIExplainer:
    """
    Explainable AI module for deepfake detection models
    Provides various interpretation methods for model predictions
    """

    def __init__(self, device: str = 'cpu'):
        """
        Initialize XAI Explainer

        Args:
            device: Device to run computations on
        """
        self.device = torch.device(device)
        self.gradients = {}
        self.activations = {}
        self.hooks = []

    def register_hooks(self, model: nn.Module, layer_names: List[str]) -> None:
        """
        Register forward and backward hooks for gradient computation

        Args:
            model: Model to register hooks on
            layer_names: Names of layers to hook
        """
        self.clear_hooks()

        for name, module in model.named_modules():
            if name in layer_names:
                # Forward hook for activations
                handle_f = module.register_forward_hook(
                    lambda module, input, output, name=name: 
                    self._save_activation(name, output)
                )

                # Backward hook for gradients
                handle_b = module.register_full_backward_hook(
                    lambda module, grad_input, grad_output, name=name:
                    self._save_gradient(name, grad_output)
                )

                self.hooks.extend([handle_f, handle_b])

    def clear_hooks(self) -> None:
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        self.gradients.clear()
        self.activations.clear()

    def _save_activation(self, name: str, activation: torch.Tensor) -> None:
        """Save activation for later use"""
        self.activations[name] = activation.detach()

    def _save_gradient(self, name: str, grad: Tuple[torch.Tensor, ...]) -> None:
        """Save gradient for later use"""
        if grad[0] is not None:
            self.gradients[name] = grad[0].detach()

    def generate_gradcam(self, model: nn.Module, input_data: torch.Tensor,
                        target_class: Optional[int] = None,
                        layer_name: str = 'features') -> np.ndarray:
        """
        Generate Grad-CAM heatmap

        Args:
            model: Model to analyze
            input_data: Input data
            target_class: Target class for gradient computation
            layer_name: Layer name for Grad-CAM

        Returns:
            Grad-CAM heatmap as numpy array
        """
        model.eval()

        # Register hooks
        self.register_hooks(model, [layer_name])

        # Forward pass
        input_data = input_data.to(self.device)
        input_data.requires_grad_(True)

        outputs = model(input_data)

        if isinstance(outputs, dict):
            logits = outputs.get('logits', outputs.get('probabilities', None))
        else:
            logits = outputs

        if logits is None:
            raise ValueError("Could not extract logits from model output")

        # Get target class
        if target_class is None:
            target_class = torch.argmax(logits, dim=1)[0].item()

        # Backward pass
        model.zero_grad()
        class_score = logits[0, target_class]
        class_score.backward()

        # Get gradients and activations
        if layer_name not in self.gradients or layer_name not in self.activations:
            logger.warning(f"Layer {layer_name} not found. Available layers: {list(self.gradients.keys())}")
            return np.zeros((224, 224))

        gradients = self.gradients[layer_name]
        activations = self.activations[layer_name]

        # Compute weights
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)

        # Generate heatmap
        heatmap = torch.sum(weights * activations, dim=1, keepdim=True)
        heatmap = F.relu(heatmap)

        # Normalize
        heatmap = heatmap.squeeze().cpu().numpy()
        if heatmap.max() > 0:
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

        self.clear_hooks()

        return heatmap

    def generate_attention_map(self, model: nn.Module, input_data: torch.Tensor,
                             attention_layer: str = 'attention') -> Dict[str, np.ndarray]:
        """
        Generate attention maps from transformer models

        Args:
            model: Model with attention mechanisms
            input_data: Input data
            attention_layer: Name of attention layer

        Returns:
            Dictionary with attention maps
        """
        model.eval()
        attention_maps = {}

        # Hook to capture attention weights
        def attention_hook(module, input, output):
            if isinstance(output, tuple) and len(output) >= 2:
                # Multi-head attention returns (output, attention_weights)
                attention_weights = output[1]
                if attention_weights is not None:
                    attention_maps['attention_weights'] = attention_weights.detach().cpu().numpy()

        # Register hook
        for name, module in model.named_modules():
            if attention_layer in name and hasattr(module, 'forward'):
                handle = module.register_forward_hook(attention_hook)
                break
        else:
            logger.warning(f"Attention layer {attention_layer} not found")
            return {}

        # Forward pass
        with torch.no_grad():
            input_data = input_data.to(self.device)
            _ = model(input_data)

        # Remove hook
        handle.remove()

        return attention_maps

    def generate_integrated_gradients(self, model: nn.Module, input_data: torch.Tensor,
                                    baseline: Optional[torch.Tensor] = None,
                                    steps: int = 50,
                                    target_class: Optional[int] = None) -> np.ndarray:
        """
        Generate Integrated Gradients explanation

        Args:
            model: Model to analyze
            input_data: Input data
            baseline: Baseline input (default: zeros)
            steps: Number of integration steps
            target_class: Target class

        Returns:
            Integrated gradients as numpy array
        """
        model.eval()

        input_data = input_data.to(self.device)

        if baseline is None:
            baseline = torch.zeros_like(input_data)
        else:
            baseline = baseline.to(self.device)

        # Generate interpolated inputs
        alphas = torch.linspace(0, 1, steps).to(self.device)
        interpolated_inputs = []

        for alpha in alphas:
            interpolated = baseline + alpha * (input_data - baseline)
            interpolated_inputs.append(interpolated)

        # Compute gradients for each interpolated input
        integrated_grads = torch.zeros_like(input_data)

        for interpolated_input in interpolated_inputs:
            interpolated_input.requires_grad_(True)

            outputs = model(interpolated_input)
            if isinstance(outputs, dict):
                logits = outputs.get('logits', outputs.get('probabilities'))
            else:
                logits = outputs

            if target_class is None:
                target_class = torch.argmax(logits, dim=1)[0].item()

            class_score = logits[0, target_class]

            # Compute gradients
            grad = torch.autograd.grad(class_score, interpolated_input,
                                     retain_graph=False, create_graph=False)[0]
            integrated_grads += grad

        # Average gradients and multiply by input difference
        integrated_grads = integrated_grads / steps
        integrated_grads = integrated_grads * (input_data - baseline)

        return integrated_grads.squeeze().cpu().numpy()

    def generate_lime_explanation(self, model: nn.Module, input_data: torch.Tensor,
                                num_samples: int = 1000,
                                num_features: int = 100) -> Dict[str, Any]:
        """
        Generate LIME explanation (simplified version)

        Args:
            model: Model to analyze
            input_data: Input data
            num_samples: Number of samples for LIME
            num_features: Number of features to explain

        Returns:
            LIME explanation dictionary
        """
        model.eval()

        input_data = input_data.to(self.device)
        batch_size, channels, height, width = input_data.shape

        # Get original prediction
        with torch.no_grad():
            original_output = model(input_data)
            if isinstance(original_output, dict):
                original_probs = original_output.get('probabilities', original_output.get('logits'))
            else:
                original_probs = F.softmax(original_output, dim=1)
            original_pred = torch.argmax(original_probs, dim=1)[0].item()

        # Generate superpixels (simplified: use grid)
        grid_size = int(np.sqrt(num_features))
        segment_height = height // grid_size
        segment_width = width // grid_size

        # Create perturbation mask
        perturbations = []
        predictions = []

        for _ in range(num_samples):
            # Random binary mask for segments
            mask = np.random.randint(0, 2, (grid_size, grid_size))

            # Create full image mask
            full_mask = np.zeros((height, width))
            for i in range(grid_size):
                for j in range(grid_size):
                    y_start = i * segment_height
                    y_end = min((i + 1) * segment_height, height)
                    x_start = j * segment_width
                    x_end = min((j + 1) * segment_width, width)
                    full_mask[y_start:y_end, x_start:x_end] = mask[i, j]

            # Apply mask to input
            perturbed_input = input_data.clone()
            for c in range(channels):
                perturbed_input[0, c] = perturbed_input[0, c] * torch.tensor(full_mask).to(self.device)

            # Get prediction
            with torch.no_grad():
                output = model(perturbed_input)
                if isinstance(output, dict):
                    probs = output.get('probabilities', output.get('logits'))
                else:
                    probs = F.softmax(output, dim=1)

                predictions.append(probs[0, original_pred].item())
                perturbations.append(mask.flatten())

        perturbations = np.array(perturbations)
        predictions = np.array(predictions)

        # Fit linear model (simplified)
        from sklearn.linear_model import Ridge

        ridge = Ridge(alpha=1.0)
        ridge.fit(perturbations, predictions)

        # Get feature importance
        feature_importance = ridge.coef_

        # Reshape back to grid
        importance_grid = feature_importance.reshape((grid_size, grid_size))

        return {
            'feature_importance': importance_grid,
            'local_prediction': original_pred,
            'explanation_fit': ridge.score(perturbations, predictions),
            'num_samples': num_samples
        }

    def visualize_explanation(self, image: np.ndarray, 
                            explanation: np.ndarray,
                            alpha: float = 0.4,
                            colormap: str = 'jet') -> np.ndarray:
        """
        Visualize explanation overlay on original image

        Args:
            image: Original image
            explanation: Explanation heatmap
            alpha: Overlay transparency
            colormap: Colormap for heatmap

        Returns:
            Visualization as numpy array
        """
        if len(image.shape) == 3 and image.shape[0] == 3:
            # Convert CHW to HWC
            image = np.transpose(image, (1, 2, 0))

        # Normalize image to 0-255
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)

        # Resize explanation to match image
        if explanation.shape != image.shape[:2]:
            explanation = cv2.resize(explanation, (image.shape[1], image.shape[0]))

        # Apply colormap to explanation
        cmap = plt.get_cmap(colormap)
        colored_explanation = cmap(explanation)[:, :, :3]  # Remove alpha channel
        colored_explanation = (colored_explanation * 255).astype(np.uint8)

        # Blend with original image
        blended = cv2.addWeighted(image, 1 - alpha, colored_explanation, alpha, 0)

        return blended

    def generate_comprehensive_explanation(self, model: nn.Module, 
                                         input_data: torch.Tensor,
                                         methods: List[str] = ['gradcam', 'integrated_gradients'],
                                         **kwargs) -> Dict[str, Any]:
        """
        Generate comprehensive explanation using multiple methods

        Args:
            model: Model to analyze
            input_data: Input data
            methods: List of explanation methods to use
            **kwargs: Additional arguments for specific methods

        Returns:
            Dictionary with all explanations
        """
        explanations = {}

        for method in methods:
            try:
                if method == 'gradcam':
                    explanations['gradcam'] = self.generate_gradcam(
                        model, input_data, 
                        layer_name=kwargs.get('gradcam_layer', 'features')
                    )

                elif method == 'integrated_gradients':
                    explanations['integrated_gradients'] = self.generate_integrated_gradients(
                        model, input_data,
                        steps=kwargs.get('ig_steps', 50)
                    )

                elif method == 'attention':
                    explanations['attention'] = self.generate_attention_map(
                        model, input_data,
                        attention_layer=kwargs.get('attention_layer', 'attention')
                    )

                elif method == 'lime':
                    explanations['lime'] = self.generate_lime_explanation(
                        model, input_data,
                        num_samples=kwargs.get('lime_samples', 1000)
                    )

            except Exception as e:
                logger.warning(f"Failed to generate {method} explanation: {e}")
                explanations[method] = None

        return explanations

    def generate_text_explanation(self, prediction_result: Dict[str, Any],
                                feature_breakdown: Dict[str, float],
                                confidence_threshold: float = 0.7) -> str:
        """
        Generate human-readable text explanation

        Args:
            prediction_result: Model prediction results
            feature_breakdown: Feature importance breakdown
            confidence_threshold: Confidence threshold for reliable predictions

        Returns:
            Text explanation string
        """
        label = prediction_result.get('label', 'UNKNOWN')
        confidence = prediction_result.get('confidence', 0.0)
        probabilities = prediction_result.get('probabilities', [0.5, 0.5])

        # Start explanation
        if label == 'FAKE':
            explanation = f"This content is classified as DEEPFAKE with {confidence:.1%} confidence. "
        else:
            explanation = f"This content appears to be AUTHENTIC with {confidence:.1%} confidence. "

        # Add confidence assessment
        if confidence >= confidence_threshold:
            explanation += "The model is highly confident in this prediction. "
        elif confidence >= 0.5:
            explanation += "The model has moderate confidence in this prediction. "
        else:
            explanation += "The model has low confidence in this prediction. "

        # Add feature analysis
        if feature_breakdown:
            high_features = []
            for feature, score in feature_breakdown.items():
                if score > 0.6:
                    feature_name = feature.replace('_', ' ').title()
                    high_features.append(f"{feature_name} ({score:.1%})")

            if high_features:
                explanation += f"Key indicators include: {', '.join(high_features)}. "

        # Add interpretation guidance
        if label == 'FAKE':
            explanation += "This suggests the content may have been digitally manipulated or generated using AI."
        else:
            explanation += "No significant signs of digital manipulation were detected."

        return explanation

    def save_explanation_report(self, explanations: Dict[str, Any],
                              prediction_result: Dict[str, Any],
                              output_path: Path,
                              include_visualizations: bool = True) -> None:
        """
        Save comprehensive explanation report

        Args:
            explanations: Dictionary of explanations
            prediction_result: Model prediction results
            output_path: Output file path
            include_visualizations: Whether to include visual explanations
        """
        report = {
            'prediction': prediction_result,
            'explanations': {},
            'text_explanation': self.generate_text_explanation(
                prediction_result, 
                explanations.get('feature_breakdown', {})
            )
        }

        # Process explanations
        for method, result in explanations.items():
            if result is not None:
                if isinstance(result, np.ndarray):
                    # Save numpy arrays as lists for JSON serialization
                    if include_visualizations:
                        report['explanations'][method] = {
                            'shape': result.shape,
                            'data': result.tolist()
                        }
                    else:
                        report['explanations'][method] = {
                            'shape': result.shape,
                            'summary': {
                                'min': float(result.min()),
                                'max': float(result.max()),
                                'mean': float(result.mean())
                            }
                        }
                else:
                    report['explanations'][method] = result

        # Save report
        import json
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Explanation report saved to {output_path}")

    def __del__(self):
        """Cleanup hooks when object is destroyed"""
        self.clear_hooks()
