import torch
import shap
import numpy as np
import cv2
import base64
from io import BytesIO
from PIL import Image


class ShapExplainer:

    def __init__(self, model):
        self.model = model
        self.model.eval()

        # Dummy background image
        self.background = torch.zeros((1, 3, 224, 224))

        self.explainer = shap.GradientExplainer(
            self.model,
            self.background
        )

    def explain_image(self, image_tensor):

        image_tensor.requires_grad = True

        shap_values = self.explainer.shap_values(image_tensor)

        # Take class 1 (fake) or 0 (real)
        heatmap = shap_values[0][0].mean(axis=0)

        heatmap = np.maximum(heatmap, 0)
        heatmap /= heatmap.max() + 1e-8

        heatmap = cv2.resize(heatmap, (224, 224))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        # Convert to base64
        pil_img = Image.fromarray(heatmap)
        buffer = BytesIO()
        pil_img.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode()

        return {
            "heatmap_base64": encoded
        }
