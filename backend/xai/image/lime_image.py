from lime import lime_image
import torch
import numpy as np

class ImageLIME:
    def __init__(self, model, device="cpu"):
        self.model = model
        self.device = device
        self.explainer = lime_image.LimeImageExplainer()

    def predict(self, images):
        images = torch.tensor(images).permute(0,3,1,2).float().to(self.device)
        outputs = self.model(images)
        return torch.softmax(outputs, dim=1).detach().cpu().numpy()

    def explain(self, image_np):
        explanation = self.explainer.explain_instance(
            image_np,
            self.predict,
            top_labels=2,
            num_samples=500
        )
        return explanation