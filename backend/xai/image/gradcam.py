import torch
import cv2
import numpy as np

class GradCAM:

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_tensor, original_image):

        self.model.zero_grad()

        output = self.model(input_tensor)
        class_idx = torch.argmax(output, dim=1).item()

        loss = output[0, class_idx]
        loss.backward()

        gradients = self.gradients[0]
        activations = self.activations[0]

        weights = torch.mean(gradients, dim=(1, 2))

        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)

        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = torch.relu(cam)
        cam = cam.detach().numpy()

        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        cam = cv2.resize(cam, (original_image.shape[1], original_image.shape[0]))

        heatmap = cv2.applyColorMap(
            np.uint8(255 * cam),
            cv2.COLORMAP_JET
        )

        overlay = 0.6 * heatmap + original_image

        return overlay.astype("uint8")