import cv2
import torch
import torch.nn as nn 
import torchvision.models as models

from xai.image.gradcam import GradCAM
from xai.audio.audio_saliency import AudioSaliency
from xai.utils.preprocessing import preprocess_image
from xai.video.video_explainer import VideoExplainer

class XAIEngine:

    def __init__(self, model):

        self.model = model
        self.model.eval()

        target_layer = self._get_target_layer(self.model)

        from xai.image.gradcam import GradCAM
        self.grad_cam = GradCAM(self.model, target_layer)

        # from xai.audio.audio_saliency import AudioSaliency
        # self.audio_explainer = AudioSaliency(self.model)

    def _get_target_layer(self, model):
        """
        Automatically select last convolution layer
        Supports ResNet, EfficientNet (timm), etc.
        """

        # ✅ Case 1: ResNet
        if hasattr(model, "layer4"):
            return model.layer4[-1]

        # ✅ Case 2: EfficientNet (timm)
        if hasattr(model, "blocks"):
            return model.blocks[-1]

        # ✅ Case 3: torchvision EfficientNet
        if hasattr(model, "features"):
            return model.features[-1]

        # ✅ Fallback: last Conv2d
        last_conv = None
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                last_conv = module

        if last_conv is None:
            raise ValueError("No Conv2d layer found for GradCAM")

        return last_conv

    # =====================================
    # IMAGE (numpy array from frontend)
    # =====================================
    def explain_image(self, image_np):

        image = cv2.resize(image_np, (224, 224))
        input_tensor = preprocess_image(image)

        heatmap = self.grad_cam.generate(input_tensor, image)

        return heatmap

    # =====================================
    # VIDEO (path from uploaded temp file)
    # =====================================
    def explain_video(self, video_path, max_frames=5):

        cap = cv2.VideoCapture(video_path)
        results = []
        frame_count = 0

        while cap.isOpened() and frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (224, 224))
            input_tensor = preprocess_image(frame)

            heatmap = self.grad_cam.generate(input_tensor, frame)
            results.append(heatmap)

            frame_count += 1

        cap.release()

        return results

    # =====================================
    # AUDIO (.flac / .wav)
    # =====================================
    def explain_audio(self, audio_path):

        return self.audio_explainer.generate(audio_path)