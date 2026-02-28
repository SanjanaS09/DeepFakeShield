import librosa
import numpy as np
import cv2
import torch

class AudioSaliency:

    def __init__(self, model, processor, device):
        self.model = model
        self.processor = processor
        self.device = device

    def generate(self, audio_path):

        y, sr = librosa.load(audio_path, sr=16000)

        inputs = self.processor(
            y,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        )

        input_values = inputs.input_values.to(self.device)
        input_values.requires_grad = True

        # Forward pass
        logits = self.model(input_values)

        # 🔥 Binary model → backprop directly
        logits.backward(torch.ones_like(logits))

        saliency = input_values.grad.data.abs().squeeze().cpu().numpy()

        denom = saliency.max() - saliency.min()
        if denom != 0:
            saliency = (saliency - saliency.min()) / denom
        else:
            saliency = np.zeros_like(saliency)

        # Create spectrogram
        S = librosa.feature.melspectrogram(y=y, sr=sr)
        S_db = librosa.power_to_db(S, ref=np.max)

        spec_img = (S_db - S_db.min()) / (S_db.max() - S_db.min())
        spec_img = (spec_img * 255).astype(np.uint8)
        spec_img = cv2.resize(spec_img, (224, 224))
        spec_img = cv2.cvtColor(spec_img, cv2.COLOR_GRAY2BGR)

        saliency_resized = cv2.resize(saliency, (224, 224))

        return spec_img, saliency_resized