import librosa
import numpy as np
import cv2
import torch
from xai.utils.preprocessing import preprocess_image

class AudioSaliency:

    def __init__(self, model):
        self.model = model

    def generate(self, audio_path):

        y, sr = librosa.load(audio_path)

        S = librosa.feature.melspectrogram(y=y, sr=sr)
        S_db = librosa.power_to_db(S, ref=np.max)

        spec_img = (S_db - S_db.min()) / (S_db.max() - S_db.min())
        spec_img = (spec_img * 255).astype(np.uint8)

        spec_img = cv2.resize(spec_img, (224, 224))
        spec_img = cv2.cvtColor(spec_img, cv2.COLOR_GRAY2BGR)

        input_tensor = preprocess_image(spec_img)
        input_tensor.requires_grad = True

        output = self.model(input_tensor)
        class_idx = torch.argmax(output)

        output[0, class_idx].backward()

        saliency = input_tensor.grad.data.abs().squeeze().numpy()
        saliency = np.max(saliency, axis=0)

        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min())

        return spec_img, saliency