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

        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)

        # =========================================================
        # 🎨 HIGH-QUALITY MEL SPECTROGRAM
        # =========================================================

        mel_spec = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_mels=128,
            fmax=8000
        )

        mel_db = librosa.power_to_db(mel_spec, ref=np.max)

        # Normalize for image
        mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min())

        # Convert to RGB using colormap
        import matplotlib.cm as cm
        spec_img = cm.magma(mel_norm)

        spec_img = (spec_img[:, :, :3] * 255).astype(np.uint8)
        spec_img = cv2.resize(spec_img, (224, 224))

        # =========================================================
        # 🔥 SALIENCY → MATCH TIME AXIS
        # =========================================================

        # Convert 1D saliency → time bins
        saliency_time = np.interp(
            np.linspace(0, len(saliency), num=mel_spec.shape[1]),
            np.arange(len(saliency)),
            saliency
        )

        # Expand across frequency axis
        saliency_2d = np.tile(saliency_time, (mel_spec.shape[0], 1))

        # Normalize
        saliency_2d = (saliency_2d - saliency_2d.min()) / (saliency_2d.max() - saliency_2d.min() + 1e-8)

        # Resize
        saliency_resized = cv2.resize(saliency_2d, (224, 224))

        saliency_heatmap = cv2.applyColorMap(
            (saliency_resized * 255).astype(np.uint8),
            cv2.COLORMAP_JET
        )


        # =========================================================
        # 🔥 OVERLAY (THIS IS THE MAGIC)
        # =========================================================

        overlay = cv2.addWeighted(spec_img, 0.7, saliency_heatmap, 0.3, 0)

        return spec_img, overlay