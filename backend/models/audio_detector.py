import torch
import torch.nn as nn
import torchaudio
from pathlib import Path
from transformers import Wav2Vec2Processor
from models.wav2vec2_audio_model import Wav2Vec2Deepfake

class AudioDeepfakeDetector:
    def __init__(self, model_path: str, device='cpu'):
        self.device = torch.device(device)

        self.processor = Wav2Vec2Processor.from_pretrained(
            "facebook/wav2vec2-base"
        )

        # Load model
        self.model = Wav2Vec2Deepfake()
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint, strict=False)
        self.model.to(self.device)
        self.model.eval()

    def preprocess(self, audio_path):
        waveform, sample_rate = torchaudio.load(audio_path)

        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate,
                new_freq=16000
            )
            waveform = resampler(waveform)

        return waveform.squeeze(0)

    def detect(self, audio_path):
        import soundfile as sf
        import librosa

        waveform, sr = sf.read(audio_path)

        if len(waveform.shape) > 1:
            waveform = waveform.mean(axis=1)

        if sr != 16000:
            waveform = librosa.resample(
                waveform,
                orig_sr=sr,
                target_sr=16000
            )

        inputs = self.processor(
            waveform,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        ).input_values.to(self.device)

        with torch.no_grad():
            logits = self.model(inputs)
            prob = torch.sigmoid(logits).item()

        # 🔥 Calibrated threshold
        THRESHOLD = 0.65   # You can tune this after evaluation

        prediction = "FAKE" if prob > THRESHOLD else "REAL"

        confidence = prob if prediction == "FAKE" else 1 - prob

        return {
            "prediction": prediction,
            "confidence": confidence,
            "raw_probability": float(prob),
            "threshold": THRESHOLD,
            "probabilities": {
                "REAL": 1 - prob,
                "FAKE": prob
            },
            "status": "success"
        }