# """
# Audio Deepfake Detector
# Simplified audio detection model
# """

# import torch
# import torch.nn as nn
# import logging
# from pathlib import Path
# from typing import Dict, Any, Optional, Tuple

# logger = logging.getLogger(__name__)

# class SimpleAudioDetector(nn.Module):
#     """✅ SIMPLE AUDIO DETECTOR MODEL"""
    
#     def __init__(self, num_classes: int = 2, input_size: int = 128):
#         super().__init__()
        
#         self.fc = nn.Sequential(
#             nn.Linear(input_size, 256),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.3),
#             nn.Linear(256, 128),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.2),
#             nn.Linear(128, num_classes)
#         )
    
#     def forward(self, x):
#         """
#         Forward pass
#         Input: [B, input_size]
#         """
#         if x.dim() > 2:
#             x = x.view(x.size(0), -1)
#         return self.fc(x)


# class AudioDetector(nn.Module):
#     """✅ AUDIO DEEPFAKE DETECTOR"""
    
#     def __init__(self,
#                  backbone: str = 'ecapa-tdnn',
#                  num_classes: int = 2,
#                  device: str = 'cpu',
#                  model_path: Optional[str] = None,
#                  pretrained: bool = True):
#         """
#         Initialize Audio Detector
        
#         Args:
#             backbone: Model architecture (simplified to 'simple')
#             num_classes: Number of classes
#             device: Device for computation
#             model_path: Path to trained model
#             pretrained: Use pretrained weights
#         """
#         super().__init__()
        
#         self.num_classes = num_classes
#         self.device = torch.device(device)
        
#         # ✅ Use simple model instead of ECAPA-TDNN
#         logger.info("Building SimpleAudioDetector...")
#         self.model = SimpleAudioDetector(num_classes=num_classes, input_size=128)
        
#         self.model.to(self.device)
        
#         # Load checkpoint if provided
#         if model_path and Path(model_path).exists():
#             try:
#                 logger.info(f"Loading audio checkpoint from {model_path}")
#                 checkpoint = torch.load(model_path, map_location=self.device)
                
#                 if isinstance(checkpoint, dict):
#                     if 'model_state_dict' in checkpoint:
#                         self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
#                     else:
#                         self.model.load_state_dict(checkpoint, strict=False)
#                 else:
#                     self.model.load_state_dict(checkpoint, strict=False)
                
#                 logger.info("✅ Audio model checkpoint loaded")
            
#             except Exception as e:
#                 logger.warning(f"Could not load audio checkpoint: {e}")
        
#         self.model.eval()
#         logger.info("✅ AudioDetector initialized")
    
#     def forward(self, x):
#         """Forward pass"""
#         return self.model(x)
    
#     def predict(self, audio_tensor: torch.Tensor) -> Dict[str, Any]:
#         """
#         Predict deepfake in audio
        
#         Args:
#             audio_tensor: Audio features [B, feature_dim] or [feature_dim]
        
#         Returns:
#             Detection results
#         """
#         try:
#             if audio_tensor.dim() == 1:
#                 audio_tensor = audio_tensor.unsqueeze(0)
            
#             # Pad/truncate to expected size
#             if audio_tensor.shape[1] != 128:
#                 if audio_tensor.shape[1] < 128:
#                     padding = torch.zeros(audio_tensor.shape[0], 128 - audio_tensor.shape[1])
#                     audio_tensor = torch.cat([audio_tensor, padding], dim=1)
#                 else:
#                     audio_tensor = audio_tensor[:, :128]
            
#             audio_tensor = audio_tensor.to(self.device)
            
#             with torch.no_grad():
#                 logits = self.forward(audio_tensor)
#                 probabilities = torch.softmax(logits, dim=1)
#                 confidence, predicted_class = torch.max(probabilities, dim=1)
            
#             return {
#                 'prediction': 'FAKE' if predicted_class.item() == 1 else 'REAL',
#                 'confidence': float(confidence.item()),
#                 'probabilities': {
#                     'REAL': float(probabilities[0, 0].item()),
#                     'FAKE': float(probabilities[0, 1].item())
#                 }
#             }
        
#         except Exception as e:
#             logger.error(f"Audio prediction error: {e}")
#             return {
#                 'prediction': 'ERROR',
#                 'confidence': 0.0,
#                 'error': str(e)
#             }
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