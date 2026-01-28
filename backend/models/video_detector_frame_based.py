"""
✅ FRAME-BASED VIDEO DETECTOR
Loads trained frame classifier, extracts video frames, aggregates predictions
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from pathlib import Path
from typing import Dict, Any
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
import timm

logger = logging.getLogger(__name__)

class FrameBasedVideoDetector(nn.Module):
    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        num_frames: int = 8,
        frame_size: tuple = (224, 224),
    ):
        super().__init__()

        self.device = torch.device(device)
        self.num_frames = num_frames
        self.frame_size = frame_size

        # ✅ LOAD YOUR TRAINED IMAGE MODEL
        self.model = timm.create_model(
            "efficientnet_b0",
            pretrained=False,
            num_classes=2
        )

        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device)
        )

        self.model.to(self.device)
        self.model.eval()

        self.normalize = transforms.Compose([
            transforms.Resize(frame_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
        ])

        logger.info("✅ FrameBasedVideoDetector initialized")

    def extract_frames(self, video_path: str):
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total <= 0:
            cap.release()
            return None

        idxs = np.linspace(0, total - 1, self.num_frames, dtype=int)
        frames = []

        for idx in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)
                frames.append(self.normalize(frame))

        cap.release()
        return torch.stack(frames) if len(frames) == self.num_frames else None

    def predict(self, video_path: str):
        frames = self.extract_frames(video_path)
        if frames is None:
            return {
                "prediction": "ERROR",
                "confidence": 0.0,
                "status": "error"
            }

        frames = frames.to(self.device)

        with torch.no_grad():
            logits = self.model(frames)        # [T, 2]
            probs = torch.softmax(logits, dim=1)
            video_prob = probs.mean(dim=0)

        real_probs = probs[:, 1]
        fake_probs = probs[:, 0]

        # Averages
        avg_real = real_probs.mean().item()
        avg_fake = fake_probs.mean().item()

        fake_votes = (probs[:, 0] > 0.6).sum().item()
        real_votes = (probs[:, 1] > 0.6).sum().item()

        if fake_votes + real_votes == 0:
            confidence = probs[:, 0].mean().item()
            prediction = "FAKE" if confidence > 0.5 else "REAL"
        else:
            prediction = "FAKE" if fake_votes > real_votes else "REAL"
            confidence = max(fake_votes, real_votes) / self.num_frames
        return {
            "prediction": prediction,
            "confidence": float(confidence),
            "probabilities": {
                "FAKE": float(probs[:, 0].mean().item()),
                "REAL": float(probs[:, 1].mean().item()),
            },
            "frames_analyzed": self.num_frames,
            "frame_votes": {
                "FAKE": int(fake_votes),
                "REAL": int(real_votes)
            },
            "status": "success"
        }

