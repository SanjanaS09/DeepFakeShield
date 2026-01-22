import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
from transformers import Wav2Vec2Model

class Wav2Vec2Deepfake(nn.Module):
    def __init__(self):
        super().__init__()

        self.wav2vec = Wav2Vec2Model.from_pretrained(
            "facebook/wav2vec2-base"
        )

        self.classifier = nn.Linear(
            self.wav2vec.config.hidden_size, 1
        )

    def forward(self, input_values):
        outputs = self.wav2vec(input_values)
        hidden_states = outputs.last_hidden_state

        pooled = hidden_states.mean(dim=1)

        logits = self.classifier(pooled)
        return logits.squeeze()
