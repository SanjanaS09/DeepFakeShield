import torch
import soundfile as sf
import librosa
import os
from transformers import Wav2Vec2Processor
from .model import Wav2Vec2Deepfake

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get correct model path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best_model.pt")

# Load processor and model
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")

model = Wav2Vec2Deepfake().to(device)
model.load_state_dict(
    torch.load(MODEL_PATH, map_location=device)
)
model.eval()


def predict_audio(audio_path):
    # Load audio
    waveform, sr = sf.read(audio_path)
    waveform = torch.tensor(waveform, dtype=torch.float32)

    # Stereo to mono
    if waveform.ndim > 1:
        waveform = waveform.mean(dim=1)

    # Resample to 16kHz
    if sr != 16000:
        waveform = torch.from_numpy(
            librosa.resample(
                waveform.numpy(),
                orig_sr=sr,
                target_sr=16000
            )
        )

    # Prepare input
    inputs = processor(
        waveform.numpy(),
        sampling_rate=16000,
        return_tensors="pt",
        padding=True
    ).input_values.to(device)

    # Inference
    with torch.no_grad():
        logit = model(inputs)
        prob = torch.sigmoid(logit).item()

    # Label and confidence
    if prob > 0.5:
        label = "FAKE"
        confidence = prob * 100
    else:
        label = "REAL"
        confidence = (1 - prob) * 100

    return label, round(confidence, 2)
