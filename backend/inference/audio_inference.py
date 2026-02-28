import torch
import soundfile as sf
import librosa
from transformers import Wav2Vec2Processor
from model import Wav2Vec2Deepfake

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load processor and model
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")

model = Wav2Vec2Deepfake().to(device)
model.load_state_dict(
    torch.load("../checkpoints/best_model.pt", map_location=device)
)
model.eval()


def predict(audio_path):
    # Load audio
    waveform, sr = sf.read(audio_path)
    waveform = torch.tensor(waveform, dtype=torch.float32)

    # Stereo → mono
    if waveform.ndim > 1:
        waveform = waveform.mean(dim=1)

    # Resample
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

    if prob > 0.5:
        label = "FAKE"
        confidence = prob*100
    else:
        label = "REAL"
        confidence = (1-prob)*100

    return label,round(confidence,2)


if __name__ == "__main__":
    audio_path=r"C:\Users\TEJASHREE\Music\PA_E_7138785.flac"
  # 👈 change this
    label, confidence = predict(audio_path)

    print(f"Prediction : {label}")
    print(f"Confidence : {confidence}%")
