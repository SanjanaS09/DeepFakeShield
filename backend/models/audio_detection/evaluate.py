import os
import torch
from torch.utils.data import DataLoader
from transformers import Wav2Vec2Processor
from sklearn.metrics import classification_report, confusion_matrix

from dataset import AudioDeepfakeDataset, collate_fn
from model import Wav2Vec2Deepfake

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 🔹 Processor
processor = Wav2Vec2Processor.from_pretrained(
    "facebook/wav2vec2-base"
)

# 🔹 Resolve dataset path correctly
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIO_DATASET_DIR = os.path.join(BASE_DIR, "dataset", "audio")

# 🔹 Test dataset
test_ds = AudioDeepfakeDataset(
    os.path.join(AUDIO_DATASET_DIR, "test"),
    processor
)

# ✅ THIS WAS MISSING
test_loader = DataLoader(
    test_ds,
    batch_size=8,
    shuffle=False,
    collate_fn=lambda x: collate_fn(x, processor)
)

# 🔹 Load model
model = Wav2Vec2Deepfake().to(device)
model.load_state_dict(
    torch.load("checkpoints/best_model.pt", map_location=device)
)
model.eval()

preds = []
labels = []

with torch.no_grad():
    for x, y in test_loader:
        x = x.to(device)

        logits = model(x)
        probs = torch.sigmoid(logits)

        preds.extend((probs > 0.5).cpu().numpy())
        labels.extend(y.numpy())

# 🔹 Metrics
print("\nClassification Report:\n")
print(classification_report(labels, preds, target_names=["Real", "Fake"]))

print("Confusion Matrix:\n")
print(confusion_matrix(labels, preds))
