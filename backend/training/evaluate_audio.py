# import os
# import torch
# from torch.utils.data import DataLoader
# from transformers import Wav2Vec2Processor
# from sklearn.metrics import classification_report, confusion_matrix

# from dataset import AudioDeepfakeDataset, collate_fn
# from model import Wav2Vec2Deepfake

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # 🔹 Processor
# processor = Wav2Vec2Processor.from_pretrained(
#     "facebook/wav2vec2-base"
# )

# # 🔹 Resolve dataset path correctly
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# AUDIO_DATASET_DIR = os.path.join(BASE_DIR, "dataset", "audio")

# # 🔹 Test dataset
# test_ds = AudioDeepfakeDataset(
#     os.path.join(AUDIO_DATASET_DIR, "test"),
#     processor
# )

# # ✅ THIS WAS MISSING
# test_loader = DataLoader(
#     test_ds,
#     batch_size=8,
#     shuffle=False,
#     collate_fn=lambda x: collate_fn(x, processor)
# )

# # 🔹 Load model
# model = Wav2Vec2Deepfake().to(device)
# model.load_state_dict(
#     torch.load("checkpoints/best_model.pth", map_location=device)
# )
# model.eval()

# preds = []
# labels = []

# with torch.no_grad():
#     for x, y in test_loader:
#         x = x.to(device)

#         logits = model(x)
#         probs = torch.sigmoid(logits)

#         preds.extend((probs > 0.5).cpu().numpy())
#         labels.extend(y.numpy())

# # 🔹 Metrics
# print("\nClassification Report:\n")
# print(classification_report(labels, preds, target_names=["Real", "Fake"]))

# print("Confusion Matrix:\n")
# print(confusion_matrix(labels, preds))


import os
import torch
from torch.utils.data import DataLoader
from transformers import Wav2Vec2Processor
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from training.audio_dataset import AudioDeepfakeDataset, collate_fn
from models.audio_detector import Wav2Vec2Deepfake

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 🔹 Load processor
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")

# 🔹 Dataset root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIO_DATASET_DIR = os.path.join(BASE_DIR, "dataset", "audio")

# 🔹 Load trained model
model = Wav2Vec2Deepfake().to(device)
checkpoint = torch.load("checkpoints/audio/best_model.pth", map_location=device)
model.load_state_dict(checkpoint, strict=False)
model.eval()


def evaluate_split(split_name):
    print(f"\n{'='*60}")
    print(f"Evaluating: {split_name.upper()} SET")
    print(f"{'='*60}")

    split_path = os.path.join(AUDIO_DATASET_DIR, split_name)

    if not os.path.exists(split_path):
        print(f"❌ {split_name} folder not found.")
        return None

    dataset = AudioDeepfakeDataset(split_path, processor)
    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,
        collate_fn=lambda x: collate_fn(x, processor)
    )

    preds = []
    labels = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            probs = torch.sigmoid(logits)

            predictions = (probs > 0.5).cpu().numpy()
            preds.extend(predictions)
            labels.extend(y.numpy())

    # Confusion Matrix
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()

    accuracy = accuracy_score(labels, preds)

    print("\nConfusion Matrix:")
    print(f"True Negatives  (REAL → REAL): {tn}")
    print(f"False Positives (REAL → FAKE): {fp}")
    print(f"False Negatives (FAKE → REAL): {fn}")
    print(f"True Positives  (FAKE → FAKE): {tp}")

    print(f"\nAccuracy: {accuracy:.4f}")

    print("\nDetailed Classification Report:")
    print(classification_report(labels, preds, target_names=["Real", "Fake"]))

    return accuracy


# 🔹 Evaluate all splits
splits = ["train", "validation", "test"]

accuracies = []

for split in splits:
    acc = evaluate_split(split)
    if acc is not None:
        accuracies.append(acc)

# 🔹 Overall Average Accuracy
if accuracies:
    print(f"\n{'='*60}")
    print(f"Overall Average Accuracy: {sum(accuracies)/len(accuracies):.4f}")
    print(f"{'='*60}")