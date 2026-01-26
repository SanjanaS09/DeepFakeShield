import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import timm
from pathlib import Path
from PIL import Image

# =========================
# CONFIG
# =========================
DATASET_DIR = "/content/drive/MyDrive/DeepFakeShield/backend/dataset/video_frames"
EPOCHS = 12
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
NUM_WORKERS = 2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# IMAGE VALIDATION
# =========================
def is_valid_image(path):
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except:
        return False

class SafeImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None):
        super().__init__(root, transform)

        original_count = len(self.samples)
        self.samples = [s for s in self.samples if is_valid_image(s[0])]
        filtered_count = len(self.samples)

        print(
            f"📂 Loaded {filtered_count}/{original_count} valid images from {root}"
        )

# =========================
# TRANSFORMS
# =========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =========================
# DATASETS
# =========================
train_dataset = SafeImageFolder(
    root=f"{DATASET_DIR}/train",
    transform=transform
)

val_dataset = SafeImageFolder(
    root=f"{DATASET_DIR}/validation",
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

# =========================
# MODEL
# =========================
model = timm.create_model(
    "efficientnet_b0",
    pretrained=True,
    num_classes=2
)
model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# =========================
# CHECKPOINT DIR
# =========================
ckpt_dir = Path("checkpoints/video_frame")
ckpt_dir.mkdir(parents=True, exist_ok=True)

best_acc = 0.0

# =========================
# TRAINING LOOP
# =========================
for epoch in range(1, EPOCHS + 1):
    # ---- TRAIN ----
    model.train()
    correct = 0
    total = 0
    running_loss = 0.0

    for images, labels in train_loader:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_acc = 100.0 * correct / total
    train_loss = running_loss / len(train_loader)

    # ---- VALIDATION ----
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_acc = 100.0 * correct / total

    print(
        f"Epoch {epoch:02d}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_acc:.2f}% | "
        f"Val Acc: {val_acc:.2f}%"
    )

    # ---- SAVE BEST MODEL ----
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(
            model.state_dict(),
            ckpt_dir / "best_image_model.pth"
        )
        print("✅ Saved new best model")

# =========================
# DONE
# =========================
print("\n🎯 Training complete")
print(f"🏆 Best Validation Accuracy: {best_acc:.2f}%")
