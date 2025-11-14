# Quick test with strict=False
# Place in: backend/quick_test.py

import torch
from pathlib import Path
from models.image_detector import ImageDetector
from PIL import Image
import torchvision.transforms as transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}\n")

# Load model
detector = ImageDetector(backbone='xception', num_classes=2, device=device, pretrained=False)
checkpoint = torch.load('checkpoints/image/best_model.pth', map_location=device)

# Load with strict=False to ignore architecture mismatches
detector.load_state_dict(checkpoint['model_state_dict'], strict=False)
detector.eval()

print("✅ Model loaded with strict=False")
print(f"Best Val Accuracy: {checkpoint.get('best_val_acc', 'N/A')}")

# Test on random image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
])

# Red image
img = Image.new('RGB', (224, 224), color='red')
img_tensor = transform(img).unsqueeze(0).to(device)

with torch.no_grad():
    output = detector(img_tensor)
    prob = torch.softmax(output, dim=1)
    pred = torch.argmax(output, dim=1).item()

print(f"\nRed image test:")
print(f"  Prediction: {'FAKE' if pred == 1 else 'REAL'}")
print(f"  Confidence: {prob.max().item():.4f}")
print(f"  Probabilities: REAL={prob[0,0].item():.4f}, FAKE={prob[0,1].item():.4f}")
