# # Quick test with strict=False
# # Place in: backend/quick_test.py

# import torch
# from pathlib import Path
# from models.image_detector import ImageDetector
# from PIL import Image
# import torchvision.transforms as transforms

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f"Device: {device}\n")

# # Load model
# detector = ImageDetector(backbone='xception', num_classes=2, device=device, pretrained=False)
# checkpoint = torch.load('checkpoints/image/best_model.pth', map_location=device)

# # Load with strict=False to ignore architecture mismatches
# detector.load_state_dict(checkpoint['model_state_dict'], strict=False)
# detector.eval()

# print("✅ Model loaded with strict=False")
# print(f"Best Val Accuracy: {checkpoint.get('best_val_acc', 'N/A')}")

# # Test on random image
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                        std=[0.229, 0.224, 0.225])
# ])

# # Red image
# img = Image.new('RGB', (224, 224), color='red')
# img_tensor = transform(img).unsqueeze(0).to(device)

# with torch.no_grad():
#     output = detector(img_tensor)
#     prob = torch.softmax(output, dim=1)
#     pred = torch.argmax(output, dim=1).item()

# print(f"\nRed image test:")
# print(f"  Prediction: {'FAKE' if pred == 1 else 'REAL'}")
# print(f"  Confidence: {prob.max().item():.4f}")
# print(f"  Probabilities: REAL={prob[0,0].item():.4f}, FAKE={prob[0,1].item():.4f}")


#!/usr/bin/env python3
"""
Quick test script for ResNet18 image model
Tests model loading and basic inference
"""

import torch
import torchvision.models as models
import torch.nn as nn
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ========== ResNet18 Model Definition ==========
class ResNet18DeepfakeDetector(nn.Module):
    """ResNet18 model for deepfake detection (must match training)"""
    def __init__(self, num_classes=2, pretrained=False):
        super().__init__()
        self.backbone = models.resnet18(weights=None)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

# ========== Load Model ==========
print("\n" + "="*60)
print("LOADING RESNET18 MODEL")
print("="*60)

try:
    # Create model
    detector = ResNet18DeepfakeDetector(num_classes=2, pretrained=False)
    detector = detector.to(device)
    
    # Load checkpoint
    checkpoint_path = Path('checkpoints/image/best_model.pth')
    
    if not checkpoint_path.exists():
        print(f"❌ Model not found at: {checkpoint_path}")
        print("   Make sure training completed successfully!")
        exit(1)
    
    print(f"Loading model from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load state dict
    if 'model_state_dict' in checkpoint:
        detector.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Model loaded from epoch {checkpoint.get('epoch', 'N/A')}")
        print(f"✓ Best validation accuracy: {checkpoint.get('best_val_acc', 'N/A'):.2f}%")
    else:
        detector.load_state_dict(checkpoint)
        print("✓ Model loaded (state dict only)")
    
    detector.eval()
    print("✓ Model set to evaluation mode")
    
except Exception as e:
    print(f"❌ Error loading model: {str(e)}")
    exit(1)

# ========== Prepare Transform ==========
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ========== Test on Real Test Image ==========
print("\n" + "="*60)
print("TESTING ON REAL IMAGES")
print("="*60)

test_images = [
    Path('dataset/image/test/REAL'),
    Path('dataset/image/test/FAKE')
]

for img_dir in test_images:
    if not img_dir.exists():
        print(f"⚠️  Directory not found: {img_dir}")
        continue
    
    label = img_dir.name  # 'REAL' or 'FAKE'
    images = list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png'))
    
    if not images:
        print(f"⚠️  No images found in {img_dir}")
        continue
    
    # Test first 3 images
    for img_path in images[:3]:
        try:
            # Load and preprocess
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0).to(device)
            
            # Run inference
            with torch.no_grad():
                output = detector(img_tensor)
                prob = torch.softmax(output, dim=1)
                pred = torch.argmax(output, dim=1).item()
            
            # Results
            pred_label = 'FAKE' if pred == 1 else 'REAL'
            confidence = prob.max().item()
            fake_prob = prob[0, 1].item()
            real_prob = prob[0, 0].item()
            
            # Check if correct
            correct = "✓" if pred_label == label else "✗"
            
            print(f"\n{correct} Image: {img_path.name}")
            print(f"  Ground Truth: {label}")
            print(f"  Prediction:   {pred_label}")
            print(f"  Confidence:   {confidence:.4f}")
            print(f"  Probabilities: REAL={real_prob:.4f}, FAKE={fake_prob:.4f}")
            
        except Exception as e:
            print(f"❌ Error processing {img_path.name}: {str(e)}")

# ========== Test on Synthetic Test Image ==========
print("\n" + "="*60)
print("TESTING ON SYNTHETIC IMAGE")
print("="*60)

try:
    # Create a red test image
    img = Image.new('RGB', (224, 224), color='red')
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = detector(img_tensor)
        prob = torch.softmax(output, dim=1)
        pred = torch.argmax(output, dim=1).item()
    
    print(f"\nSynthetic Image Test:")
    print(f"  Prediction:   {'FAKE' if pred == 1 else 'REAL'}")
    print(f"  Confidence:   {prob.max().item():.4f}")
    print(f"  Probabilities: REAL={prob[0, 0].item():.4f}, FAKE={prob[0, 1].item():.4f}")
    
except Exception as e:
    print(f"❌ Error: {str(e)}")

print("\n" + "="*60)
print("TEST COMPLETE")
print("="*60)
