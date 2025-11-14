# Test Image Model Directly
# Place in: backend/test_image_model.py

import torch
import sys
from pathlib import Path

# Add backend to path
BACKEND_ROOT = Path(__file__).parent
sys.path.insert(0, str(BACKEND_ROOT))

from models.image_detector import ImageDetector
from PIL import Image
import torchvision.transforms as transforms

def test_model():
    """Test if model loads and works"""
    
    print("=" * 60)
    print("TESTING IMAGE MODEL")
    print("=" * 60)
    
    # 1. Check if model file exists
    model_path = "checkpoints/image/best_model.pth"
    if not Path(model_path).exists():
        print(f"❌ ERROR: Model file not found at: {model_path}")
        print(f"   Current directory: {Path.cwd()}")
        print(f"   Files in checkpoints/image/:")
        if Path("checkpoints/image").exists():
            for f in Path("checkpoints/image").iterdir():
                print(f"     - {f.name}")
        else:
            print("     Directory doesn't exist!")
        return
    
    print(f"✅ Model file found: {model_path}")
    print(f"   Size: {Path(model_path).stat().st_size / (1024*1024):.2f} MB")
    
    # 2. Load model
    print("\n" + "=" * 60)
    print("Loading model...")
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # Initialize detector
        detector = ImageDetector(
            backbone='xception',
            num_classes=2,
            device=device,
            pretrained=False  # Don't load pretrained, we're loading our trained model
        )
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        
        # Check what's in the checkpoint
        print(f"\nCheckpoint keys: {checkpoint.keys()}")
        
        if 'model_state_dict' in checkpoint:
            detector.load_state_dict(checkpoint['model_state_dict'])
            print("✅ Loaded from 'model_state_dict'")
        elif 'state_dict' in checkpoint:
            detector.load_state_dict(checkpoint['state_dict'])
            print("✅ Loaded from 'state_dict'")
        else:
            detector.load_state_dict(checkpoint)
            print("✅ Loaded directly")
        
        detector.eval()
        print("✅ Model loaded successfully!")
        
        # Print checkpoint metadata
        if 'epoch' in checkpoint:
            print(f"   Epoch: {checkpoint['epoch']}")
        if 'val_acc' in checkpoint:
            print(f"   Val Accuracy: {checkpoint['val_acc']:.2f}%")
        if 'best_val_acc' in checkpoint:
            print(f"   Best Val Acc: {checkpoint['best_val_acc']:.2f}%")
            
    except Exception as e:
        print(f"❌ ERROR loading model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 3. Test with dummy image
    print("\n" + "=" * 60)
    print("Testing prediction on dummy image...")
    try:
        # Create dummy image
        dummy_image = Image.new('RGB', (224, 224), color='red')
        
        # Transform
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        img_tensor = transform(dummy_image).unsqueeze(0).to(device)
        
        # Predict
        with torch.no_grad():
            outputs = detector(img_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            prediction = torch.argmax(outputs, dim=1)
            
        print(f"✅ Prediction successful!")
        print(f"   Raw outputs: {outputs.cpu().numpy()}")
        print(f"   Probabilities: {probabilities.cpu().numpy()}")
        print(f"   Prediction: {'FAKE' if prediction.item() == 1 else 'REAL'}")
        print(f"   Confidence: {probabilities.max().item():.4f}")
        
    except Exception as e:
        print(f"❌ ERROR during prediction: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 4. Test on actual test images
    print("\n" + "=" * 60)
    print("Testing on actual test images...")
    
    test_dir = Path("dataset/image/test")
    if not test_dir.exists():
        print(f"⚠️  Test directory not found: {test_dir}")
        print("=" * 60)
        return
    
    # Test on REAL images
    real_dir = test_dir / "REAL"
    if real_dir.exists():
        real_images = list(real_dir.glob("*.jpg"))[:3]  # Test first 3
        print(f"\nTesting on {len(real_images)} REAL images:")
        
        for img_path in real_images:
            try:
                img = Image.open(img_path).convert('RGB')
                img_tensor = transform(img).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    outputs = detector(img_tensor)
                    probs = torch.softmax(outputs, dim=1)
                    pred = torch.argmax(outputs, dim=1)
                
                pred_label = "FAKE" if pred.item() == 1 else "REAL"
                confidence = probs.max().item()
                
                correct = "✅" if pred_label == "REAL" else "❌"
                print(f"  {correct} {img_path.name[:30]}: {pred_label} ({confidence:.2f})")
                
            except Exception as e:
                print(f"  ❌ Error: {e}")
    
    # Test on FAKE images
    fake_dir = test_dir / "FAKE"
    if fake_dir.exists():
        fake_images = list(fake_dir.glob("*.jpg"))[:3]  # Test first 3
        print(f"\nTesting on {len(fake_images)} FAKE images:")
        
        for img_path in fake_images:
            try:
                img = Image.open(img_path).convert('RGB')
                img_tensor = transform(img).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    outputs = detector(img_tensor)
                    probs = torch.softmax(outputs, dim=1)
                    pred = torch.argmax(outputs, dim=1)
                
                pred_label = "FAKE" if pred.item() == 1 else "REAL"
                confidence = probs.max().item()
                
                correct = "✅" if pred_label == "FAKE" else "❌"
                print(f"  {correct} {img_path.name[:30]}: {pred_label} ({confidence:.2f})")
                
            except Exception as e:
                print(f"  ❌ Error: {e}")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE!")
    print("=" * 60)

if __name__ == "__main__":
    test_model()
