import torch
import numpy as np
from pathlib import Path

checkpoint_path = Path("checkpoints/video/best_model.pth")

if checkpoint_path.exists():
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    print("\n" + "="*70)
    print("📊 VIDEO MODEL CHECKPOINT ANALYSIS")
    print("="*70)
    
    if isinstance(checkpoint, dict):
        print(f"\n🔑 Keys in checkpoint:")
        for key in checkpoint.keys():
            print(f"   • {key}")
        
        if 'best_acc' in checkpoint:
            print(f"\n✅ Best Accuracy: {checkpoint['best_acc']:.2%}")
        
        if 'epoch' in checkpoint:
            print(f"✅ Trained Epochs: {checkpoint['epoch']}")
        
        if 'model_state_dict' in checkpoint:
            state = checkpoint['model_state_dict']
            print(f"\n📦 Model Parameters:")
            param_count = sum(p.numel() for p in state.values())
            print(f"   Total: {param_count:,}")
        
        print(f"\n⚠️  Current model expects:")
        print(f"   Input: [Batch, Frames=8, Channels=3, H=224, W=224]")
        print(f"   Output: [Batch, 2] (REAL/FAKE)")
    else:
        print("❌ Checkpoint is NOT a dict - cannot analyze")
else:
    print(f"❌ Checkpoint not found: {checkpoint_path}")

print("\n" + "="*70)
