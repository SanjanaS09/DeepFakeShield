# backend/preprocessing/image_dataloader.py
import os
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class DeepfakeImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        for label in ['real', 'fake']:
            folder = os.path.join(root_dir, label)
            for img_name in os.listdir(folder):
                self.image_paths.append(os.path.join(folder, img_name))
                self.labels.append(0 if label == 'real' else 1)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


def get_image_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
