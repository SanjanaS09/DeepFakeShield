import torchvision.transforms as transforms
from torchvision.transforms import RandomHorizontalFlip, RandomRotation, ColorJitter, GaussianBlur

class ImageAugmentationPipeline:
    def __init__(self):
        self.augmentations = transforms.Compose([
            RandomHorizontalFlip(p=0.5),
            RandomRotation(degrees=15),
            ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0))
        ])

    def __call__(self, image):
        return self.augmentations(image)
