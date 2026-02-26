import cv2
import torch
import numpy as np

def preprocess_image(image, size=(224,224)):
    image = cv2.resize(image, size)
    image = image / 255.0
    tensor = torch.tensor(image).permute(2,0,1).unsqueeze(0).float()
    return tensor

def normalize_tensor(tensor):
    return (tensor - tensor.mean()) / tensor.std()