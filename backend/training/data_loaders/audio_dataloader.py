"""
Audio Dataset Loader
Handles loading and preprocessing audio data for deepfake detection
"""

import os
import torch
import librosa
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset


def get_audio_transforms(augment=True, n_mels=128):
    """
    Get audio transforms (Mel-spectrogram)
    Args:
        augment: Whether to apply augmentation
        n_mels: Number of mel bins
    Returns:
        dict with transform parameters
    """
    return {
        'n_mels': n_mels,
        'n_fft': 2048,
        'hop_length': 512,
        'augment': augment
    }


class DeepfakeAudioDataset(Dataset):
    """
    Audio Dataset for Deepfake Detection
    Processes audio files into Mel-spectrograms
    """
    
    def __init__(self, root_dir, split='train', sample_rate=16000, 
                 duration=4.0, n_mels=128, augment=True):
        """
        Args:
            root_dir: Path to dataset root containing REAL/ and FAKE/ folders
            split: 'train', 'validation', or 'test'
            sample_rate: Audio sample rate
            duration: Audio clip duration in seconds
            n_mels: Number of mel bins
            augment: Whether to apply augmentation
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_mels = n_mels
        self.augment = augment
        
        self.audio_files = []
        self.labels = []
        self.class_names = ['REAL', 'FAKE']
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        
        self._load_audio_files()
    
    def _load_audio_files(self):
        """Load audio file paths and labels"""
        for class_name in self.class_names:
            class_dir = self.root_dir / class_name
            
            if not class_dir.exists():
                print(f"Warning: {class_dir} does not exist")
                continue
            
            label = self.class_to_idx[class_name]
            audio_extensions = ['.wav', '.mp3', '.m4a', '.flac']
            
            for audio_file in class_dir.iterdir():
                if audio_file.suffix.lower() in audio_extensions:
                    self.audio_files.append(str(audio_file))
                    self.labels.append(label)
        
        print(f"Loaded {len(self.audio_files)} audio files from {self.root_dir}")
    
    def _load_audio(self, audio_path):
        """Load and preprocess audio"""
        try:
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            # Trim/pad to fixed duration
            max_length = int(self.sample_rate * self.duration)
            if len(audio) > max_length:
                audio = audio[:max_length]
            else:
                audio = np.pad(audio, (0, max_length - len(audio)), 'constant')
            
            # Compute mel-spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=audio,
                sr=self.sample_rate,
                n_mels=self.n_mels,
                n_fft=2048,
                hop_length=512
            )
            
            # Convert to dB scale
            mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Normalize
            mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-8)
            
            return torch.FloatTensor(mel_spec)
        
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            return torch.zeros(self.n_mels, int(self.sample_rate * self.duration / 512))
    
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        """Get sample"""
        audio_path = self.audio_files[idx]
        label = self.labels[idx]
        
        mel_spec = self._load_audio(audio_path)
        
        return mel_spec, label


if __name__ == '__main__':
    dataset = DeepfakeAudioDataset(
        root_dir='../dataset/audio/train',
        split='train',
        augment=True
    )
    print(f"Dataset size: {len(dataset)}")
