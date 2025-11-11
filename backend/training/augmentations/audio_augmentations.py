"""
Audio Augmentation Pipeline
Augmentation techniques for audio deepfake detection
"""

import torch
import numpy as np
import librosa
import random


class AudioAugmentationPipeline:
    """
    Audio augmentation pipeline using spectral and time-domain techniques
    """
    
    def __init__(self, sample_rate=16000, augmentation_config=None):
        """
        Args:
            sample_rate: Audio sample rate
            augmentation_config: Dict with augmentation parameters
        """
        self.sample_rate = sample_rate
        self.config = augmentation_config or {}
    
    def __call__(self, audio_spec, is_training=True):
        """
        Apply augmentation to audio spectrogram
        Args:
            audio_spec: Mel-spectrogram tensor
            is_training: Whether to apply training augmentations
        """
        if not is_training:
            return audio_spec
        
        # Apply random augmentations with probability
        if random.random() < 0.5:
            audio_spec = self.apply_time_masking(audio_spec)
        
        if random.random() < 0.5:
            audio_spec = self.apply_frequency_masking(audio_spec)
        
        if random.random() < 0.3:
            audio_spec = self.apply_noise(audio_spec)
        
        return audio_spec
    
    def apply_time_masking(self, spec, max_mask_width=10):
        """
        Apply time masking (SpecAugment)
        Mask random time steps
        """
        spec = spec.clone() if isinstance(spec, torch.Tensor) else torch.tensor(spec)
        
        if spec.dim() >= 1:
            time_axis = spec.shape[-1]
            mask_width = random.randint(1, min(max_mask_width, time_axis // 10))
            start_pos = random.randint(0, max(0, time_axis - mask_width))
            spec[..., start_pos:start_pos + mask_width] = 0
        
        return spec
    
    def apply_frequency_masking(self, spec, max_mask_width=5):
        """
        Apply frequency masking (SpecAugment)
        Mask random frequency bins
        """
        spec = spec.clone() if isinstance(spec, torch.Tensor) else torch.tensor(spec)
        
        if spec.dim() >= 1:
            freq_axis = spec.shape[-2] if spec.dim() >= 2 else 1
            mask_width = random.randint(1, min(max_mask_width, freq_axis // 10))
            start_pos = random.randint(0, max(0, freq_axis - mask_width))
            spec[start_pos:start_pos + mask_width, :] = 0
        
        return spec
    
    def apply_noise(self, spec, noise_factor=0.01):
        """Add Gaussian noise"""
        noise = torch.randn_like(spec) * noise_factor
        return spec + noise
    
    def apply_pitch_shift(self, audio, n_steps=2):
        """Pitch shift augmentation"""
        if isinstance(audio, torch.Tensor):
            audio = audio.numpy()
        
        shifted_audio = librosa.effects.pitch_shift(
            audio,
            sr=self.sample_rate,
            n_steps=n_steps
        )
        
        return torch.tensor(shifted_audio, dtype=torch.float32)
    
    def apply_time_stretch(self, audio, rate=1.1):
        """Time stretching augmentation"""
        if isinstance(audio, torch.Tensor):
            audio = audio.numpy()
        
        stretched_audio = librosa.effects.time_stretch(audio, rate=rate)
        
        return torch.tensor(stretched_audio, dtype=torch.float32)


if __name__ == '__main__':
    pipeline = AudioAugmentationPipeline()
    print("AudioAugmentationPipeline initialized successfully")
