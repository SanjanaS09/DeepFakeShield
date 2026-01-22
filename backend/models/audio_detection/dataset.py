import os
import torch
import soundfile as sf
import librosa
from torch.utils.data import Dataset


class AudioDeepfakeDataset(Dataset):
    def __init__(self, root_dir, processor, max_length=16000 * 5):
        self.samples = []
        self.processor = processor
        self.max_length = max_length

        for label, cls in enumerate(["real", "fake"]):
            cls_path = os.path.join(root_dir, cls)

            if not os.path.exists(cls_path):
                raise FileNotFoundError(f"Missing folder: {cls_path}")

            for file in os.listdir(cls_path):
                if file.endswith((".wav", ".flac")):
                    self.samples.append(
                        (os.path.join(cls_path, file), label)
                    )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        # 🔹 Load audio using soundfile (NO torchaudio)
        waveform, sr = sf.read(path)

        # convert to torch tensor
        waveform = torch.tensor(waveform, dtype=torch.float32)

        # stereo → mono
        if waveform.ndim > 1:
            waveform = waveform.mean(dim=1)

        # resample to 16kHz
        if sr != 16000:
            waveform = torch.from_numpy(
                librosa.resample(
                    waveform.numpy(),
                    orig_sr=sr,
                    target_sr=16000
                )
            )

        # truncate long audio
        if waveform.shape[0] > self.max_length:
            waveform = waveform[:self.max_length]

        return waveform, label


def collate_fn(batch, processor):
    waveforms, labels = zip(*batch)

    # 🔹 Convert torch tensors → numpy arrays
    waveforms = [w.numpy() for w in waveforms]

    inputs = processor(
        waveforms,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True
    )

    return inputs.input_values, torch.tensor(labels)
