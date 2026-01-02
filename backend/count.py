import os
from pathlib import Path

dataset_root = Path('dataset/image')

for split in ['train', 'validation', 'test']:
    real_count = len(list((dataset_root / split / 'REAL').glob('*')))
    fake_count = len(list((dataset_root / split / 'FAKE').glob('*')))
    print(f"{split}: {real_count} REAL, {fake_count} FAKE")
