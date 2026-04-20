import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader

class MultimodalHumorDataset(Dataset):
    def __init__(self, text_path, audio_path, video_path, csv_path):
        self.df = pd.read_csv(csv_path)
        self.text = np.load(text_path)
        self.audio = np.load(audio_path)
        self.video = np.load(video_path)
        
        # Basic checks
        print("Shapes:", self.text.shape, self.audio.shape, self.video.shape)
        print("Labels distribution:\n", self.df['Labels'].value_counts())
        
        assert len(self.df) == len(self.text) == len(self.audio) == len(self.video)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        return {
            'text': torch.tensor(self.text[idx], dtype=torch.float32),
            'audio': torch.tensor(self.audio[idx], dtype=torch.float32),
            'video': torch.tensor(self.video[idx], dtype=torch.float32),
            'label': torch.tensor(self.df['Labels'].iloc[idx], dtype=torch.long)
        }

# Usage
dataset = MultimodalHumorDataset(
    text_path=r"Features\Text_features_LaBSE.npy",
    audio_path=r"Features\Audio_feature.npy",
    video_path=r"Features\video_Feature.npy",
    csv_path=r"Dataset.csv"
)

train_idx, test_idx = train_test_split(range(len(dataset)), test_size=0.2, stratify=dataset.df['Labels'], random_state=42)
# You can further split train into train/val