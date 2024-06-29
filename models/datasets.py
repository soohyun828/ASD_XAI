#dataset.py
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class HumanFigureDataset(Dataset):
    def __init__(self, img_dir, csv_file, transform=None):
        self.img_dir = img_dir
        self.annotations = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.annotations.iloc[idx, 1], self.annotations.iloc[idx, 0] + '.jpg')
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        part_proportions = self.annotations.iloc[idx, 2:].values.astype(float)
        label = 0 if self.annotations.iloc[idx, 1] == 'ASD' else 1 # ASD면 1
        
        return image, part_proportions, label,self.annotations.iloc[idx, 0]
