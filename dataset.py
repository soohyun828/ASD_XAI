import os
import json
import glob
import torch
from torch.utils.data import Dataset
from PIL import Image

class HumanFigureDataset(Dataset):
    def __init__(self, base_dir, transform=None):
        self.image_paths = []
        self.annotation_paths = []
        self.labels = []
        self.transform = transform
        self.body_parts = set()
        
        for label, sub_dir in enumerate(['ASD', 'TD']):
            image_files = glob.glob(os.path.join(base_dir, sub_dir, "*.jpg"))
            for image_file in image_files:
                annotation_file = image_file.replace('.jpg', '.json')
                if os.path.exists(annotation_file):
                    self.image_paths.append(image_file)
                    self.annotation_paths.append(annotation_file)
                    self.labels.append(label)
                    self._extract_body_parts(annotation_file)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        annotation_path = self.annotation_paths[idx]
        label = self.labels[idx]
        
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        
        with open(annotation_path, 'r') as f:
            annotation = json.load(f)
        
        total_area = annotation["imageWidth"] * annotation["imageHeight"]
        proportions = {part: 0 for part in self.body_parts}
        
        for shape in annotation["shapes"]:
            part = shape["label"]
            if part in proportions:
                points = shape["points"]
                width = points[1][0] - points[0][0]
                height = points[1][1] - points[0][1]
                part_area = width * height
                proportions[part] += part_area
        
        proportions_vector = [proportions[part] / total_area for part in self.body_parts]
        
        return image, torch.tensor(proportions_vector, dtype=torch.float32), label
    
    def _extract_body_parts(self, annotation_file):
        with open(annotation_file, 'r') as f:
            annotation = json.load(f)
            for shape in annotation["shapes"]:
                self.body_parts.add(shape["label"])
