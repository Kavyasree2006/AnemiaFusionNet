import os
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset

class FusionDataset(Dataset):

    def __init__(self, csv_file, image_dir, transform=None):

        self.data = pd.read_csv(csv_file)

        self.image_dir = image_dir

        self.transform = transform

        self.clinical_cols = [
            "age",
            "gender",
            "hb",
            "bmi",
            "fatigue",
            "diet",
            "region"
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        row = self.data.iloc[idx]

        image_path = os.path.join(
            self.image_dir,
            row["image"]
        )

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        clinical = torch.tensor(
            row[self.clinical_cols].values.astype(float),
            dtype=torch.float32
        )

        geo = torch.tensor(
            [row["geo_risk"]],
            dtype=torch.float32
        )

        label = torch.tensor(
            row["label"],
            dtype=torch.float32
        )

        return image, clinical, geo, label